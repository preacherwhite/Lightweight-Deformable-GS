# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import math
import random
import logging
import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
from torch import nn
import torch
import torchcde
from physiopro.network.contiformer import AttrDict, EncoderLayer

matplotlib.use('agg')


def get_logger(name):
    logger = logging.getLogger(name)
    filename = f'{name}.log'
    fh = logging.FileHandler(filename, mode='a+', encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default='./spiral_contiformer')
parser.add_argument('--model_name', type=str, default='Neural_ODE',
                    choices=['Neural_ODE', 'Contiformer'])
parser.add_argument('--log_step', type=int, default=5)
parser.add_argument('--seed', type=int, default=27)
parser.add_argument('--noise_std', type=float, default=0.05)
parser.add_argument('--noise_a', type=float, default=0)
parser.add_argument('--cc', type=eval, default=True)

# parameters for ContiFormer
parser.add_argument('--atol', type=float, default=0.1)
parser.add_argument('--rtol', type=float, default=0.1)
parser.add_argument('--method', type=str, default='dopri5')
parser.add_argument('--dropout', type=float, default=0)

# ---- NEW (for scaling up) ----
parser.add_argument('--batch_size', type=int, default=200,
                    help="Batch size for training")
parser.add_argument('--latent_dim', type=int, default=8,
                    help="Dimension of latent space in Neural ODE")
parser.add_argument('--nhidden', type=int, default=16,
                    help="Hidden layer size in Neural ODE components")
parser.add_argument('--nspiral', type=int, default=300,
                    help="Number of spirals (essentially dataset size multiplier)")
parser.add_argument('--ntrain', type=int, default=200,
                    help="How many sequences to use for training")
parser.add_argument('--ntest', type=int, default=100,
                    help="How many sequences to use for testing")
# -------------------------------

args = parser.parse_args()

if not os.path.exists(args.train_dir):
    os.makedirs(args.train_dir)

log = get_logger(os.path.join(args.train_dir, 'log'))

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def generate_spiral2d(nspiral=1000,
                      ntotal=500,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      noise_a=.002,
                      a=0.,
                      b=1.):
    """Parametric formula for 2d spiral is `r = a + b * theta`.
    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
    Returns:
      Tuple (orig_trajs, samp_trajs, orig_ts)
    """
    orig_ts = np.linspace(start, stop, num=ntotal)  # [ntotal]
    aa = npr.randn(nspiral) * noise_a + a  # [nspiral]
    bb = npr.randn(nspiral) * noise_a + b  # [nspiral]

    # Generate two sets of time-invariant latent dynamics (cw and ccw)
    zs_cw = stop + 1. - orig_ts  # [ntotal]
    rs_cw = aa.reshape(-1, 1) + bb.reshape(-1, 1) * 50. / zs_cw  # [nspiral, ntotal]
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=-1)
    orig_traj_cw = np.flip(orig_traj_cw, axis=1)

    zs_cc = orig_ts
    rw_cc = aa.reshape(-1, 1) + bb.reshape(-1, 1) * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=-1)

    # Merge them depending on args.cc
    orig_trajs = []
    for i in range(nspiral):
        if args.cc == 2:
            cc = bool(npr.rand() > .5)
        else:
            cc = args.cc
        orig_traj = orig_traj_cc[i] if cc else orig_traj_cw[i]
        orig_trajs.append(orig_traj)

    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = npr.randn(*orig_trajs.shape) * noise_std + orig_trajs

    return orig_trajs, samp_trajs, orig_ts


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.elu(self.fc1(x))
        out = self.elu(self.fc2(out))
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(1, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.relu(self.fc1(z))
        out = self.fc2(out)
        return out


def log_normal_pdf(x, mean, logvar):
    const = torch.log(torch.tensor(2. * np.pi, device=x.device))
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    # KL divergence
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


class NeuralODE(nn.Module):
    def __init__(self, obs_dim, device, latent_dim=8, nhidden=16, batch_size=200):
        super(NeuralODE, self).__init__()
        self.latent_dim = latent_dim
        self.func = LatentODEfunc(latent_dim=self.latent_dim, nhidden=nhidden).to(device)
        self.rec = RecognitionRNN(latent_dim=self.latent_dim,
                                  obs_dim=obs_dim + 1,
                                  nhidden=nhidden, 
                                  nbatch=1).to(device)
        self.dec = Decoder(latent_dim=self.latent_dim,
                           obs_dim=obs_dim,
                           nhidden=nhidden).to(device)
        self.batch_size = batch_size
        self.device = device

    def forward(self, samples, orig_ts, idx=None, is_train=False):
        if is_train:
            # Subsample mini-batch
            bs = samples.shape[0]
            sample_idx = npr.choice(bs, self.batch_size, replace=False)
            samples = samples[sample_idx, ...]
            h = self.rec.initHidden().to(self.device).repeat(samples.shape[0], 1)

            # RNN backward pass
            for t in reversed(range(samples.size(1))):
                obs = samples[:, t, :-1]  # last dimension is time
                out, h = self.rec(obs, h)

            qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(self.device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # ODE forward pass
            pred_z = odeint(self.func, z0, torch.tensor(orig_ts, device=self.device))
            pred_z = pred_z.permute(1, 0, 2)  # (batch, seq, latent_dim)

            pred_x = self.dec(pred_z)
            return pred_x, qz0_mean, qz0_logvar, sample_idx
        else:
            # Full forward pass for testing
            h = self.rec.initHidden().to(self.device).repeat(samples.shape[0], 1)

            for t in reversed(range(samples.size(1))):
                obs = samples[:, t, :-1]
                out, h = self.rec(obs, h)
            qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(self.device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            pred_z = odeint(self.func, z0, torch.tensor(orig_ts, device=self.device))
            pred_z = pred_z.permute(1, 0, 2)
            pred_x = self.dec(pred_z)
            return pred_x, qz0_mean, qz0_logvar, None

    def calculate_loss(self, out, target):
        pred_x, qz0_mean, qz0_logvar, idx = out
        target_x, pz0_mean, pz0_logvar = target
        noise_std_ = torch.zeros(pred_x.size(), device=self.device) + args.noise_std
        noise_logvar = 2. * torch.log(noise_std_)

        if idx is not None:
            # Training
            logpx = log_normal_pdf(target_x[idx, ...], pred_x, noise_logvar).sum(-1).sum(-1)
        else:
            # Testing
            logpx = log_normal_pdf(target_x, pred_x, noise_logvar).sum(-1).sum(-1)

        if pz0_mean is None:
            pz0_mean = torch.zeros(qz0_mean.size(), device=self.device)
        if pz0_logvar is None:
            pz0_logvar = torch.zeros(qz0_mean.size(), device=self.device)

        analytic_kl = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum(-1)
        loss = torch.mean(-logpx + analytic_kl, dim=0)
        return loss


class ContiFormer(nn.Module):
    def __init__(self, obs_dim, device, batch_size=64, d_model=16):
        super(ContiFormer, self).__init__()
        self.d_model = d_model

        # Example of how you might scale up model capacity:
        # Increase the 'd_model', number of heads, etc.
        args_ode = {
            'use_ode': True,
            'actfn': 'tanh',
            'layer_type': 'concat',
            'zero_init': True,
            'atol': args.atol,
            'rtol': args.rtol,
            'method': args.method,
            'regularize': False,
            'approximate_method': 'bilinear',
            'nlinspace': 1,
            'linear_type': 'before',
            'interpolate': 'linear',
            'itol': 1e-2
        }
        args_ode = AttrDict(args_ode)

        # You could also add more layers or modify heads
        self.encoder = EncoderLayer(self.d_model,
                                    64,     # feed-forward dimension
                                    4,      # number of heads
                                    4,      # number of ODE blocks or sub-layers
                                    4,
                                    args=args_ode,
                                    dropout=args.dropout).to(device)

        self.lin_in = nn.Linear(obs_dim, self.d_model).to(device)
        self.lin_out = nn.Linear(self.d_model, obs_dim).to(device)

        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / self.d_model) for i in range(self.d_model)]
        )

        self.batch_size = batch_size
        self.device = device

    def temporal_enc(self, time):
        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def pad_input(self, input, t0, tmax=6 * math.pi):
        input_last = input[:, -1:, :]
        input = torch.cat((input, input_last), dim=1)
        t0 = torch.cat((t0, torch.tensor([tmax], device=t0.device)), dim=0)
        return input, t0

    def forward(self, samples, orig_ts, idx=None, is_train=False):
        bs = samples.shape[0]
        ls = samples.shape[1]  # length (time steps)
        if is_train:
            # Subsample the sequences if you want a mini-batch
            sample_idx = npr.choice(bs, self.batch_size, replace=False)
            samples = samples[sample_idx, ...]

            # Here, time is in the last dimension
            t0 = samples[..., -1]
            input = self.lin_in(samples[..., :-1])
            input = input + self.temporal_enc(t0)

            # We "pad" an extra time step
            _input, _t0 = self.pad_input(input, t0[0])

            X = torchcde.LinearInterpolation(_input, t=_t0)
            # Evaluate at all times in orig_ts
            input = X.evaluate(orig_ts).float()
            orig_ts_tensor = torch.tensor(orig_ts, device=self.device).float()

            mask = torch.zeros(self.batch_size, len(orig_ts), 1, device=self.device)
            out, _ = self.encoder(
                input, orig_ts_tensor.unsqueeze(0).repeat(self.batch_size, 1), mask=mask.bool()
            )
            return self.lin_out(out), sample_idx
        else:
            t0 = samples[..., -1]
            input = self.lin_in(samples[..., :-1])
            input = input + self.temporal_enc(t0)

            _input, _t0 = self.pad_input(input, t0[0])
            X = torchcde.LinearInterpolation(_input, t=_t0)
            input = X.evaluate(orig_ts).float()
            orig_ts_tensor = torch.tensor(orig_ts, device=self.device)

            mask = torch.zeros(bs, len(orig_ts), 1, device=self.device)
            out, _ = self.encoder(
                input, orig_ts_tensor.unsqueeze(0).repeat(bs, 1), mask=mask.bool()
            )
            return self.lin_out(out), None

    def calculate_loss(self, out, target):
        pred_x, idx = out
        target_x, _, _ = target
        if idx is not None:
            return ((pred_x - target_x[idx, ...]) ** 2).sum()
        else:
            return ((pred_x - target_x) ** 2).sum()


if __name__ == '__main__':
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    obs_dim = 2

    # Adjusted to use args.nspiral
    nspiral = args.nspiral
    start = 0.
    stop = 6 * np.pi
    noise_std = args.noise_std
    noise_a = args.noise_a
    a = 0.
    b = .3
    ntotal = 150
    nsample = 50

    # Now also adjustable
    ntrain = args.ntrain
    ntest = args.ntest

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    best_val = np.inf

    # generate toy spiral data
    orig_trajs, samp_traj, orig_ts = generate_spiral2d(
        nspiral=nspiral,
        ntotal=ntotal,
        start=start,
        stop=stop,
        noise_std=noise_std,
        noise_a=noise_a,
        a=a, b=b
    )
    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    samp_traj = torch.from_numpy(samp_traj).float().to(device)

    # normalize data
    trajs_min_x, trajs_min_y = torch.min(orig_trajs[:, :, 0]), torch.min(orig_trajs[:, :, 1])
    trajs_max_x, trajs_max_y = torch.max(orig_trajs[:, :, 0]), torch.max(orig_trajs[:, :, 1])
    orig_trajs[:, :, 0] = (orig_trajs[:, :, 0] - trajs_min_x) / (trajs_max_x - trajs_min_x)
    orig_trajs[:, :, 1] = (orig_trajs[:, :, 1] - trajs_min_y) / (trajs_max_y - trajs_min_y)
    samp_traj[:, :, 0] = (samp_traj[:, :, 0] - trajs_min_x) / (trajs_max_x - trajs_min_x)
    samp_traj[:, :, 1] = (samp_traj[:, :, 1] - trajs_min_y) / (trajs_max_y - trajs_min_y)

    # Split data
    train_trajs = samp_traj[:ntrain]
    test_trajs = samp_traj[ntrain:ntrain+ntest]
    train_target = orig_trajs[:ntrain]
    test_target = orig_trajs[ntrain:ntrain+ntest]

    # Choose test sample indices
    test_idx = npr.choice(int(ntotal * 0.5), nsample, replace=False)
    test_idx = sorted(test_idx.tolist())

    # Build model
    if args.model_name == 'Neural_ODE':
        model = NeuralODE(
            obs_dim=obs_dim,
            device=device,
            latent_dim=args.latent_dim,
            nhidden=args.nhidden,
            batch_size=args.batch_size
        )
    elif args.model_name == 'Contiformer':
        model = ContiFormer(
            obs_dim=obs_dim,
            device=device,
            batch_size=args.batch_size,
            d_model=args.nhidden  # you can rename or expand as needed
        )
    else:
        raise NotImplementedError

    # Optional: Wrap with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        log.info(f"Using {torch.cuda.device_count()} GPUs for DataParallel...")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_meter = RunningAverageMeter()

    st = 0

    # Possibly load from a checkpoint
    ckpt_path = os.path.join(args.train_dir, f'ckpt_{args.model_name}.pth')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        # If using DataParallel, you might need to load state_dict properly:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        st = checkpoint['itr']
        log.info('Loaded ckpt from {}'.format(ckpt_path))

    for itr in range(st + 1, args.niters + 1):
        model.train()
        optimizer.zero_grad()

        # Prepare mini-batch for training
        # We'll pick random time indices, similar to original
        idx = npr.choice(int(ntotal * 0.5), nsample, replace=False)
        idx = sorted(idx.tolist())
        samp_ts = torch.tensor(orig_ts[idx]).to(device)
        # shape = [ntrain, nsample, obs_dim], add time in last channel
        batch_data = train_trajs[:, idx, :]
        # Expand time dimension to match shape
        samp_ts = samp_ts.reshape(1, -1, 1).repeat(ntrain, 1, 1)
        batch_data = torch.cat((batch_data, samp_ts), dim=-1).float()

        out = model(batch_data, orig_ts, idx=idx, is_train=True)
        # For the standard VAE-style Neural ODE, we define pz0 as zeros
        try:
            pz0_mean = pz0_logvar = torch.zeros(out[1].size(), device=device)
        except:
            pz0_mean = pz0_logvar = None

        loss = model.calculate_loss(out, (train_target, pz0_mean, pz0_logvar))
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())

        log.info('Iter: {}, running loss: {:.4f}'.format(itr, loss_meter.avg))

        # Save checkpoint
        torch.save({
            # for DataParallel, save model.module.state_dict() if needed
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'itr': itr,
        }, ckpt_path)
        log.info('Stored ckpt at {}'.format(ckpt_path))

        # Validation
        model.eval()
        with torch.no_grad():
            samp_trajs_test = test_trajs[:, test_idx, :]
            samp_ts_test = torch.tensor(orig_ts[test_idx], device=device).reshape(1, -1, 1).repeat(ntest, 1, 1)
            samp_trajs_test = torch.cat((samp_trajs_test, samp_ts_test), dim=-1).float()
            # When testing, we ignore idx or pass None
            # turn orig_ts to float32
            orig_ts = np.array(orig_ts, dtype=np.float32)
            pred_x = model(samp_trajs_test, orig_ts, idx=test_idx)[0]
            mae = torch.abs(pred_x - test_target).sum(dim=-1).mean()
            rmse = torch.sqrt(((pred_x - test_target) ** 2).sum(dim=-1).mean())
            log.info('Iter: {}, MAE: {:.4f}, RMSE: {:.4f}'.format(itr, mae.item(), rmse.item()))

            if mae.item() < best_val:
                best_val = mae.item()

                ckpt_best_path = os.path.join(args.train_dir, f'ckpt_{args.model_name}_best.pth')
                torch.save({
                    'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'itr': itr,
                }, ckpt_best_path)
                log.info('Stored best ckpt at {}'.format(ckpt_best_path))

        # Optional: visualization
        if args.visualize and itr % args.log_step == 0:
            from matplotlib import pyplot as plt
            with torch.no_grad():
                ckpt_best = torch.load(ckpt_best_path, map_location=device)
                if isinstance(model, nn.DataParallel):
                    model.module.load_state_dict(ckpt_best['model_state_dict'])
                else:
                    model.load_state_dict(ckpt_best['model_state_dict'])

                # Generate predictions for a single test example
                samp_trajs_plot = test_trajs[0:1, test_idx, :]
                samp_ts_plot = torch.tensor(orig_ts[test_idx], device=device).reshape(1, -1, 1)
                samp_trajs_plot = torch.cat((samp_trajs_plot, samp_ts_plot), dim=-1).float()

                pred_plot = model(samp_trajs_plot, orig_ts, idx=test_idx)[0]
                orig_plot = test_target[0]

                # If using only the first sequence, shape [seq_len, 2]
                pred_plot = pred_plot[0].cpu().numpy()
                orig_plot = orig_plot.cpu().numpy()
                sample_plot = samp_trajs_plot[0, :, :2].cpu().numpy()

                plt.figure()
                plt.plot(orig_plot[:, 0], orig_plot[:, 1], label='True', c='green')
                plt.plot(pred_plot[:, 0], pred_plot[:, 1], label='Prediction', c='blue')
                plt.scatter(sample_plot[:, 0], sample_plot[:, 1], label='Samples', c='red')
                plt.legend()
                plt.title(f'Iteration {itr}')
                save_path = os.path.join(args.train_dir, f'vis_{itr}.png')
                plt.savefig(save_path, dpi=300)
                plt.close()
                log.info('Saved visualization figure at {}'.format(save_path))
