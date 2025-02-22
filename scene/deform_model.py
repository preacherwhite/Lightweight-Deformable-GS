import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork, DeformHead
from utils.ptv3_model import PointTransformerV3
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func

from pointcept.models.sparse_unet import SpUNetBaseWrap, SpUNetBaseV3Wrap, SpUNetBaseWrapLatent, SpUNetBaseWrapODE, SpUNetBaseWrapTorchODE
class DeformModel:
    def __init__(self, is_blender=False, is_6dof=False):
        self.deform = DeformNetwork(is_blender=is_blender, is_6dof=is_6dof).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, time_emb):
        return self.deform(xyz, time_emb)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
USE_SPUNET = True
USE_PTV3= False
USE_PONDER = False
FREEZE_ENCODER = True   
FREEZE_DECODER = True
RE_INIT_DECODER = False
PRETRAIN = True
USE_COLOR = True
D_XYZ_ONLY = False
USE_SPUNET_LATENT = False
USE_SPUNET_ODE = False
USE_SPUNET_TORCHODE = False
LARGE_HEAD = True
if USE_COLOR:
    if LARGE_HEAD:
        in_channels = 6
    else:
        in_channels = 7
else:
    if LARGE_HEAD:
        in_channels = 3
    else:
        in_channels = 4
class SetDeformModel:
    def __init__(self):
        if USE_PONDER:
            self.deform = SpUNetBaseV3Wrap(in_channels=in_channels,
                                            num_classes=0,
                                            base_channels=32,
                                            context_channels=256,
                                            channels=(32, 64, 128, 256, 256, 128, 96, 96),
                                            layers=(2, 3, 4, 6, 2, 2, 2, 2),
                                            cls_mode=False,
                                            conditions=("ScanNet", "S3DIS", "Structured3D"),
                                            zero_init=False,
                                            norm_decouple=True,
                                            norm_adaptive=False,
                                            norm_affine=True).cuda()
            if PRETRAIN:
                # Load pretrained weights for backbone
                backbone_ckpt = torch.load('/media/staging2/dhwang/Lightweight-Deformable-GS/pretrained_weights/ponderv2-ppt-pretrain-scannet-s3dis-structured3d.pth', weights_only=False)
                new_state_dict = {k.replace('module.backbone.', ''): v for k, v in backbone_ckpt['state_dict'].items()}
                # pop "module.logit_scale", "module.class_embedding", "module.embedding_table.weight", "module.proj_head.weight", "module.proj_head.bias". 
                new_state_dict.pop('module.logit_scale', None)
                new_state_dict.pop('module.class_embedding', None)
                new_state_dict.pop('module.embedding_table.weight', None)
                new_state_dict.pop('module.proj_head.weight', None)
                new_state_dict.pop('module.proj_head.bias', None)
                #popo all the keys that include modulation
                for k in list(new_state_dict.keys()):
                    if 'modulation' in k:
                        new_state_dict.pop(k)
                for k in list(new_state_dict.keys()):
                    if k.startswith('module.ponder_decoder.'):
                        new_state_dict.pop(k)
                stem_weight = new_state_dict['conv_input.conv.weight']
                if USE_COLOR:
                    # take all 6 channels and expand to 7 with random values
                    stem_weight_6ch = stem_weight
                    random_ch = torch.randn_like(stem_weight_6ch[...,:1])
                    new_state_dict['conv_input.conv.weight'] = torch.cat([stem_weight_6ch, random_ch], dim=-1)
                else:
                    # Take first 3 channels and expand to 4 with random values
                    stem_weight_3ch = stem_weight[...,:3]
                    random_ch = torch.randn_like(stem_weight_3ch[...,:1])
                    new_state_dict['conv_input.conv.weight'] = torch.cat([stem_weight_3ch, random_ch], dim=-1)
                self.deform.load_state_dict(new_state_dict)
                print("Loaded pretrained weights for deform model")
                

            if FREEZE_ENCODER:
                # Look for all parameters that include enc.
                for name, param in self.deform.named_parameters(): 
                    if 'enc' in name:
                        param.requires_grad = False
            if FREEZE_DECODER:
                    print("Freezing decoder")
                    for name, param in self.deform.named_parameters():
                        if 'dec' in name:
                            param.requires_grad = False
        if USE_SPUNET:
            if USE_SPUNET_LATENT:
                self.deform = SpUNetBaseWrapLatent(in_channels=in_channels,num_classes=0 ).cuda()
            elif USE_SPUNET_ODE:
                self.deform = SpUNetBaseWrapODE(in_channels=in_channels,num_classes=0 ).cuda()
            elif USE_SPUNET_TORCHODE:
                self.deform = SpUNetBaseWrapTorchODE(in_channels=in_channels,num_classes=0 ).cuda()
            else:
                self.deform = SpUNetBaseWrap(in_channels=in_channels,num_classes=0 ).cuda()
            if PRETRAIN:
                backbone_ckpt = torch.load('/media/staging2/dhwang/Lightweight-Deformable-GS/pretrained_weights/spunet/model/model_last.pth', weights_only=False)
                new_state_dict = {k.replace('module.backbone.', ''): v for k, v in backbone_ckpt['state_dict'].items()}
                new_state_dict.pop('module.mask_token', None)
                new_state_dict.pop('module.color_head.weight', None)
                new_state_dict.pop('module.color_head.bias', None)
                # Remove segmentation head parameters since we don't need them
                new_state_dict.pop('module.seg_head.weight', None)
                new_state_dict.pop('module.seg_head.bias', None)
                new_state_dict.pop('module.logit_scale', None)
                new_state_dict.pop('module.class_embedding', None)
                new_state_dict.pop('module.embedding_table.weight', None)
                new_state_dict.pop('module.proj_head.weight', None)
                new_state_dict.pop('module.proj_head.bias', None)
                new_state_dict.pop('final.weight', None)
                new_state_dict.pop('final.bias', None)
                stem_weight = new_state_dict['conv_input.0.weight']
                if RE_INIT_DECODER:
                    # Pop out all decoder weights
                    keys_to_pop = [k for k in new_state_dict.keys() if 'dec' in k]
                    for k in keys_to_pop:
                        new_state_dict.pop(k)
                if not USE_COLOR:
                    stem_weight = stem_weight[...,:3]
                # for color we take all 6 channels and expand to 7 with random values
                if not LARGE_HEAD:
                    random_ch = torch.randn_like(stem_weight[...,:1])
                    new_state_dict['conv_input.0.weight'] = torch.cat([stem_weight, random_ch], dim=-1)
                self.deform.load_state_dict(new_state_dict, strict=False)
                print("Loaded pretrained weights for deform model")
            if FREEZE_ENCODER:
                # Look for all parameters that include enc.
                for name, param in self.deform.named_parameters(): 
                    if 'enc' in name:
                        param.requires_grad = False
            if FREEZE_DECODER:
                    print("Freezing decoder")
                    for name, param in self.deform.named_parameters():
                        if 'dec' in name:
                            param.requires_grad = False
        if USE_PTV3:
            self.deform = PointTransformerV3(in_channels=in_channels).cuda()
            # Load pretrained weights for backbone
            # Commented out version is using scannet semantic segmentation pretrained model
            #backbone_ckpt = torch.load('/media/staging2/dhwang/Lightweight-Deformable-GS/pretrained_weights/spunet/model/model_last.pth', weights_only=False)
            # Here we use pretrained model for MSC
            if PRETRAIN:
                backbone_ckpt = torch.load('/media/staging2/dhwang/Pointcept/exp/scannet/pretrain-msc-v1m1-0-ptv3-base/model/model_last.pth', weights_only=False)
                new_state_dict = {k.replace('module.backbone.', ''): v for k, v in backbone_ckpt['state_dict'].items()}
                # Verify shape of embedding stem conv weight
                stem_weight = new_state_dict['embedding.stem.conv.weight']
                if USE_COLOR:
                    # take all 6 channels and expand to 7 with random values
                    stem_weight_6ch = stem_weight
                    random_ch = torch.randn_like(stem_weight_6ch[...,:1])
                    new_state_dict['embedding.stem.conv.weight'] = torch.cat([stem_weight_6ch, random_ch], dim=-1)
                else:
                    # Take first 3 channels and expand to 4 with random values
                    stem_weight_3ch = stem_weight[...,:3]
                    random_ch = torch.randn_like(stem_weight_3ch[...,:1])
                    new_state_dict['embedding.stem.conv.weight'] = torch.cat([stem_weight_3ch, random_ch], dim=-1)

                # Remove segmentation head parameters since we don't need them
                new_state_dict.pop('module.seg_head.weight', None)
                new_state_dict.pop('module.seg_head.bias', None)
                new_state_dict.pop('module.logit_scale', None)
                new_state_dict.pop('module.class_embedding', None)
                new_state_dict.pop('module.embedding_table.weight', None)
                new_state_dict.pop('module.proj_head.weight', None)
                new_state_dict.pop('module.proj_head.bias', None)
                #drop  "module.mask_token", "module.color_head.weight", "module.color_head.bias".   
                new_state_dict.pop('module.mask_token', None)
                new_state_dict.pop('module.color_head.weight', None)
                new_state_dict.pop('module.color_head.bias', None)

                self.deform.load_state_dict(new_state_dict)
                print("Loaded pretrained weights for deform model")
                if FREEZE_ENCODER:
                    # Look for all parameters that include enc.
                    for name, param in self.deform.named_parameters(): 
                        if 'enc' in name:
                            param.requires_grad = False
                if FREEZE_DECODER:
                    print("Freezing decoder")
                    for name, param in self.deform.named_parameters():
                        if 'dec' in name:
                            param.requires_grad = False
        self.optimizer = None
        self.spatial_lr_scale = 5
        if USE_PTV3:
            if D_XYZ_ONLY:
                self.reconstruct_head = nn.Linear(64, 3).cuda()
            else:
                self.reconstruct_head = nn.Linear(64, 10).cuda()
        elif USE_SPUNET:
            if LARGE_HEAD:
                assert D_XYZ_ONLY == False, "D_XYZ_ONLY must be False for large head"
                self.reconstruct_head = DeformHead(is_blender=True, is_6dof=False, input_ch=96).cuda()
            else:   
                if D_XYZ_ONLY:
                    self.reconstruct_head = nn.Linear(96, 3).cuda()
                else:
                    self.reconstruct_head = nn.Linear(96, 10).cuda()
        elif USE_PONDER:
            if D_XYZ_ONLY:
                self.reconstruct_head = nn.Linear(96, 3).cuda()
            else:
                self.reconstruct_head = nn.Linear(96, 10).cuda()
        
    def step(self, xyz, time_emb, additional_features=None, time_seq = None):
        # xyz: [N, 3] - gaussian positions
        # time_emb: [N, 1] - time embeddings for each gaussian
        if time_seq is None:
            assert USE_SPUNET_ODE == False, "time_seq must be provided if using ODE"
        # Normalize coordinates to a reasonable range
        xyz_min = xyz.min(dim=0)[0]
        xyz_max = xyz.max(dim=0)[0]
        xyz_normalized = (xyz - xyz_min) / (xyz_max - xyz_min + 1e-8)  # Scale to [0,1]
        
        # Concatenate position and time features
        if additional_features is not None:
            if LARGE_HEAD:
                point_features = torch.cat([xyz, additional_features], dim=-1)  # [N, 3+F]
            else:
                point_features = torch.cat([xyz, additional_features, time_emb], dim=-1)  # [N, 3+F]
        else:
            if LARGE_HEAD:
                point_features = torch.cat([xyz], dim=-1)  # [N, 4]
            else:
                point_features = torch.cat([xyz, time_emb], dim=-1)  # [N, 4]
        
        # Create the data dictionary required by PointTransformerV3
        data_dict = {
            "feat": point_features,  # [N, 4] combined features
            "coord": xyz_normalized,  # [N, 3] normalized and scaled coordinates
            "grid_size": 0.01,  # Default grid size for voxelization
            "batch": torch.zeros(xyz.shape[0], dtype=torch.long, device=xyz.device),  # All points in same batch
            "condition": "ScanNet"
        }
        
        if USE_SPUNET_ODE or USE_SPUNET_TORCHODE:
            # time_embed is used to inform encoder of current time step (included as additional feature), time sequence is used to integrate the decoder
            output = self.deform(data_dict, time_seq)
            # output is a sequence of features, we run reconstruct head for each time step
            # also construct the list of deformations for each time step
            deform_xyz = []
            deform_quat = []
            deform_scale = []
            for i in range(len(output)):
                deformations = self.reconstruct_head(output[i])
                deform_xyz.append(deformations[:, :3])
                if D_XYZ_ONLY:
                    deform_quat.append(torch.zeros_like(deformations[:, 3:7]))
                    deform_scale.append(torch.zeros_like(deformations[:, 7:]))
                else:
                    deform_quat.append(deformations[:, 3:7])
                    deform_scale.append(deformations[:, 7:])
            return deform_xyz, deform_quat, deform_scale
        
        elif LARGE_HEAD:
            output = self.deform(data_dict)
            d_xyz, d_quat, d_scale = self.reconstruct_head(output, time_emb)
            return d_xyz, d_quat, d_scale

        else:
            # Forward pass through backbone
            output = self.deform(data_dict)

            if USE_PTV3:
                output = self.reconstruct_head(output.feat)
            else:
                output = self.reconstruct_head(output)
            # Take first 3 channels as XYZ offsets
            deform_xyz = output[:, :3]
            if D_XYZ_ONLY:
                #create a zero tensor for quat and scale
                deform_quat = torch.zeros(output.shape[0], 4).cuda()
                deform_scale = torch.zeros(output.shape[0], 3).cuda()
                return deform_xyz, deform_quat, deform_scale
            else:
                deform_quat = output[:, 3:7]
                deform_scale = output[:, 7:]
                return deform_xyz, deform_quat, deform_scale

    def train_setting(self, training_args):
        # if freezing encoderr or decoder, we need to remove the parameters from the optimizer
        params = []
        for name, param in self.deform.named_parameters():
            if not (FREEZE_ENCODER and 'enc' in name) and not (FREEZE_DECODER and 'dec' in name):
                params.append({'params': param, 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "deform"})
        # l = [
        #     {'params': list(self.deform.parameters()),
        #      'lr': training_args.position_lr_init * self.spatial_lr_scale,
        #      "name": "deform"}
        # ]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            
