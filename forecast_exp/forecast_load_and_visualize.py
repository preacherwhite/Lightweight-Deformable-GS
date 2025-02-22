import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transformers import TimeSeriesTransformerForPrediction
from forecast_test import TimeSeriesDataset
from tqdm import tqdm
def collate_fn(batch):
    past_values = torch.stack([item["past_values"] for item in batch])
    future_values = torch.stack([item["future_values"] for item in batch])
    past_time_features = torch.stack([item["past_time_features"] for item in batch])
    future_time_features = torch.stack([item["future_time_features"] for item in batch])
    past_observed_mask = torch.stack([item["past_observed_mask"] for item in batch])
    return {"past_values": past_values, "future_values": future_values, "past_time_features": past_time_features, "future_time_features": future_time_features, "past_observed_mask": past_observed_mask}
def load_and_predict():
    # Constants (make sure these match your training parameters)
    INPUT_LENGTH = 12
    TARGET_LENGTH = 6
    BATCH_SIZE = 32  # Smaller batch size for inference
    
    # Load the trained model
    print("Loading model...")
    model = TimeSeriesTransformerForPrediction.from_pretrained("./forecast_exp")
    model.eval()
    
    # Load validation data
    print("Loading data...")
    data = np.load("data/trajectories.npy")
    train_size = int(0.8 * data.shape[0])
    val_data = data[train_size:]
    
    # Create validation dataset
    val_dataset = TimeSeriesDataset(val_data, INPUT_LENGTH, TARGET_LENGTH)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Generate predictions
    print("Generating predictions...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    forecasts = []
    actuals = []
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Generate predictions
            outputs = model.generate(
                past_time_features=batch["past_time_features"],
                past_values=batch["past_values"],
                future_time_features=batch["future_time_features"],
                past_observed_mask=batch["past_observed_mask"],
            )
            
            forecasts.append(outputs.sequences.cpu().numpy())
            actuals.append(batch["future_values"].cpu().numpy())
    
    # Concatenate all predictions and actual values
    forecasts = np.concatenate(forecasts, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    # Calculate error metrics
    mse = np.mean((forecasts - actuals) ** 2)
    mae = np.mean(np.abs(forecasts - actuals))
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Visualize results for a few random samples
    num_samples = 5
    sample_indices = np.random.choice(len(forecasts), num_samples, replace=False)
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(sample_indices):
        plt.subplot(num_samples, 1, i+1)
        plt.plot(range(TARGET_LENGTH), actuals[idx, :, 0], 'b-', label='Actual')
        plt.plot(range(TARGET_LENGTH), forecasts[idx, :, 0], 'r--', label='Predicted')
        plt.title(f'Sample {i+1}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('forecast_results.png')
    print("Results visualization saved as 'forecast_results.png'")

if __name__ == "__main__":
    load_and_predict()
