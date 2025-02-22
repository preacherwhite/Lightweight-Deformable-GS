import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
    Trainer,
    TrainingArguments,
)
# Custom Dataset class for time series
class TimeSeriesDataset(Dataset):
    def __init__(self, trajectories, input_length, target_length):
        """
        Args:

            trajectories: Trajectories as a NumPy array (n_samples_time, n_samples_space, n_features).
            input_length: Length of input sequence.
            target_length: Length of target sequence to predict.
        """
        self.input_length = input_length
        self.target_length = target_length
        self.trajectories = trajectories
        self.num_features = trajectories.shape[2]

    def __len__(self):
        return (self.trajectories.shape[0] - self.input_length - self.target_length)*self.trajectories.shape[1]

    def __getitem__(self, idx):
        # Convert flat index to (time, space) indices
        time_idx = idx // self.trajectories.shape[1]
        space_idx = idx % self.trajectories.shape[1]
        
        # Get sequences for specific spatial point
        input_sequence = self.trajectories[time_idx:time_idx + self.input_length, space_idx, :].squeeze()
        target_sequence = self.trajectories[
            time_idx + self.input_length:time_idx + self.input_length + self.target_length,
            space_idx,
            :
        ].squeeze()
        past_time_features = np.arange(time_idx, time_idx + self.input_length)
        past_observed_mask = np.ones((self.input_length, self.num_features))
        future_time_features = np.arange(time_idx + self.input_length, time_idx + self.input_length + self.target_length)
        return {
            "past_values": torch.tensor(input_sequence, dtype=torch.float),
            "future_values": torch.tensor(target_sequence, dtype=torch.float),
            "past_time_features": torch.tensor(past_time_features, dtype=torch.float).unsqueeze(1),
            "future_time_features": torch.tensor(future_time_features, dtype=torch.float).unsqueeze(1),
            "past_observed_mask": torch.tensor(past_observed_mask, dtype=torch.float),
        }


def main():
    # Hyperparameters
    INPUT_LENGTH = 12  # Input sequence length
    TARGET_LENGTH = 6  # Target sequence length
    BATCH_SIZE = 1024
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4

    # Load dataset
    print("Loading dataset...")
    data = np.load("data/trajectories.npy")

    # Split into train and validation
    train_size = int(0.8 * data.shape[0])
    train_data, val_data = data[:train_size], data[train_size:]

    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, INPUT_LENGTH, TARGET_LENGTH)
    val_dataset = TimeSeriesDataset(val_data, INPUT_LENGTH, TARGET_LENGTH)

    # Model Configuration
    print("Initializing model...")
    config = TimeSeriesTransformerConfig(
        prediction_length=TARGET_LENGTH,
        context_length=INPUT_LENGTH-2,
        input_size=train_dataset.num_features,
        num_decoder_layers=4,
        num_encoder_layers=4,
        d_model=128,
        num_time_features = 1,
        lags_sequence = [1,2]
    )
    model = TimeSeriesTransformerForPrediction(config)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./results/lag2",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        fp16=True,
        metric_for_best_model="eval_runtime",
        greater_is_better = False,
    )

    # Data Collator
    def collate_fn(batch):
        past_values = torch.stack([item["past_values"] for item in batch])
        future_values = torch.stack([item["future_values"] for item in batch])
        past_time_features = torch.stack([item["past_time_features"] for item in batch])
        future_time_features = torch.stack([item["future_time_features"] for item in batch])
        past_observed_mask = torch.stack([item["past_observed_mask"] for item in batch])
        return {"past_values": past_values, "future_values": future_values, "past_time_features": past_time_features, "future_time_features": future_time_features, "past_observed_mask": past_observed_mask}

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save the final model
    print("Saving model...")
    model.save_pretrained("./forecast_exp")

    print("Training complete!")


if __name__ == "__main__":
    main()
