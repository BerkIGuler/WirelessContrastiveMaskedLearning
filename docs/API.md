# API Documentation

## Models

### WiMAE

The base Wireless Masked Autoencoder model.

```python
from wimae.models import WiMAE

model = WiMAE(
    patch_size=(4, 4),
    encoder_dim=256,
    encoder_layers=12,
    encoder_nhead=8,
    decoder_layers=8,
    decoder_nhead=8,
    mask_ratio=0.75,
    device="cuda"
)
```

**Parameters:**
- `patch_size`: Tuple of (height, width) for each patch
- `encoder_dim`: Dimension of encoder
- `encoder_layers`: Number of encoder layers
- `encoder_nhead`: Number of encoder attention heads
- `decoder_layers`: Number of decoder layers
- `decoder_nhead`: Number of decoder attention heads
- `mask_ratio`: Ratio of patches to mask during training
- `device`: Device to place the model on

**Methods:**
- `forward(x, mask_ratio=None, return_reconstruction=True)`: Forward pass
- `encode(x)`: Encode input data without masking
- `reconstruct(x)`: Reconstruct input data
- `get_embeddings(x, pooling="mean")`: Get embeddings from encoded features
- `save_checkpoint(filepath, **kwargs)`: Save model checkpoint
- `from_checkpoint(filepath, device=None)`: Load model from checkpoint

### ContraWiMAE

Contrastive Wireless Masked Autoencoder that extends WiMAE.

```python
from wimae.models import ContraWiMAE

model = ContraWiMAE(
    patch_size=(4, 4),
    encoder_dim=256,
    encoder_layers=12,
    encoder_nhead=8,
    decoder_layers=8,
    decoder_nhead=8,
    mask_ratio=0.75,
    contrastive_dim=256,
    temperature=0.1,
    snr_min=0.0,
    snr_max=30.0,
    device="cuda"
)
```

**Additional Parameters:**
- `contrastive_dim`: Dimension of contrastive projection
- `temperature`: Temperature parameter for contrastive loss
- `snr_min`: Minimum SNR for augmentations
- `snr_max`: Maximum SNR for augmentations

**Additional Methods:**
- `forward_with_augmentation(x, mask_ratio=None, return_reconstruction=True)`: Forward pass with augmentation
- `compute_contrastive_loss(features1, features2, temperature=None)`: Compute contrastive loss
- `get_contrastive_embeddings(x, pooling="mean")`: Get contrastive embeddings

## Training

### WiMAETrainer

Trainer for WiMAE models.

```python
from wimae.training import WiMAETrainer

trainer = WiMAETrainer(config)
trainer.train()
```

### ContraWiMAETrainer

Trainer for ContraWiMAE models.

```python
from wimae.training import ContraWiMAETrainer

trainer = ContraWiMAETrainer(config)
trainer.train()
```

## Encoding

### Encoder

Interface for generating embeddings from trained models.

```python
from wimae.encoding import Encoder

encoder = Encoder(config)
embeddings = encoder.encode_data("path/to/data.npz")
encoder.save_embeddings(embeddings, "embeddings.pt")
```

**Methods:**
- `encode_data(data_path, data_format="npz")`: Encode data and return embeddings
- `encode_and_save(data_path, data_format="npz", filename=None)`: Encode data and save embeddings
- `save_embeddings(embeddings, filename=None)`: Save embeddings to file
- `from_config(config_path)`: Create encoder from configuration file
- `from_checkpoint(checkpoint_path, device="cuda")`: Create encoder from model checkpoint

## Downstream Tasks

### BeamPredictionTask

Beam prediction downstream task.

```python
from wimae.downstream import BeamPredictionTask

task = BeamPredictionTask(config)
train_loader, val_loader, test_loader = task.load_data()
history = task.train(train_loader, val_loader)
metrics = task.evaluate(test_loader)
```

### LOSClassificationTask

Line-of-Sight classification downstream task.

```python
from wimae.downstream import LOSClassificationTask

task = LOSClassificationTask(config)
train_loader, val_loader, test_loader = task.load_data()
history = task.train(train_loader, val_loader)
metrics = task.evaluate(test_loader)
```

### DownstreamTrainer

Unified trainer for downstream tasks.

```python
from wimae.downstream import DownstreamTrainer

trainer = DownstreamTrainer(config)
metrics = trainer.run()
```

## Configuration

The package uses YAML configuration files for all components. Example configurations are provided in the `configs/` directory:

- `default_training.yaml`: Training configuration
- `default_encoding.yaml`: Encoding configuration
- `default_downstream.yaml`: Downstream task configuration

## Command Line Interface

The package provides command-line scripts for easy usage:

```bash
# Training
python examples/training_example.py configs/default_training.yaml --model wimae

# Encoding
python examples/encoding_example.py --checkpoint model.pt --data data.npz

# Downstream tasks
python examples/downstream_example.py configs/default_downstream.yaml --task beam_prediction
```

## Data Format

The package expects wireless channel data in `.npz` format with the following structure:
- Channel data as numpy arrays with shape (batch_size, channels, height, width)
- Labels for downstream tasks as separate files

## Metrics

### Beam Prediction
- Top-1 accuracy
- Top-3 accuracy
- Cross-entropy loss

### LOS Classification
- Accuracy
- Precision
- Recall
- F1-score
- AUC
- Specificity
- NPV
- Confusion matrix elements 