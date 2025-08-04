# Examples

## Interactive Training Demo

The best way to get started is with the interactive Jupyter notebook:

```bash
jupyter notebook examples/training_demo.ipynb
```

This notebook demonstrates:
- Data preparation and loading
- WiMAE training from scratch
- ContraWiMAE training with pretrained weights
- Model evaluation and visualization

## Command Line Training

### Train WiMAE

```bash
python wimae/training/train_wimae.py configs/default_training.yaml
```

### Train ContraWiMAE

```bash
python wimae/training/train_contramae.py configs/default_training.yaml
```

## Programmatic Usage

### Custom Training Loop

```python
import yaml
from wimae.training.train_wimae import WiMAETrainer

# Load and modify config
with open("configs/default_training.yaml", "r") as f:
    config = yaml.safe_load(f)

# Customize for your needs
config["training"]["batch_size"] = 32
config["training"]["epochs"] = 50
config["data"]["debug_size"] = 1000  # Use subset for testing

# Train model
trainer = WiMAETrainer(config)
results = trainer.train()
```

### Model Inference

```python
from wimae.models import WiMAE
import torch

# Load trained model
model = WiMAE.from_checkpoint("checkpoints/best_model.pt")
model.eval()

# Extract embeddings
with torch.no_grad():
    embeddings = model.encode(channel_data)
    
# Reconstruct data
with torch.no_grad():
    reconstructed = model.reconstruct(channel_data)
```

## More Examples

For complete examples and tutorials, see:
- `examples/training_demo.ipynb`: Complete training workflow
- Source code in `wimae/` for implementation details
- Test files in `tests/` for usage patterns 