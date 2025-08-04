# Quick Start

## Basic WiMAE Usage

```python
import yaml
from wimae.models import WiMAE
from wimae.training.train_wimae import WiMAETrainer

# Load configuration
with open("configs/default_training.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create and train model
trainer = WiMAETrainer(config)
trainer.train()
```

## Basic ContraWiMAE Usage

```python
from wimae.training.train_contramae import ContraWiMAETrainer

# Use same config, but change model type
config["model"]["type"] = "contramae"

trainer = ContraWiMAETrainer(config)
trainer.train()
```

## Data Preparation

Prepare your wireless channel data:

```python
import numpy as np

# Generate complex channel data: (N_samples, 1, H, W)
channel_data = np.random.randn(1000, 1, 32, 64) + 1j * np.random.randn(1000, 1, 32, 64)

# Save with 'channels' key
np.savez('channels_001.npz', channels=channel_data)
```

## Next Steps

- See the [Configuration Guide](configuration.md) for detailed parameter descriptions
- Check out the [Examples](examples.md) for complete training workflows
- Review the [API Reference](api/models.rst) for detailed method documentation 