# Configuration Guide

For the complete and detailed configuration options, see the [Configuration Options](../README.md#configuration-options) section in the main README.

## Configuration Structure

The package uses YAML configuration files with the following main sections:

- **model**: Architecture and model-specific parameters
- **data**: Data loading and preprocessing settings  
- **training**: Training parameters, optimizer, and scheduler
- **logging**: Experiment tracking and checkpointing

## Quick Configuration Examples

### Minimal Configuration

```yaml
model:
  type: "wimae"
  patch_size: [1, 16]
  encoder_dim: 64

data:
  data_dir: "data/"
  normalize: true
  calculate_statistics: true

training:
  batch_size: 32
  epochs: 100
  device: "cuda:0"
  optimizer:
    type: "adam"
    lr: 0.0003
```

### Production Configuration

```yaml
# Use the provided configs/default_training.yaml as a starting point
# Copy and modify for your specific needs
```

## Configuration Loading

```python
import yaml

with open("configs/your_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Modify config programmatically if needed
config["training"]["batch_size"] = 64
config["data"]["debug_size"] = 1000  # For testing
```

## Configuration Files

The package includes several pre-configured files:

- `configs/default_training.yaml`: Complete training setup
- `configs/scenario_split_simple.yaml`: Data splitting by scenarios

See the main [README](../README.md#configuration-options) for detailed parameter descriptions and optimal settings. 