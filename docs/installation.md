# Installation

## Quick Install

```bash
pip install -e .
```

## Development Install

```bash
git clone <repository-url>
cd WirelessContrastiveMaskedLearning
pip install -e ".[dev]"
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.21+

For complete installation instructions and requirements, see the main [README](../README.md#installation).

## Verification

Test your installation:

```python
import wimae
from wimae.models import WiMAE, ContraWiMAE

print(f"WiMAE version: {wimae.__version__}")
print("Installation successful!")
``` 