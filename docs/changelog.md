# Changelog

## [0.1.0] - 2025-01-XX

### Added
- Initial release of ContraWiMAE package
- WiMAE (Wireless Masked Autoencoder) model implementation
- ContraWiMAE (Contrastive WiMAE) model with multi-task learning
- Automated documentation generation from docstrings
- Comprehensive training framework with BaseTrainer
- Optimized data loading with OptimizedPreloadedDataset
- Scenario-based data splitting capabilities
- Configuration-driven training setup
- Interactive Jupyter notebook examples
- Git LFS support for sample data files

### Changed
- Model type string: `"contramae"` → `"contrawimae"` for consistency
- File structure: `base.py` → `wimae.py`, `contramae.py` → `contrawimae.py`
- Loss weighting: `contrastive_weight` parameter removed (now derived from `reconstruction_weight`)
- Validation metrics: Simplified to match training metrics (removed redundant full reconstruction loss)

### Documentation
- Automated API documentation generation using Sphinx
- Configuration guides and examples
- Complete installation and quickstart guides
- Developer contribution guidelines
- Updated all examples to use `"contrawimae"` model type

### Features
- Complex-valued channel data support
- Wireless-specific data augmentations
- Flexible patch-based processing
- Multi-task loss (reconstruction + contrastive)
- TensorBoard integration for experiment tracking
- Checkpoint management and model resumption 