# WiMAE Documentation Makefile

.PHONY: docs docs-clean docs-serve docs-install help

help:
	@echo "WiMAE Documentation Commands:"
	@echo "  docs         - Build complete documentation"
	@echo "  docs-clean   - Clean build artifacts"
	@echo "  docs-serve   - Serve documentation locally"
	@echo "  docs-install - Install documentation dependencies"

docs:
	python scripts/build_docs.py --all

docs-clean:
	python scripts/build_docs.py --clean

docs-serve:
	python scripts/build_docs.py --serve

docs-install:
	python scripts/build_docs.py --install

# Legacy support
clean-docs: docs-clean
build-docs: docs
serve-docs: docs-serve 