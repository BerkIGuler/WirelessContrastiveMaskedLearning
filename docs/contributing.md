# Contributing

## Documentation Updates

The API documentation is now **automatically generated** from code docstrings. When you update code:

1. **Update docstrings** in your code (not the API.md file)
2. **Rebuild documentation**: `make docs` or `python scripts/build_docs.py`
3. **Review changes**: `make docs-serve` to preview locally

## Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper docstrings
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Update documentation: `make docs`
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Documentation Guidelines

- Write clear docstrings in Google/NumPy style
- Include parameter types and descriptions
- Add usage examples where helpful
- The API docs will be auto-generated from these docstrings

## Development Setup

```bash
pip install -e ".[dev]"
make docs-install  # Install documentation dependencies
``` 