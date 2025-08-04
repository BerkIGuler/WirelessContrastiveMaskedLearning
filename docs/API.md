# Automated API Documentation

> **Important:** This package now uses **automated API documentation** generated directly from code docstrings!

## How to Access Current API Documentation

### 1. Build and View Locally (Recommended)

```bash
# Install documentation dependencies
pip install sphinx sphinx-autodoc-typehints sphinx-rtd-theme myst-parser

# Build documentation
make docs
# OR: python scripts/build_docs.py

# Serve locally and view in browser
make docs-serve
# OR: python scripts/build_docs.py --serve
```

Then open: http://localhost:8000

### 2. Quick Commands

```bash
make docs        # Build complete documentation
make docs-serve  # Serve documentation locally  
make docs-clean  # Clean build artifacts
```

## Why Automated Documentation?

- **Always Up-to-Date**: API docs are generated directly from code docstrings
- **No Manual Maintenance**: Documentation updates automatically when code changes
- **Comprehensive Coverage**: All classes, methods, and functions included
- **Professional Quality**: Sphinx-generated docs with cross-references and search

## Documentation Structure

The automated system generates:

- **API Reference**: Complete class and method documentation
- **User Guides**: Installation, quickstart, configuration  
- **Examples**: Training workflows and usage patterns
- **Developer Docs**: Contributing guidelines and changelog

## For Developers

When updating code, **update docstrings (not this file)**:

```python
class MyClass:
    """
    Brief description of the class.
    
    Args:
        param1: Description of parameter
        param2: Another parameter description
        
    Returns:
        Description of return value
        
    Example:
        >>> obj = MyClass(param1="value")
        >>> result = obj.method()
    """
```

Then rebuild docs: `make docs`

## Need the Old Manual API.md?

The comprehensive manual API documentation is preserved in this file's git history. However, we strongly recommend using the new automated system as it will always be current with your code.

---

**Bottom Line**: Run `make docs && make docs-serve` to access complete, up-to-date API documentation! 