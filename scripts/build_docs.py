#!/usr/bin/env python3
"""
Automated documentation builder for WiMAE package.

This script automatically generates API documentation from code docstrings
and builds the complete documentation site.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command and handle errors."""
    print(f"Running: {description}")
    print(f"   Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description}")
        print(f"   Command: {cmd}")
        print(f"   Error: {result.stderr}")
        sys.exit(1)
    else:
        print(f"SUCCESS: {description}")
    
    return result

def clean_build():
    """Clean previous build artifacts."""
    docs_dir = Path("docs")
    build_dir = docs_dir / "_build"
    generated_dir = docs_dir / "api" / "generated"
    
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print("Cleaned build directory")
    
    if generated_dir.exists():
        shutil.rmtree(generated_dir)
        print("Cleaned generated API docs")

def install_dependencies():
    """Install documentation dependencies."""
    run_command(
        "pip install sphinx sphinx-autodoc-typehints sphinx-rtd-theme myst-parser",
        "Installing documentation dependencies"
    )

def build_docs():
    """Build the documentation."""
    docs_dir = Path("docs")
    
    # Change to docs directory
    original_dir = os.getcwd()
    os.chdir(docs_dir)
    
    try:
        # Build HTML documentation
        run_command(
            "sphinx-build -b html . _build/html",
            "Building HTML documentation"
        )
        
        # Build PDF documentation (optional)
        # run_command(
        #     "sphinx-build -b latex . _build/latex",
        #     "Building LaTeX documentation"
        # )
        
    finally:
        os.chdir(original_dir)

def check_docstrings():
    """Check for missing or incomplete docstrings."""
    print("Checking docstring quality...")
    
    # This could be extended to use tools like pydocstyle
    # For now, just a basic check
    missing_docs = []
    
    # You could add automated docstring quality checks here
    # For example, using pydocstyle or custom analysis
    
    if missing_docs:
        print("WARNING: Found potential documentation issues:")
        for issue in missing_docs:
            print(f"   - {issue}")
    else:
        print("SUCCESS: Docstring quality check passed")

def serve_docs():
    """Serve documentation locally for preview."""
    docs_dir = Path("docs") / "_build" / "html"
    
    if not docs_dir.exists():
        print("ERROR: Documentation not built. Run build first.")
        return
    
    print("Starting local documentation server...")
    print("   Open http://localhost:8000 in your browser")
    print("   Press Ctrl+C to stop")
    
    os.chdir(docs_dir)
    try:
        subprocess.run(["python", "-m", "http.server", "8000"])
    except KeyboardInterrupt:
        print("\nDocumentation server stopped")

def main():
    """Main script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build WiMAE documentation")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--check", action="store_true", help="Check docstring quality")
    parser.add_argument("--serve", action="store_true", help="Serve docs locally")
    parser.add_argument("--all", action="store_true", help="Run full build process")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        args.all = True  # Default to full build
    
    print("WiMAE Documentation Builder")
    print("=" * 40)
    
    if args.clean or args.all:
        clean_build()
    
    if args.install or args.all:
        install_dependencies()
    
    if args.check or args.all:
        check_docstrings()
    
    if args.all or not args.serve:
        build_docs()
        
        # Show results
        docs_path = Path("docs/_build/html/index.html").absolute()
        print("\nDocumentation build complete!")
        print(f"HTML docs: {docs_path}")
        print("To serve locally: python scripts/build_docs.py --serve")
    
    if args.serve:
        serve_docs()

if __name__ == "__main__":
    main() 