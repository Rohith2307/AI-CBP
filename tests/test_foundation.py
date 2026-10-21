"""
Foundational test suite to verify workspace structure, environment, and imports.
"""

from pathlib import Path
import sys


def test_python_version():
    """Ensure python version is 3.8 or above."""
    assert sys.version_info >= (3, 8), f"Python version too low: {sys.version}"


def test_project_structure():
    """Verify that all core project directories and files exist."""
    base_dir = Path(__file__).resolve().parent.parent

    required_dirs = [
        base_dir / "data" / "raw",
        base_dir / "data" / "processed",
        base_dir / "data" / "sample",
        base_dir / "notebooks",
        base_dir / "src",
        base_dir / "models",
        base_dir / "dashboard",
        base_dir / "tests",
    ]

    for directory in required_dirs:
        assert directory.is_dir(), f"Expected directory missing: {directory}"

    required_files = [
        base_dir / "requirements.txt",
        base_dir / "README.md",
        base_dir / ".gitignore",
        base_dir / "src" / "__init__.py",
        base_dir / "tests" / "__init__.py",
    ]

    for file_path in required_files:
        assert file_path.is_file(), f"Expected file missing: {file_path}"


def test_src_package_importable():
    """Verify that src package can be imported and exposes version."""
    import src

    assert hasattr(src, "__version__")
    assert src.__version__ == "0.1.0"


def test_core_dependencies_importable():
    """Verify that core third-party libraries import correctly."""
    import pandas as pd
    import numpy as np
    import sklearn
    import nltk
    import plotly
    import streamlit
    import joblib

    assert pd is not None
    assert np is not None
    assert sklearn is not None
    assert nltk is not None
    assert plotly is not None
    assert streamlit is not None
    assert joblib is not None

