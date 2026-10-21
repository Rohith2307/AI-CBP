"""
Customer Sentiment Analysis Framework
A modular machine-learning framework for analyzing sentiment in customer and social media text.
"""

from pathlib import Path
import nltk

__version__ = "0.1.0"

# Automatically register workspace nltk_data directory if present
_workspace_nltk_data = Path(__file__).resolve().parent.parent / "data" / "nltk_data"
if _workspace_nltk_data.exists():
    _nltk_path_str = str(_workspace_nltk_data)
    if _nltk_path_str not in nltk.data.path:
        nltk.data.path.insert(0, _nltk_path_str)
