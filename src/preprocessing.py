"""
Text Preprocessing Module for Customer Sentiment Analysis Framework.

Provides modular, reproducible text cleaning, tokenization, optional lemmatization,
and sentiment-aware token handling (preserving negations and emojis).
"""

from typing import List, Optional, Set, Union
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure workspace nltk_data is active
import src  # noqa: F401


# Comprehensive set of negation words that must never be removed
NEGATION_WORDS: Set[str] = {
    "not", "no", "never", "nor", "neither", "none", "nobody", "nothing",
    "nowhere", "hardly", "barely", "scarcely", "cannot", "without",
    "dont", "don't", "didnt", "didn't", "wont", "won't", "wouldnt", "wouldn't",
    "couldnt", "couldn't", "shouldnt", "shouldn't", "cant", "can't",
    "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't", "werent", "weren't",
    "hasnt", "hasn't", "havent", "haven't", "hadnt", "hadn't", "aint", "ain't",
}

# Common English contraction expansions
CONTRACTIONS = {
    r"won\'t": "will not",
    r"can\'t": "can not",
    r"n\'t": " not",
    r"\'re": " are",
    r"\'s": " is",
    r"\'d": " would",
    r"\'ll": " will",
    r"\'t": " not",
    r"\'ve": " have",
    r"\'m": " am",
}

# Sentiment-bearing emoji mappings to distinct semantic tokens
EMOJI_SENTIMENT_MAP = {
    # Positive emojis
    "😊": " emoji_pos ", "😃": " emoji_pos ", "😄": " emoji_pos ",
    "😁": " emoji_pos ", "😍": " emoji_pos ", "🥰": " emoji_pos ",
    "❤️": " emoji_pos ", "💕": " emoji_pos ", "👍": " emoji_pos ",
    "👏": " emoji_pos ", "🎉": " emoji_pos ", "🙌": " emoji_pos ",
    "🙂": " emoji_pos ", "😀": " emoji_pos ", "✨": " emoji_pos ",
    ":)": " emoji_pos ", ":-)": " emoji_pos ", ":D": " emoji_pos ",
    # Negative emojis
    "😡": " emoji_neg ", "😠": " emoji_neg ", "💔": " emoji_neg ",
    "👎": " emoji_neg ", "😢": " emoji_neg ", "😭": " emoji_neg ",
    "😞": " emoji_neg ", "😒": " emoji_neg ", "🤦": " emoji_neg ",
    "🤮": " emoji_neg ", "😤": " emoji_neg ", "🙄": " emoji_neg ",
    ":(": " emoji_neg ", ":-(": " emoji_neg ", ">:(": " emoji_neg ",
}


class TextPreprocessor:
    """
    Configurable NLP Text Preprocessor.

    Attributes:
        lowercase: Whether to convert text to lowercase.
        remove_urls: Whether to strip web URLs.
        replace_mentions: Token to replace @user with (e.g. 'USER' or '' to remove).
        strip_hashtags: Whether to remove the '#' symbol while retaining the word.
        convert_emojis: Whether to convert known sentiment emojis to tokens.
        remove_stopwords: Whether to remove common English stopwords (default False).
        preserve_negations: Whether to protect negation words from stopword filtering.
        lemmatize: Whether to lemmatize tokens using WordNetLemmatizer.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        replace_mentions: str = "USER",
        strip_hashtags: bool = True,
        convert_emojis: bool = True,
        remove_stopwords: bool = False,
        preserve_negations: bool = True,
        lemmatize: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.replace_mentions = replace_mentions
        self.strip_hashtags = strip_hashtags
        self.convert_emojis = convert_emojis
        self.remove_stopwords = remove_stopwords
        self.preserve_negations = preserve_negations
        self.lemmatize = lemmatize

        self.lemmatizer = WordNetLemmatizer() if lemmatize else None

        # Build custom stopword set
        raw_stopwords = set(stopwords.words("english"))
        if self.preserve_negations:
            # Remove all negation words from stopwords
            self.stop_words = raw_stopwords - NEGATION_WORDS
        else:
            self.stop_words = raw_stopwords

    def clean_text(self, text: Optional[str]) -> str:
        """
        Run text through regex normalization: URLs, mentions, emojis, contractions, and punctuation.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text string.
        """
        if text is None or pd.isna(text):
            return ""

        text = str(text)

        # 1. Convert emojis to semantic tokens before lowercasing/stripping punctuation
        if self.convert_emojis:
            for emoji_char, token in EMOJI_SENTIMENT_MAP.items():
                text = text.replace(emoji_char, token)

        # 2. Lowercasing
        if self.lowercase:
            text = text.lower()

        # 3. URL handling
        if self.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # 4. Mention handling (@user -> replacement token or empty)
        if self.replace_mentions:
            mention_token = f" {self.replace_mentions.lower() if self.lowercase else self.replace_mentions} "
            text = re.sub(r"@\w+", mention_token, text)
        else:
            text = re.sub(r"@\w+", " ", text)

        # 5. Hashtag symbol handling (#greatservice -> greatservice)
        if self.strip_hashtags:
            text = re.sub(r"#(\w+)", r"\1", text)

        # 6. Expand contractions
        for pattern, replacement in CONTRACTIONS.items():
            text = re.sub(pattern, replacement, text)

        # 7. Remove remaining non-alphanumeric punctuation (preserve spaces and letters/digits)
        text = re.sub(r"[^\w\s]", " ", text)

        # 8. Collapse repeated whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize_and_lemmatize(self, cleaned_text: str) -> List[str]:
        """
        Tokenize cleaned text, filter stopwords (if enabled), and apply lemmatization.

        Args:
            cleaned_text: Cleaned string.

        Returns:
            List of processed word tokens.
        """
        if not cleaned_text:
            return []

        tokens = word_tokenize(cleaned_text)

        processed_tokens: List[str] = []
        for token in tokens:
            # Filter stopwords if requested
            if self.remove_stopwords and token in self.stop_words:
                continue

            # Optional lemmatization
            if self.lemmatize and self.lemmatizer:
                # Lemmatize both verb and noun forms if distinct
                lemmatized_v = self.lemmatizer.lemmatize(token, pos="v")
                if lemmatized_v != token:
                    token = lemmatized_v
                else:
                    token = self.lemmatizer.lemmatize(token, pos="n")

            processed_tokens.append(token)

        return processed_tokens

    def process(self, text: Optional[str]) -> str:
        """
        Full end-to-end preprocessing pipeline for a single text.

        Returns:
            A cleaned, processed string of space-separated tokens.
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        return " ".join(tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    target_col: str = "clean_text",
    original_col: str = "original_text",
    preprocessor: Optional[TextPreprocessor] = None,
) -> pd.DataFrame:
    """
    Apply text preprocessing across a DataFrame, preserving the original comment.

    Args:
        df: Input DataFrame.
        text_col: Source column containing text to preprocess.
        target_col: Destination column for preprocessed text.
        original_col: Destination/source column preserving the exact original comment.
        preprocessor: TextPreprocessor instance (creates default instance if None).

    Returns:
        DataFrame with original_col preserved and target_col populated.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")

    proc = preprocessor or TextPreprocessor()
    result_df = df.copy()

    # Ensure original text is explicitly preserved
    if original_col not in result_df.columns:
        result_df[original_col] = result_df[text_col]

    # Apply preprocessing pipeline
    result_df[target_col] = result_df[text_col].apply(proc.process)

    return result_df
