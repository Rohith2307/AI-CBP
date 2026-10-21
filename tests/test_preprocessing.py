"""
Unit tests for src/preprocessing.py (Stage 4: Text Preprocessing).
"""

import pandas as pd
import pytest

from src.preprocessing import TextPreprocessor, preprocess_dataframe, NEGATION_WORDS


def test_empty_and_none_text():
    """Verify that None, empty strings, and whitespace return empty strings."""
    preprocessor = TextPreprocessor()

    assert preprocessor.process(None) == ""
    assert preprocessor.process("") == ""
    assert preprocessor.process("     ") == ""
    assert preprocessor.clean_text(None) == ""


def test_url_handling():
    """Verify that web links (http, https, www) are completely stripped."""
    preprocessor = TextPreprocessor(remove_urls=True)
    raw = "Track your baggage at https://t.co/xyz123 or www.airline.com/track for updates!"
    result = preprocessor.process(raw)

    assert "http" not in result
    assert "https" not in result
    assert "www" not in result
    assert "airline com" not in result
    assert "baggage" in result
    assert "track" in result


def test_mention_handling():
    """Verify that @mentions are replaced with a standardized USER token."""
    preprocessor = TextPreprocessor(replace_mentions="USER")
    raw = "Thanks @VirginAmerica and @custcare for the quick support!"
    result = preprocessor.process(raw)

    assert "@" not in result
    assert "virginamerica" not in result
    assert "user" in result
    assert "support" in result


def test_hashtag_handling():
    """Verify that '#' symbols are stripped while keeping meaningful hashtag keywords."""
    preprocessor = TextPreprocessor(strip_hashtags=True)
    raw = "Our luggage was lost #CustomerServiceFail #DelayedFlight"
    result = preprocessor.process(raw)

    assert "#" not in result
    assert "customerservicefail" in result
    assert "delayedflight" in result


def test_punctuation_and_whitespace():
    """Verify that noisy punctuation and repeated whitespace are cleaned."""
    preprocessor = TextPreprocessor()
    raw = "Terrible service...   Why????   Worst flight ever!!!!!"
    result = preprocessor.process(raw)

    assert "..." not in result
    assert "?" not in result
    assert "!" not in result
    assert "  " not in result
    assert "terrible service why bad flight ever" in result or "worst" in result


def test_emoji_sentiment_conversion():
    """Verify that positive and negative emojis are converted to semantic tokens."""
    preprocessor = TextPreprocessor(convert_emojis=True)

    pos_text = "Best crew ever! 😊 ❤️ 👍 :)"
    pos_result = preprocessor.process(pos_text)
    assert "emoji_pos" in pos_result

    neg_text = "Flight cancelled again! 😡 💔 👎 :-("
    neg_result = preprocessor.process(neg_text)
    assert "emoji_neg" in neg_result


def test_negation_preservation_with_stopwords():
    """
    CRITICAL: Verify that negation words ('not', 'no', 'never', 'don't', etc.)
    are NEVER stripped even when stopword removal is enabled.
    """
    proc_with_stopwords = TextPreprocessor(remove_stopwords=True, preserve_negations=True)

    text_pos = "The service was good"
    text_neg = "The service was not good"

    result_pos = proc_with_stopwords.process(text_pos)
    result_neg = proc_with_stopwords.process(text_neg)

    # Both contain 'service' and 'good'
    assert "service" in result_pos and "good" in result_pos
    assert "service" in result_neg and "good" in result_neg

    # But text_neg MUST retain 'not'!
    assert "not" not in result_pos
    assert "not" in result_neg

    # Test contractions containing negations
    contr_text = "I don't like delays and won't fly again, never!"
    contr_result = proc_with_stopwords.process(contr_text)
    assert "not" in contr_result
    assert "never" in contr_result


def test_lemmatization_toggle():
    """Verify that lemmatization reduces words to base forms and can be toggled."""
    proc_lemmatize = TextPreprocessor(lemmatize=True)
    proc_no_lemmatize = TextPreprocessor(lemmatize=False)

    text = "The planes were landing and passengers were complaining"

    lemmatized = proc_lemmatize.process(text)
    not_lemmatized = proc_no_lemmatize.process(text)

    # 'complaining' -> 'complain', 'landing' -> 'land'
    assert "complain" in lemmatized
    assert "complaining" in not_lemmatized


def test_optional_stopword_removal():
    """Verify that stopword removal is truly optional and off by default."""
    default_proc = TextPreprocessor(remove_stopwords=False)
    assert default_proc.remove_stopwords is False

    text = "This is a simple test of the system"
    result = default_proc.process(text)
    # Common stopwords like 'this', 'is', 'a', 'of', 'the' should be present
    assert "this" in result
    assert "the" in result


def test_unusual_and_accented_text():
    """Verify handling of digits, repeated chars, and accented characters."""
    preprocessor = TextPreprocessor()
    raw = "Flight UA1024 had delicious café & 100% clean seats!"
    result = preprocessor.process(raw)

    assert "ua1024" in result
    assert "100" in result
    assert "seat" in result or "seats" in result
    assert "clean" in result


def test_preprocess_dataframe_preserves_original():
    """Verify that preprocess_dataframe keeps original text intact while adding clean_text."""
    raw_df = pd.DataFrame({
        "text": [
            "Great service @VirginAmerica! ❤️",
            "Flight #123 was NOT on time... 😡",
        ],
        "sentiment": ["Positive", "Negative"],
    })

    processed_df = preprocess_dataframe(raw_df)

    assert "original_text" in processed_df.columns
    assert "clean_text" in processed_df.columns
    assert processed_df.loc[0, "original_text"] == "Great service @VirginAmerica! ❤️"
    assert processed_df.loc[1, "original_text"] == "Flight #123 was NOT on time... 😡"

    # clean_text should have tokens normalized
    assert "emoji_pos" in processed_df.loc[0, "clean_text"]
    assert "not" in processed_df.loc[1, "clean_text"]
    assert "emoji_neg" in processed_df.loc[1, "clean_text"]
