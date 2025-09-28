"""
Lab 1

Extract keywords based on frequency related metrics
"""

# pylint:disable=unused-argument
import math
from typing import Any


def check_list(user_input: Any, elements_type: type, can_be_empty: bool) -> bool:
    """
    Check if the object is a list containing elements of a certain type.

    Args:
        user_input (Any): Object to check
        elements_type (type): Expected type of list elements
        can_be_empty (bool): Whether an empty list is allowed

    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(user_input, list):
        return False
    if not can_be_empty and len(user_input)==0:
        return False
    return all(isinstance(el, elements_type) for el in user_input)


def check_dict(user_input: Any, key_type: type, value_type: type, can_be_empty: bool) -> bool:
    """
    Check if the object is a dictionary with keys and values of given types.

    Args:
        user_input (Any): Object to check
        key_type (type): Expected type of dictionary keys
        value_type (type): Expected type of dictionary values
        can_be_empty (bool): Whether an empty dictionary is allowed
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(user_input, dict):
        return False
    if not can_be_empty and len(user_input)==0:
        return False
    return all (
        isinstance(key, key_type) and isinstance(value, value_type)
        for key, value in user_input.items()
    )


def check_positive_int(user_input: Any) -> bool:
    """
    Check if the object is a positive integer (not bool).

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(user_input, int):
        return False
    if isinstance(user_input, bool):
        return False
    if user_input <= 0:
        return False
    return True


def check_float(user_input: Any) -> bool:
    """
    Check if the object is a float.

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(user_input, float)


def clean_and_tokenize(text: str) -> list[str] | None:
    """
    Remove punctuation, convert to lowercase, and split into tokens.

    Args:
        text (str): Original text

    Returns:
        list[str] | None: A list of lowercase tokens without punctuation.
        In case of corrupt input arguments, None is returned.
    """
    if not isinstance (text,str):
        return None
    best_text=''
    for let in text:
        if let.isalnum() or let.isspace():
            best_text+=let.lower()
    return best_text.split()


def remove_stop_words(tokens: list[str], stop_words: list[str]) -> list[str] | None:
    """
    Exclude stop words from the token sequence.

    Args:
        tokens (list[str]): Original token sequence
        stop_words (list[str]): Tokens to exclude

    Returns:
        list[str] | None: Token sequence without stop words.
        In case of corrupt input arguments, None is returned.
    """
    if (not check_list (tokens, str, can_be_empty=False)
        or not check_list(stop_words, str, can_be_empty=True)):
        return None

    final_text=[]
    for x in tokens:
        if x not in stop_words:
            final_text.append(x)
    return final_text


def calculate_frequencies(tokens: list[str]) -> dict[str, int] | None:
    """
    Create a frequency dictionary from the token sequence.

    Args:
        tokens (list[str]): Token sequence

    Returns:
        dict[str, int] | None: A dictionary {token: occurrences}.
        In case of corrupt input arguments, None is returned.
    """
    if not check_list(tokens, str, can_be_empty=False):
        return None

    dic_freq={}
    for i in tokens:
        dic_freq[i]= tokens.count(i)
    return dic_freq


def get_top_n(frequencies: dict[str, int | float], top: int) -> list[str] | None:
    """
    Extract the most frequent tokens.

    Args:

        frequencies (dict[str, int | float]): A dictionary with tokens and their frequencies
        top (int): Number of tokens to extract

    Returns:
        list[str] | None: Top-N tokens sorted by frequency.
        In case of corrupt input arguments, None is returned.
    """
    if (not check_dict(frequencies, str, int, False) or
        check_dict(frequencies, str, float, False)) or not check_positive_int(top):
        return None
    sorted_dict = sorted(frequencies, key=lambda k: frequencies[k], reverse=True)
    if top>len(sorted_dict):
        return sorted_dict
    return sorted_dict[:top]


def calculate_tf(frequencies: dict[str, int]) -> dict[str, float] | None:
    """
    Calculate Term Frequency (TF) for each token.

    Args:
        frequencies (dict[str, int]): Raw occurrences of tokens

    Returns:
        dict[str, float] | None: Dictionary with tokens and TF values.
        In case of corrupt input arguments, None is returned.
    """
    if not check_dict(frequencies, str, int, can_be_empty=False):
        return None
    tf_dict={}
    for key, value in frequencies.items():
        tf_value=value/sum(frequencies.values())
        tf_dict[key]=tf_value
    return tf_dict


def calculate_tfidf(term_freq: dict[str, float], idf: dict[str, float]) -> dict[str, float] | None:
    """
    Calculate TF-IDF score for tokens.

    Args:
        term_freq (dict[str, float]): Term frequency values
        idf (dict[str, float]): Inverse document frequency values

    Returns:
        dict[str, float] | None: Dictionary with tokens and TF-IDF values.
        In case of corrupt input arguments, None is returned.
    """
    if not check_dict(term_freq, str, float, False) or not check_dict(idf, str, float, True):
        return None
    tfidf_dict={}
    logarifm=math.log(47)
    for key,value in term_freq.items():
        tfidf_value=value*idf.get(key,logarifm)
        tfidf_dict[key]=tfidf_value
    return tfidf_dict
