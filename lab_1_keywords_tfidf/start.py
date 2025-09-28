"""
Frequency-driven keyword extraction starter
"""

# pylint:disable=too-many-locals, unused-argument, unused-variable, invalid-name, duplicate-code
from json import load

from lab_1_keywords_tfidf.main import (
    calculate_frequencies,
    calculate_tf,
    calculate_tfidf,
    clean_and_tokenize,
    get_top_n,
    remove_stop_words,
)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("assets/Дюймовочка.txt", "r", encoding="utf-8") as file:
        target_text = file.read()
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    #with open("assets/IDF.json", "r", encoding="utf-8") as file:
     #   idf = load(file)
    words=clean_and_tokenize(target_text)
    if words is None:
        return
    final_words=remove_stop_words(words,stop_words)
    if final_words is None:
        return
    d_calc=calculate_frequencies(final_words)
    if d_calc is None:
        return
    print(get_top_n(d_calc, 10))
    #tf_dict=calculate_tf(d_calc)
    #if tf_dict is None:
    #    return
    #tfidf_dict=calculate_tfidf(tf_dict, idf)
    #if tfidf_dict is None:
    #    return
   # print(get_top_n(tfidf_dict, 10))
    #with open("assets/corpus_frequencies.json", "r", encoding="utf-8") as file:
    #    corpus_freqs = load(file)
    #result = None
    #assert result, "Keywords are not extracted"

if __name__ == "__main__":
    main()
