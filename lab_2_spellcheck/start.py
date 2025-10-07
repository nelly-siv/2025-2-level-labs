"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_1_keywords_tfidf.main import (
    clean_and_tokenize,
    remove_stop_words,
)

from lab_2_spellcheck.main import (
    build_vocabulary,
    calculate_distance,
#    calculate_jaccard_distance,
#    find_correct_word,
    find_out_of_vocab_words
)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("assets/Master_and_Margarita_chapter1.txt", "r", encoding="utf-8") as file:
        text = file.read()
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    with (
        open("assets/incorrect_sentence_1.txt", "r", encoding="utf-8") as f1,
        open("assets/incorrect_sentence_2.txt", "r", encoding="utf-8") as f2,
        open("assets/incorrect_sentence_3.txt", "r", encoding="utf-8") as f3,
        open("assets/incorrect_sentence_4.txt", "r", encoding="utf-8") as f4,
        open("assets/incorrect_sentence_5.txt", "r", encoding="utf-8") as f5,
    ):
        sentences = [f.read() for f in (f1, f2, f3, f4, f5)]
    alphabet=["а","б","в","г","д","е","ё","ж","з","и","й","к","л","м","н","о","п","р","с","т","у","ф","х","ц","ч","ш","щ","ъ","ы","ь","э","ю","я"]
    tokens=clean_and_tokenize(text)
    no_stop_words=remove_stop_words(tokens,stop_words)
    voc=build_vocabulary(no_stop_words)
    print(voc)
    tokens_in_sentences=[]
    for sentence in sentences:
        sentence=sentence[:-2]
        for word in sentence.split():
            tokens_in_sentences.append(word)
    aliens=find_out_of_vocab_words(tokens_in_sentences,voc)
    print(aliens)
    for word in tokens_in_sentences:
        jaccards_voc=calculate_distance(word,voc,"jaccard",alphabet)
        print(jaccards_voc)
#    result = None
#    assert result, "Result is None"


if __name__ == "__main__":
    main()
