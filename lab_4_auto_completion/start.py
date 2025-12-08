"""
Auto-completion start
"""

# pylint:disable=unused-variable
from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, GreedyTextGenerator
from lab_4_auto_completion.main import (
    DynamicBackOffGenerator,
    DynamicNgramLMTrie,
    load,
    NGramTrieLanguageModel,
    PrefixTrie,
    save,
    WordProcessor,
)


def clean_text(text: str):
    """
    Clean text out of extra spaces
    """
    if text is None:
        return None
    return ''.join(text.split())

def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()
    processor = WordProcessor('<EOS>')
    hp_text = processor.encode_sentences(hp_letters)

    trie = PrefixTrie()
    trie.fill(hp_text)
    suggestion = trie.suggest((2,))
    if suggestion:
        decoded = []
        for word_id in suggestion[0]:
            for word, word_id_value in processor._storage.items():
                if word_id_value == word_id:
                    decoded.append(word)
                    break
        decoded_text = processor._postprocess_decoded_text(tuple(decoded))
    print(f"Decoded result: {decoded_text}")

    model = NGramTrieLanguageModel(hp_text, 5)
    model.build()

    print(f"Greedy result before: {clean_text(GreedyTextGenerator(model, processor).run(52, 'Dear'))}")
    print(f"Beam result before: {BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)}")

    ussr_text = processor.encode_sentences(ussr_letters)
    model.update(ussr_text)

    print(f"Greedy result after: {clean_text(GreedyTextGenerator(model, processor).run(52, 'Dear'))}")
    print(f"Beam result after: {BeamSearchTextGenerator(model, processor, 3).run('Dear', 52)}")

    dynamic_trie = DynamicNgramLMTrie(hp_text, 5)
    dynamic_trie.build()

    save(dynamic_trie, "./saved_dynamic_trie.json")
    loaded_trie = load("./saved_dynamic_trie.json")

    print(f"Dynamic result before: {DynamicBackOffGenerator(loaded_trie, processor).run(50, 'Ivanov')}")

    loaded_trie.update(ussr_text)
    loaded_trie.set_current_ngram_size(3)

    print(f"Dynamic result after: {DynamicBackOffGenerator(loaded_trie, processor).run(50, 'Ivanov')}\n")
    result = DynamicBackOffGenerator(loaded_trie, processor)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
