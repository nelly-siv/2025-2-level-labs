"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
import json

from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    NGramLanguageModelReader,
    TextProcessor,
)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    text_processor = TextProcessor('_')
    encoded = text_processor.encode(text) or tuple()
    decoded = text_processor.decode(encoded)
    print(decoded)

    main_model = NGramLanguageModel(encoded, 7)
    main_model.build()
    greedy_generator = GreedyTextGenerator(main_model, text_processor).run(51, 'Vernon')
    print(f"With greedy: {greedy_generator}")

    beam_search_generator = BeamSearchTextGenerator(main_model, text_processor, 3).run('Vernon', 56)
    print(f"With beam search: {beam_search_generator}")

    models = []
    reader = NGramLanguageModelReader("./assets/en_own.json", "_")
    for n_gram_size in [2, 3, 4]:
        model = reader.load(n_gram_size)
        if model is not None:
            models.append(model)
            test_tokens = model.generate_next_token((1,))
            print(f"Loaded n_gram_size {n_gram_size} generated {bool(test_tokens)}")
    back_off_generator = BackOffGenerator(tuple(models), text_processor).run(58, 'Vernon')
    print(f"With Back off: {back_off_generator}")
    result = back_off_generator
    assert result

if __name__ == "__main__":
    main()
