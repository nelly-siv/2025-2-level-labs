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
    encoded_text = text_processor.encode(text) or tuple()
    decoded_text = text_processor.decode(encoded_text)
    print(decoded_text)

    language_model = NGramLanguageModel(encoded_text, 7)
    language_model.build()
    greedy_generator = GreedyTextGenerator(language_model, text_processor).run(51, 'Vernon')
    print(f"With greedy: {greedy_generator}")

    beam_search_generator = BeamSearchTextGenerator(language_model, text_processor, 3)
    beam_search_result = beam_search_generator.run("Vernon", 56)
    print(f"With beam search: {beam_search_result}")

    models = []
    for n_gram_size in [2, 3, 4]:
        model = NGramLanguageModelReader("./assets/en_own.json", "_").load(n_gram_size)
        if model is not None:
            models.append(model)
    back_off_algorithm = BackOffGenerator(tuple(models), text_processor).run(60, 'Vernon')
    print(f"With Back off: {back_off_algorithm}")
    result = back_off_algorithm
    assert result

if __name__ == "__main__":
    main()
