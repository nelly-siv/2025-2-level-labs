"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
import json

from lab_3_generate_by_ngrams.main import (
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    TextProcessor)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    text_processor = TextProcessor('_')
    encoded_text = text_processor.encode(text) or tuple()
    language_model = NGramLanguageModel(encoded_text, 7)
    language_model.build()
    greedy_generator = GreedyTextGenerator(language_model, text_processor).run(51, 'Vernon')
    print(greedy_generator)
    beam_search_generator = BeamSearchTextGenerator(language_model, text_processor, 3).run('Vernon', 56)
    print(beam_search_generator)
    result = beam_search_generator
    assert result

if __name__ == "__main__":
    main()
