"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
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
    processor = TextProcessor(".")
    encoded_text = processor.encode(text)
    if encoded_text is None:
        return
    model = NGramLanguageModel(encoded_text, 7)
    model.build()
    generator = GreedyTextGenerator(model, processor)
    result_generator = generator.run(51, "Vernon")
    print(result_generator)

    beam_search = BeamSearchTextGenerator(model, processor, 3)
    beam_search_ = beam_search.run("Vernon", 56)
    result_beam = beam_search_
    print(result_beam)

    language_models = []
    for n_gram_size in [1, 2, 3]:
        loaded_model = NGramLanguageModelReader("./assets/en_own.json", "_").load(n_gram_size)
        if loaded_model is not None:
            language_models.append(model)

    back_off = BackOffGenerator(tuple(language_models), processor).run(60, 'Vernon')
    result = back_off
    print(result)
    assert result

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
