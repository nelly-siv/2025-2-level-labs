"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable


from lab_3_generate_by_ngrams.main import TextProcessor, NGramLanguageModel, GreedyTextGenerator


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    text_processor = TextProcessor('_')
    encoded_text = text_processor.encode(text)
    language_model = NGramLanguageModel(encoded_text, 7)
    build_result = language_model.build()
    text_generator = GreedyTextGenerator(language_model, text_processor)
    prompt = 'Vernon'
    seq_len = 51
    generated_text = text_generator.run(seq_len, prompt)
    print(generated_text)
    result = generated_text
    assert result

if __name__ == "__main__":
    main()
