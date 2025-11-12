"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable


from lab_3_generate_by_ngrams.main import TextProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    
    process = TextProcessor('_')
    
    sentence = text[:38]
    print(f'Исходный текст: {sentence}')

    encoded_sentence = process.encode(sentence)
    print(f'Закодированный текст: {encoded_sentence}')

    decoded_sentence = process.decode(encoded_sentence)
    print(f'Раскодированный текст: {decoded_sentence}')
    result = decoded_sentence
    assert result

if __name__ == "__main__":
    main()
