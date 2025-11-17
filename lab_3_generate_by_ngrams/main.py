"""
Lab 3.

Beam-search and natural language generation evaluation
"""

# pylint:disable=too-few-public-methods, unused-import
import json
import math

from lab_1_keywords_tfidf.main import check_dict, check_list


class TextProcessor:
    """
    Handle text tokenization, encoding and decoding.

    Attributes:
        _end_of_word_token (str): A token denoting word boundary
        _storage (dict): Dictionary in the form of <token: identifier>
    """

    def __init__(self, end_of_word_token: str) -> None:
        """
        Initialize an instance of LetterStorage.

        Args:
            end_of_word_token (str): A token denoting word boundary
        """
        self._end_of_word_token = end_of_word_token
        self._storage = {self._end_of_word_token: 0}

    def _tokenize(self, text: str) -> tuple[str, ...] | None:
        """
        Tokenize text into unigrams, separating words with special token.

        Punctuation and digits are removed. EoW token is appended after the last word in two cases:
        1. It is followed by punctuation
        2. It is followed by space symbol

        Args:
            text (str): Original text

        Returns:
            tuple[str, ...] | None: Tokenized text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not isinstance(text, str):
            return None

        text = text.lower()

        tokens = []
        words = text.split()

        for word in words:
            word_tokens = [letter for letter in word if letter.isalpha()]

            if word_tokens:
                tokens.extend(word_tokens)
                tokens.append(self._end_of_word_token)

        if not tokens:
            return None

        if (tokens and
            tokens[-1] == self._end_of_word_token and
            text[-1].isalnum()):
            tokens = tokens[:-1]

        return tuple(tokens)

    def get_id(self, element: str) -> int | None:
        """
        Retrieve a unique identifier of an element.

        Args:
            element (str): String element to retrieve identifier for

        Returns:
            int | None: Integer identifier that corresponds to the given element

        In case of corrupt input arguments or arguments not included in storage,
        None is returned
        """
        if (
            not element or
            not isinstance(element, str) or
            element not in self._storage
            ):
            return None

        return self._storage.get(element)

    def get_end_of_word_token(self) -> str:  # type: ignore[empty-body]
        """
        Retrieve value stored in self._end_of_word_token attribute.

        Returns:
            str: EoW token
        """
        return self._end_of_word_token

    def get_token(self, element_id: int) -> str | None:
        """
        Retrieve an element by unique identifier.

        Args:
            element_id (int): Identifier to retrieve identifier for

        Returns:
            str | None: Element that corresponds to the given identifier

        In case of corrupt input arguments or arguments not included in storage, None is returned
        """
        if (not isinstance(element_id, int) or
            element_id not in set(self._storage.values())):
            return None

        for element, ind in self._storage.items():
            if ind == element_id:
                return element

        return None

    def encode(self, text: str) -> tuple[int, ...] | None:
        """
        Encode text.

        Tokenize text, assign each symbol an integer identifier and
        replace letters with their ids.

        Args:
            text (str): An original text to be encoded

        Returns:
            tuple[int, ...] | None: Processed text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """

        if not isinstance(text, str):
            return None

        tokens = self._tokenize(text)
        if tokens is None:
            return None

        encoded_tokens = []

        for element in tokens:
            if element not in self._storage:
                self._put(element)

            token_id = self.get_id(element)
            if token_id is None:
                return None

            encoded_tokens.append(token_id)

        return tuple(encoded_tokens)

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): An element to put into storage

        In case of corrupt input arguments or invalid argument length,
        an element is not added to storage
        """
        if (
            not isinstance(element, str) or
            len(element) != 1
            ):
            return

        if element not in self._storage:
            self._storage[element] = len(self._storage)


    def decode(self, encoded_corpus: tuple[int, ...]) -> str | None:
        """
        Decode and postprocess encoded corpus by converting integer identifiers to string.

        Special symbols are replaced with spaces (no multiple spaces in a row are allowed).
        The first letter is capitalized, resulting sequence must end with a full stop.

        Args:
            encoded_corpus (tuple[int, ...]): A tuple of encoded tokens

        Returns:
            str | None: Resulting text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not encoded_corpus or not isinstance(encoded_corpus, tuple):
            return None

        decoded_corpus = self._decode(encoded_corpus)
        if decoded_corpus is None:
            return None

        result = self._postprocess_decoded_text(decoded_corpus)
        if result is None:
            return None

        return str(result)

    def fill_from_ngrams(self, content: dict) -> None:
        """
        Fill internal storage with letters from external JSON.

        Args:
            content (dict): ngrams from external JSON
        """
        if not isinstance(content, dict) or not content:
            return

        for n_gram in content['freq']:
            for char in n_gram.lower():
                if char.isalpha():
                    self._put(char)

    def _decode(self, corpus: tuple[int, ...]) -> tuple[str, ...] | None:
        """
        Decode sentence by replacing ids with corresponding letters.

        Args:
            corpus (tuple[int, ...]): A tuple of encoded tokens

        Returns:
            tuple[str, ...] | None: Sequence with decoded tokens

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not corpus or not isinstance(corpus, tuple):
            return None

        decoded_tokens = []

        for ind in corpus:
            token = self.get_token(ind)
            if token is None:
                return None

            decoded_tokens.append(token)


        return tuple(decoded_tokens)

    def _postprocess_decoded_text(self, decoded_corpus: tuple[str, ...]) -> str | None:
        """
        Convert decoded sentence into the string sequence.

        Special symbols are replaced with spaces (no multiple spaces in a row are allowed).
        The first letter is capitalized, resulting sequence must end with a full stop.

        Args:
            decoded_corpus (tuple[str, ...]): A tuple of decoded tokens

        Returns:
            str | None: Resulting text

        In case of corrupt input arguments, None is returned
        """
        if not decoded_corpus or not isinstance(decoded_corpus, tuple):
            return None

        phrase_letters = []
        ex_space = False

        for token in decoded_corpus:
            if token == self._end_of_word_token:
                if not ex_space:
                    phrase_letters.append(' ')
                    ex_space = True
            else:
                phrase_letters.append(token)
                ex_space = False

        phrase = ''.join(phrase_letters).strip()
        if not phrase:
            return None

        final_phrase = phrase.capitalize()
        if final_phrase[-1] != '.':
            final_phrase += '.'

        return final_phrase


class NGramLanguageModel:
    """
    Store language model by n_grams, predict the next token.

    Attributes:
        _n_gram_size (int): A size of n-grams to use for language modelling
        _n_gram_frequencies (dict): Frequencies for n-grams
        _encoded_corpus (tuple): Encoded text
    """

    def __init__(self, encoded_corpus: tuple | None, n_gram_size: int) -> None:
        """
        Initialize an instance of NGramLanguageModel.

        Args:
            encoded_corpus (tuple | None): Encoded text
            n_gram_size (int): A size of n-grams to use for language modelling
        """
        self._encoded_corpus = encoded_corpus
        self._n_gram_size = n_gram_size
        self._n_gram_frequencies = {}

    def get_n_gram_size(self) -> int:  # type: ignore[empty-body]
        """
        Retrieve value stored in self._n_gram_size attribute.

        Returns:
            int: Size of stored n_grams
        """
        return self._n_gram_size

    def set_n_grams(self, frequencies: dict) -> None:
        """
        Setter method for n-gram frequencies.

        Args:
            frequencies (dict): Computed in advance frequencies for n-grams
        """
        if not frequencies or not check_dict(frequencies, tuple, float, True):
            return None

        self._n_gram_frequencies = frequencies

        return None

    def build(self) -> int:  # type: ignore[empty-body]
        """
        Fill attribute `_n_gram_frequencies` from encoded corpus.

        Encoded corpus is stored in the attribute `_encoded_corpus`

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1

        In case of corrupt input arguments or methods used return None,
        1 is returned
        """
        if not isinstance(self._encoded_corpus, tuple) or not self._encoded_corpus:
            return 1

        corpus = self._extract_n_grams(self._encoded_corpus)
        if corpus is None:
            return 1

        abs_freq = {}

        prefix_freq = {}

        for n_gram in corpus:
            abs_freq[n_gram] = abs_freq.get(n_gram, 0) + 1
            prefix = n_gram[:-1]
            prefix_freq[prefix] = prefix_freq.get(prefix, 0) + 1

        for key, value in abs_freq.items():
            prefix = key[:-1]
            prefix_frequency = prefix_freq.get(prefix, 0)
            if prefix_frequency > 0:
                self._n_gram_frequencies[key] = value / prefix_frequency

        return 0


    def generate_next_token(self, sequence: tuple[int, ...]) -> dict | None:
        """
        Retrieve tokens that can continue the given sequence along with their probabilities.

        Args:
            sequence (tuple[int, ...]): A sequence to match beginning of NGrams for continuation

        Returns:
            dict | None: Possible next tokens with their probabilities

        In case of corrupt input arguments, None is returned
        """
        if (
            not sequence or
            not isinstance(sequence, tuple) or
            len(sequence) < self._n_gram_size - 1
            ):
            return None

        context_len = self._n_gram_size - 1
        context = sequence[-context_len:]

        tokens = {}

        for n_gram, freq in self._n_gram_frequencies.items():
            if len(n_gram) != self._n_gram_size:
                continue
            n_gram_context = n_gram[:-1]
            if n_gram_context == context:
                token = n_gram[-1]
                tokens[token] = freq

        if not tokens:
            return None

        sorted_tokens = dict(sorted(tokens.items(),
            key=lambda x: (-x[1], -x[0])))
        return sorted_tokens

    def _extract_n_grams(
        self, encoded_corpus: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...] | None:
        """
        Split encoded sequence into n-grams.

        Args:
            encoded_corpus (tuple[int, ...]): A tuple of encoded tokens

        Returns:
            tuple[tuple[int, ...], ...] | None: A tuple of extracted n-grams

        In case of corrupt input arguments, None is returned
        """
        if not encoded_corpus or not isinstance(encoded_corpus, tuple):
            return None

        if len(encoded_corpus) < self._n_gram_size:
            return tuple()

        main_list = []

        for i in range (len(encoded_corpus) - self._n_gram_size + 1):
            n_gram = tuple(encoded_corpus[i:i + self._n_gram_size])
            main_list.append(n_gram)

        return tuple(main_list)

class GreedyTextGenerator:
    """
    Greedy text generation by N-grams.

    Attributes:
        _model (NGramLanguageModel): A language model to use for text generation
        _text_processor (TextProcessor): A TextProcessor instance to handle text processing
    """

    def __init__(self, language_model: NGramLanguageModel, text_processor: TextProcessor) -> None:
        """
        Initialize an instance of GreedyTextGenerator.

        Args:
            language_model (NGramLanguageModel): A language model to use for text generation
            text_processor (TextProcessor): A TextProcessor instance to handle text processing
        """
        self._model = language_model
        self._text_processor = text_processor

    def run(self, seq_len: int, prompt: str) -> str | None:
        """
        Generate sequence based on NGram language model and prompt provided.

        Args:
            seq_len (int): Number of tokens to generate
            prompt (str): Beginning of sequence

        Returns:
            str | None: Generated sequence

        In case of corrupt input arguments or methods used return None,
        None is returned
        """
        if (
            not isinstance(seq_len, int) or
            not prompt or
            not isinstance(prompt, str) or
            seq_len <= 0
            ):
            return None

        encoded_text = self._text_processor.encode(prompt)
        if encoded_text is None:
            return None

        for _ in range(seq_len):
            candidates = self._model.generate_next_token(encoded_text)
            if not candidates:
                break
            next_token = next(iter(candidates.keys()))
            encoded_text = encoded_text + (next_token,)

        return self._text_processor.decode(encoded_text)

class BeamSearcher:
    """
    Beam Search algorithm for diverse text generation.

    Attributes:
        _beam_width (int): Number of candidates to consider at each step
        _model (NGramLanguageModel): A language model to use for next token prediction
    """

    def __init__(self, beam_width: int, language_model: NGramLanguageModel) -> None:
        """
        Initialize an instance of BeamSearchAlgorithm.

        Args:
            beam_width (int): Number of candidates to consider at each step
            language_model (NGramLanguageModel): A language model to use for next token prediction
        """
        self._beam_width = beam_width
        self._model = language_model

    def get_next_token(self, sequence: tuple[int, ...]) -> list[tuple[int, float]] | None:
        """
        Retrieve candidate tokens for sequence continuation.

        The valid candidate tokens are those that are included in the N-gram with.
        Number of tokens retrieved must not be bigger that beam width parameter.

        The return value has the following format: [(token, probability), ...].
        The return value length matches the Beam Size parameter.

        Args:
            sequence (tuple[int, ...]): Base sequence to continue

        Returns:
            list[tuple[int, float]] | None: Tokens to use for base sequence continuation

        In case of corrupt input arguments or methods used return None.
        """
        if not isinstance(sequence, tuple) or not sequence:
            return None

        next_tokens = self._model.generate_next_token(sequence)
        if next_tokens is None:
            return None
        if next_tokens == {}:
            return []
        if not next_tokens:
            return []

        sorted_tokens = sorted(next_tokens.items(), key=lambda x: ((-x[1], x[0])))

        return sorted_tokens[:self._beam_width]

    def continue_sequence(
        self,
        sequence: tuple[int, ...],
        next_tokens: list[tuple[int, float]],
        sequence_candidates: dict[tuple[int, ...], float],
    ) -> dict[tuple[int, ...], float] | None:
        """
        Generate new sequences from the base sequence with next tokens provided.

        The base sequence is deleted after continued variations are added.

        Args:
            sequence (tuple[int, ...]): Base sequence to continue
            next_tokens (list[tuple[int, float]]): Token for sequence continuation
            sequence_candidates (dict[tuple[int, ...], float]):
                Storage with all sequences generated

        Returns:
            dict[tuple[int, ...], float] | None: Updated sequence candidates

        In case of corrupt input arguments or unexpected behaviour of methods used return None.
        """
        if (
            not isinstance(sequence, tuple) or
            not check_list(next_tokens, tuple, False) or
            not check_dict(sequence_candidates, tuple, float, True) or
            len(next_tokens) > self._beam_width or
            sequence not in sequence_candidates
            ):
            return None

        new_candidates = sequence_candidates.copy()
        current_score = new_candidates[sequence]

        for token,prob in next_tokens:
            if prob <= 0:
                continue
            new_sequence = sequence + (token,)
            new_score = current_score - math.log(prob)
            new_candidates[new_sequence] = new_score

        del new_candidates[sequence]

        return new_candidates

    def prune_sequence_candidates(
        self, sequence_candidates: dict[tuple[int, ...], float]
    ) -> dict[tuple[int, ...], float] | None:
        """
        Remove those sequence candidates that do not make top-N most probable sequences.

        Args:
            sequence_candidates (dict[tuple[int, ...], float]): Current candidate sequences

        Returns:
            dict[tuple[int, ...], float] | None: Pruned sequences

        In case of corrupt input arguments return None.
        """
        if (
            not check_dict(sequence_candidates, tuple, float, True) or
            not sequence_candidates
            ):
            return None

        sorted_candidates = sorted(sequence_candidates.items(),
                                   key=lambda item: item[1], reverse=False)

        return dict(sorted_candidates[:self._beam_width])


class BeamSearchTextGenerator:
    """
    Class for text generation with BeamSearch.

    Attributes:
        _language_model (tuple[NGramLanguageModel]): Language models for next token prediction
        _text_processor (NGramLanguageModel): A TextProcessor instance to handle text processing
        _beam_width (NGramLanguageModel): Beam width parameter for generation
        beam_searcher (NGramLanguageModel): Searcher instances for each language model
    """

    def __init__(
        self, language_model: NGramLanguageModel, text_processor: TextProcessor, beam_width: int
    ) -> None:
        """
        Initializes an instance of BeamSearchTextGenerator.

        Args:
            language_model (NGramLanguageModel): Language model to use for text generation
            text_processor (TextProcessor): A TextProcessor instance to handle text processing
            beam_width (int): Beam width parameter for generation
        """
        self._language_model = language_model
        self._text_processor = text_processor
        self._beam_width = beam_width
        self.beam_searcher = BeamSearcher(self._beam_width, language_model)

    def run(self, prompt: str, seq_len: int) -> str | None:
        """
        Generate sequence based on NGram language model and prompt provided.

        Args:
            prompt (str): Beginning of sequence
            seq_len (int): Number of tokens to generate

        Returns:
            str | None: Generated sequence

        In case of corrupt input arguments or methods used return None,
        None is returned
        """
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(seq_len, int)
            or seq_len <= 0
            ):
            return None

        encoded_prompt = self._text_processor.encode(prompt)
        if not encoded_prompt:
            return None

        sequence_candidates = {encoded_prompt: 0.0}

        for _ in range(seq_len):
            new_candidates = {}
            has_valid_candidates = False

            for sequence, score in sequence_candidates.items():
                next_tokens = self._get_next_token(sequence)
                if next_tokens is None:
                    return None
                if not next_tokens:
                    new_candidates[sequence] = score
                    continue

                temp_candidates = {sequence: score}
                continued_candidates = self.beam_searcher.continue_sequence(
                sequence, next_tokens, temp_candidates
                )
                if continued_candidates is None:
                    new_candidates[sequence] = score
                else:
                    new_candidates.update(continued_candidates)
                    has_valid_candidates = True

            if not new_candidates:
                break

            pruned_candidates = self.beam_searcher.prune_sequence_candidates(new_candidates)
            if pruned_candidates is None:
                return None

            sequence_candidates = pruned_candidates

        if not sequence_candidates:
            return None

        best_sequence = min(sequence_candidates.items(), key=lambda x: x[1])[0]
        return self._text_processor.decode(best_sequence)

    def _get_next_token(
        self, sequence_to_continue: tuple[int, ...]
    ) -> list[tuple[int, float]] | None:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            list[tuple[int, float]] | None: Next tokens for sequence
            continuation

        In case of corrupt input arguments return None.
        """
        if not isinstance(sequence_to_continue, tuple) or not sequence_to_continue:
            return None

        return self.beam_searcher.get_next_token(sequence_to_continue)


class NGramLanguageModelReader:
    """
    Factory for loading language models ngrams from external JSON.

    Attributes:
        _json_path (str): Local path to assets file
        _eow_token (str): Special token for text processor
        _text_processor (TextProcessor): A TextProcessor instance to handle text processing
    """

    def __init__(self, json_path: str, eow_token: str) -> None:
        """
        Initialize reader instance.

        Args:
            json_path (str): Local path to assets file
            eow_token (str): Special token for text processor
        """
        self._json_path = json_path
        self._eow_token = eow_token
        with open(json_path, 'r', encoding='utf-8') as file:
            self._content = json.load(file)
        self._text_processor = TextProcessor(self._eow_token)
        self._text_processor.fill_from_ngrams(self._content)

    def load(self, n_gram_size: int) -> NGramLanguageModel | None:
        """
        Fill attribute `_n_gram_frequencies` from dictionary with N-grams.

        The N-grams taken from dictionary must be cleaned from digits and punctuation,
        their length must match n_gram_size, and spaces must be replaced with EoW token.

        Args:
            n_gram_size (int): Size of ngram

        Returns:
            NGramLanguageModel | None: Built language model.

        In case of corrupt input arguments or unexpected behaviour of methods used, return 1.
        """
        if (not isinstance(n_gram_size, int) or n_gram_size < 2):
            return None

        n_grams = {}
        eow_id = self._text_processor.get_id(self._eow_token)
        if eow_id is None:
            return None

        for ngram, freq in self._content['freq'].items():
            processed_elements = []

            for element in ngram:
                if element.isalpha():
                    element_id = self._text_processor.get_id(element.lower())
                    if element_id is None:
                        continue

                    processed_elements.append(element_id)

                elif element.isspace():
                    processed_elements.append(eow_id)

            if len(processed_elements) == n_gram_size:
                processed_t = tuple(processed_elements)
                n_grams[processed_t] = n_grams.get(processed_t, 0.0) + freq

        context_frequencies = {}
        for ngram, freq in n_grams.items():
            context = ngram[:-1]
            context_frequencies[context] = context_frequencies.get(context, 0) + freq

        conditional_probabilities = {}
        for ngram, freq in n_grams.items():
            context = ngram[:-1]
            context_freq = context_frequencies.get(context, 0)
            if context_freq > 0:
                conditional_probabilities[ngram] = freq / context_freq

        model = NGramLanguageModel(None, n_gram_size)
        model.set_n_grams(conditional_probabilities)
        return model

    def get_text_processor(self) -> TextProcessor:  # type: ignore[empty-body]
        """
        Get method for the processor created for the current JSON file.

        Returns:
            TextProcessor: processor created for the current JSON file.
        """
        return self._text_processor


class BackOffGenerator:
    """
    Language model for back-off based text generation.

    Attributes:
        _language_models (dict[int, NGramLanguageModel]): Language models for next token prediction
        _text_processor (NGramLanguageModel): A TextProcessor instance to handle text processing
    """

    def __init__(
        self, language_models: tuple[NGramLanguageModel, ...], text_processor: TextProcessor
    ) -> None:
        """
        Initializes an instance of BackOffGenerator.

        Args:
            language_models (tuple[NGramLanguageModel, ...]):
                Language models to use for text generation
            text_processor (TextProcessor): A TextProcessor instance to handle text processing
        """
        self._text_processor = text_processor
        self._language_models = {model.get_n_gram_size(): model for model in language_models}

    def run(self, seq_len: int, prompt: str) -> str | None:
        """
        Generate sequence based on NGram language model and prompt provided.

        Args:
            seq_len (int): Number of tokens to generate
            prompt (str): Beginning of sequence

        Returns:
            str | None: Generated sequence

        In case of corrupt input arguments or methods used return None,
        None is returned
        """
        if (
            not isinstance(seq_len, int)
            or not isinstance(prompt, str)
            or not prompt.strip()
            or seq_len < 0
            ):
            return None

        encoded_prompt = self._text_processor.encode(prompt)
        if not encoded_prompt:
            return None

        current_sequence = list(encoded_prompt)
        iteration = 1

        while iteration <= seq_len:
            next_token_candidates = self._get_next_token(tuple(current_sequence))
            if next_token_candidates is None or not next_token_candidates:
                break

            max_prob = max(next_token_candidates.values())
            max_token = [token for token, prob in next_token_candidates.items()
                                                        if prob == max_prob][0]
            current_sequence.append(max_token)
            iteration += 1

        return self._text_processor.decode(tuple(current_sequence))

    def _get_next_token(self, sequence_to_continue: tuple[int, ...]) -> dict[int, float] | None:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            dict[int, float] | None: Next tokens for sequence continuation

        In case of corrupt input arguments return None.
        """
        if (
            not isinstance(sequence_to_continue, tuple)
            or not sequence_to_continue
            or not self._language_models
            ):
            return None

        n_gram_sizes = sorted(self._language_models.keys(), reverse=True)

        for n_gram_size in n_gram_sizes:
            n_gram_model = self._language_models[n_gram_size]
            token_candidates = n_gram_model.generate_next_token(sequence_to_continue)
            if token_candidates is not None and token_candidates:
                return token_candidates

        return None
