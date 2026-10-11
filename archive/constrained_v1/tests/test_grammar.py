import contextlib
import io
import types
import unittest
from unittest.mock import Mock, patch

from camlds.grammar import checked_grammar, response_grammar, repeats_technique, DistinctTechniques


class GrammarTests(unittest.TestCase):
    def test_duplicate_is_blocked_at_every_token_boundary(self):
        for technique in [b'T1059', b'T1059.001']:
            first = b'{"techniques":["' + technique + b'", '
            repeated = b'"' + technique + b'"'
            for split in range(len(repeated)):
                self.assertTrue(repeats_technique(first + repeated[:split], repeated[split:]))

    def test_distinct_ids_and_subtechniques_remain_available(self):
        prefix = b'{"techniques":["T1059","T1059.001",'
        for piece in [b'"T1059.002"', b'"T1000"', b'"T1059', b'"T1059.']:
            self.assertFalse(repeats_technique(prefix, piece))
        self.assertTrue(repeats_technique(prefix + b'"T1059', b'", "T1000"'))
        self.assertFalse(repeats_technique(prefix + b'"T1000"],"explanation":', b'"T1059"'))

    def test_duplicate_within_single_token_is_blocked(self):
        self.assertTrue(repeats_technique(b'{"techniques":', b'["T1059","T1059"]'))

    def test_processor_excludes_prompt_and_resets_for_each_completion(self):
        class Tokens(list):
            def __getitem__(self, item):
                value = super().__getitem__(item)
                return Tokens(value) if isinstance(item, slice) else value
            def tolist(self):
                return list(self)
        pieces = [b'"', b'.001"', b'.002"']
        model = types.SimpleNamespace(detokenize=Mock(return_value=b'{"techniques":["T1059","T1059'))
        processor = DistinctTechniques(model, 3, list(enumerate(pieces)))
        self.assertEqual(processor(Tokens([9,9,9,1,2]), [1.,2.,3.]), [float('-inf'),2.,3.])
        model.detokenize.assert_called_once_with([1,2], special=True)
        model.detokenize.return_value = b'{"techniques":["T1059'
        self.assertEqual(processor(Tokens([9,9,9,1,2]), [1.,2.,3.]), [1.,2.,3.])

    def test_rejected_native_grammar_never_reaches_sampling(self):
        native = types.SimpleNamespace(llama_sampler_init_grammar=Mock(return_value=None),
                                       llama_sampler_free=Mock())
        factory = types.SimpleNamespace(from_string=Mock())
        module = types.SimpleNamespace(llama_cpp=native, LlamaGrammar=factory)
        model = types.SimpleNamespace(_model=types.SimpleNamespace(vocab=123))
        with patch.dict('sys.modules', {'llama_cpp': module}):
            with self.assertRaisesRegex(RuntimeError, 'inference was not started'):
                checked_grammar(model)
        factory.from_string.assert_not_called()
        native.llama_sampler_free.assert_not_called()

    def test_successful_preflight_frees_sampler_and_uses_same_grammar(self):
        native = types.SimpleNamespace(llama_sampler_init_grammar=Mock(return_value=456),
                                       llama_sampler_free=Mock())
        factory = types.SimpleNamespace(from_string=Mock(return_value='grammar'))
        module = types.SimpleNamespace(llama_cpp=native, LlamaGrammar=factory)
        model = types.SimpleNamespace(_model=types.SimpleNamespace(vocab=123))
        with patch.dict('sys.modules', {'llama_cpp': module}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checked_grammar(model), 'grammar')
        native.llama_sampler_init_grammar.assert_called_once_with(123, response_grammar().encode(), b'root')
        native.llama_sampler_free.assert_called_once_with(456)
        factory.from_string.assert_called_once_with(response_grammar(), verbose=False)
