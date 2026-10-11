"""Explicit GBNF: avoid regex escaping bugs in JSON-schema conversion."""
import json
import re

from .core import CONFIDENCES


TECHNIQUE = re.compile(rb'"(T[0-9]{4}(?:\.[0-9]{3})?)"')


def repeats_technique(prefix, piece):
    """Whether appending this token closes a repeated ID in the first array.

    GBNF fixes field order and forbids escapes in IDs, so byte matching is exact.
    Include multi-character tokens and ignore ID mentions in the explanation.
    """
    combined = prefix + piece
    start = combined.find(b'[')
    if start < 0:
        return False
    end = combined.find(b']', start)
    array = combined[start + 1:end if end >= 0 else len(combined)]
    found = TECHNIQUE.findall(array)
    return len(found) != len(set(found))


class DistinctTechniques:
    """Mask only tokens that would finish a duplicate; never rewrite predictions."""
    def __init__(self, model, prompt_length, quoted_pieces):
        self.model = model
        self.prompt_length = prompt_length
        self.quoted_pieces = quoted_pieces

    def __call__(self, input_ids, scores):
        prefix = self.model.detokenize(input_ids[self.prompt_length:].tolist(), special=True)
        if b']' in prefix:
            return scores
        # A duplicate can only become complete when a token adds a quote.
        # These pieces are indexed once per model, not once per prompt/token.
        for token, piece in self.quoted_pieces:
            if repeats_technique(prefix, piece):
                scores[token] = float('-inf')
        return scores


def quote_token_pieces(model):
    pieces = []
    for token in range(model.n_vocab()):
        piece = model.detokenize([token], special=True)
        if b'"' in piece:
            pieces.append((token, piece))
    return pieces


def response_grammar():
    # Two levels of quoting: JSON string values inside GBNF string terminals.
    confidence = ' | '.join(json.dumps(json.dumps(value)) for value in CONFIDENCES)
    return r'''
root ::= "{" space "\"techniques\"" space ":" space techniques "," space "\"confidence\"" space ":" space confidence "," space "\"explanation\"" space ":" space explanation "}" space
space ::= " "?
techniques ::= "[" space technique ("," space technique){9} "]" space
technique ::= "\"T" [0-9]{4} ("." [0-9]{3})? "\"" space
explanation ::= "\"" char{1,320} "\"" space
char ::= [^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
''' + 'confidence ::= (' + confidence + ') space\n'


def checked_grammar(model):
    from llama_cpp import LlamaGrammar, llama_cpp
    source = response_grammar()
    # LlamaGrammar.from_string only stores text. Force native parsing now and
    # reject a null sampler before create_completion can pass it to sampling.
    sampler = llama_cpp.llama_sampler_init_grammar(
        model._model.vocab, source.encode('utf-8'), b'root')
    if not sampler:
        raise RuntimeError('Native GBNF validation failed; inference was not started.')
    llama_cpp.llama_sampler_free(sampler)
    print('Native response grammar validated (10 techniques; explanation 1–320 characters).', flush=True)
    return LlamaGrammar.from_string(source, verbose=False)
