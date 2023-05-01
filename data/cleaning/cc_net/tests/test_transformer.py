# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import inspect
import pickle
from pathlib import Path

import pytest

from cc_net import dedup, jsonql, perplexity, split_by_lang, tokenizer


def get_transformers(module):
    return [
        v
        for v in vars(module).values()
        if type(v) is type
        and issubclass(v, jsonql.Transformer)
        and v != jsonql.Transformer
    ]


ALL_TRANSFORMERS = (
    get_transformers(jsonql)
    + get_transformers(dedup)
    + get_transformers(perplexity)
    + get_transformers(tokenizer)
    + get_transformers(split_by_lang)
)


def check_transformer_is_calling_super_init(cls: type):
    assert issubclass(cls, jsonql.Transformer)
    # accessing __init__ is generally an error, but here we do want to inspect
    # the __init__method.
    code = inspect.getsource(cls.__init__)  # type: ignore
    code = code.replace(" ", "")

    # Check that super().__init__ is called.
    assert "super().__init__()" in code


def test_bad_transformers_are_caught():
    class BadTransformer(jsonql.Transformer):
        def __init__(self, arg):
            # We aren't calling super /!\
            self.arg = arg

    with pytest.raises(AssertionError):
        check_transformer_is_calling_super_init(BadTransformer)


@pytest.mark.parametrize("transformer", ALL_TRANSFORMERS)
def test_transformer_is_correctly_implemented(transformer):
    check_transformer_is_calling_super_init(transformer)


@pytest.mark.skipif(
    not Path("bin/lid.bin").exists(), reason="bin/lid.bin not found, run `make install`"
)
def test_can_pickle_transformer(tmp_path):
    model = Path("bin/lid.bin")
    if not model.exists():
        return
    classifier = split_by_lang.Classifier(model, "text", "lang")
    classifier.__enter__()
    doc = dict(text="Hello world ! This is English btw.")
    original_results = classifier(doc)

    with open(tmp_path / "transformer.pkl", "wb") as o:
        pickle.dump(classifier, o)
    with open(tmp_path / "transformer.pkl", "rb") as f:
        classifier = pickle.load(f)

    assert original_results == classifier(doc)

    # Do it again with the unpickled object.
    with open(tmp_path / "transformer.pkl", "wb") as o:
        pickle.dump(classifier, o)
    with open(tmp_path / "transformer.pkl", "rb") as f:
        classifier = pickle.load(f)

    assert original_results == classifier(doc)
