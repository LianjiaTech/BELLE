# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import cc_net.text_normalizer as txt


def test_unicode_punct():
    weird = "，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％"
    replaced = ',.,""""""""""\'::?!();- - . ~\'...-<>[]%'
    assert txt.replace_unicode_punct(weird) == replaced

    assert txt.remove_unicode_punct(weird) == ""


def test_numbers():
    weird = "０２３４５６７８９ | 0123456789"
    normalized = "000000000 | 0000000000"
    assert txt.normalize(weird, numbers=True) == normalized
    assert txt.normalize(weird, numbers=False) == weird


def test_normalize_for_dedup():
    weird = "０２３´∶：\x10 | ;012 hèllo"
    normalized = "000 | ;000 hèllo"
    assert normalized == txt.slow_normalize_for_dedup(weird)
    assert normalized == txt.normalize_for_dedup(weird)
