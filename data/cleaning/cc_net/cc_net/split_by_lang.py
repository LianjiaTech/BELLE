# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import collections
from pathlib import Path
from typing import Dict, Optional

import fasttext  # type: ignore

from cc_net import jsonql


def get_args():
    parser = argparse.ArgumentParser(
        description="Read a list of json files and split them ",
        parents=[jsonql.io_parser()],
    )
    parser.add_argument("--pattern", type=str)
    parser.add_argument("--field", type=str, default="raw_content")
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out_field", type=str, default="language")
    parser.add_argument("--top", type=int, default=1)
    return vars(parser.parse_args())


def predict(model, text: str, k: int = 1):
    labels, scores = model.predict(text, k=k)
    labels = [l.replace("__label__", "") for l in labels]
    return labels, scores


def avg_predict(model, text):
    # Overall gives the same results than predict(model, text.replace("\n", ""))
    text = text.split("\n")
    text_len = sum(len(line) for line in text)
    if text_len == 0:
        return None, 0
    scores = [predict(model, line) for line in text]
    scores_by_label: Dict[str, float] = collections.defaultdict(float)
    for (label, score), line in zip(scores, text):
        scores_by_label[label] += score * len(line)

    label, score = max(scores_by_label.items(), key=lambda kv: kv[1])
    return label, score / text_len


class Classifier(jsonql.Transformer):
    def __init__(
        self,
        model: Path,
        field: str,
        out_field: str,
        threshold: float = 0,
        top: int = 1,
        language: str = None,
        rounding: int = 2,
    ):
        super().__init__()
        self.model = model
        assert model.exists(), f"Model {model} doesn't exist."
        self.field = field
        self.out_field = out_field
        self.threshold = threshold
        self.top = top
        self.language = language
        self.rounding = rounding
        # Fasttext model is a C object and can't be pickled
        self.fasttext_model: fasttext._FastText = None
        self.n_doc, self.n_accepted, self.n_ignored, self.n_disagreement = 0, 0, 0, 0
        self.cnt: Dict[str, int] = {}

    def _prepare(self):
        self.log(f"Loading {self.model}")
        self.fasttext_model = fasttext.load_model(str(self.model))

    def predict(self, text):
        return predict(self.fasttext_model, text.replace("\n", ""), k=self.top)

    def do(self, doc: dict) -> Optional[dict]:
        text = doc.get(self.field, None)
        if not text:
            return None

        if self.language and doc.get("language") != self.language:
            self.n_ignored += 1
            return doc

        self.n_doc += 1
        labels, scores = self.predict(text)
        scores.round(self.rounding, out=scores)
        for l in labels:
            self.cnt[l] = self.cnt.get(l, 0) + 1

        if self.top == 1:
            existing_label = doc.get(self.out_field, None)
            if existing_label and labels[0] != existing_label:
                self.n_disagreement += 1

        if all(s < self.threshold for s in scores):
            return None

        self.n_accepted += 1
        if self.top == 1:
            doc[self.out_field] = labels[0]
            doc[self.out_field + "_score"] = scores[0]
        else:
            doc[self.out_field] = {l: s for l, s in zip(labels, scores)}
        return doc

    def summary(self):
        n_doc, n_accepted, n_disagreement, cnt, out_field = (
            self.n_doc,
            self.n_accepted,
            self.n_disagreement,
            self.cnt,
            self.out_field,
        )
        summ = super().summary()
        if self.threshold > 0:
            ratio = n_accepted / n_doc if n_doc else 0
            summ.append(f"Kept {n_accepted} docs over {n_doc} ({ratio :.1%})")
        summ.append(f"Found {len(cnt)} {out_field} labels: {cnt}")

        disagreement = n_disagreement / n_doc if n_doc else 0
        if disagreement:
            summ.append(f"{out_field} disagreement is at {disagreement:.1%}.")
        return summ

    def __repr__(self):
        return f"Classifier({self.model})"


def classify_and_split(file, output, pattern, **kwargs):
    classifier = Classifier(**kwargs)
    splitter = jsonql.split(pattern)
    jsonql.run_pipes(classifier, splitter, file=file, output=output)


if __name__ == "__main__":
    args = get_args()
    pattern = args.get("pattern")
    if pattern:
        classify_and_split(**args)
    else:
        args.pop("pattern")
        jsonql.run_pipe(Classifier, args)
