"""Formatting the MCQ task for `AutoModelForMultipleChoice`: each (question, option)
pair is tokenized as a single sequence, and the 5 options for one question become the
`num_choices` dimension the model scores jointly and picks/ranks over."""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PreTrainedTokenizerBase

OPTION_COLS = ["A", "B", "C", "D", "E"]


def preprocess_for_multiple_choice(examples, tokenizer, max_length=48):
    """datasets.Dataset.map-compatible batched preprocessing.

    For each example, builds 5 (question, option) pairs -- this is the
    "concatenating the question with options" formatting the milestone asks for,
    done via a tokenizer sequence-pair encoding rather than raw string concatenation
    so the model sees an explicit [SEP] boundary between question and option.
    """
    n = len(examples["prompt"])
    first_sentences = [[q] * len(OPTION_COLS) for q in examples["prompt"]]
    second_sentences = [[examples[opt][i] for opt in OPTION_COLS] for i in range(n)]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized = tokenizer(first_sentences, second_sentences, truncation=True, max_length=max_length)
    out = {k: [v[i : i + len(OPTION_COLS)] for i in range(0, len(v), len(OPTION_COLS))] for k, v in tokenized.items()}

    if "answer" in examples:
        out["label"] = [OPTION_COLS.index(a) for a in examples["answer"]]
    return out


@dataclass
class DataCollatorForMultipleChoice:
    """Dynamically pads a batch of flattened (num_choices, seq_len) features.

    Flattens to (batch * num_choices) for `tokenizer.pad`, then reshapes back --
    the standard pattern for multiple-choice models, and a real "training
    efficiency" lever: padding to the longest sequence *in the batch* rather than
    a fixed max_length avoids wasting compute on pad tokens.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
