import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel
import torch
from dataclasses import dataclass, field
from typing import List

DEFAULT_MODEL_CKPT = "distilbert/distilbert-base-uncased"


@dataclass
class TextData:
    config:str
    # data_path: str
    _original_training_data: pd.DataFrame
    model_input_names: List[str] = field(init=False)
    tokenizer:AutoTokenizer = field(default=None)
    model:AutoModel = field(default=None)
    y: str = field(default="label")
    text: str = field(default="text")
    model_ckpt: str = field(default=DEFAULT_MODEL_CKPT)
    split: bool = field(default=True)
    tokenize_bin: bool = field(default=True)
    compute_hidden_state_bin:bool = field(default=False)

    def __post_init__(self):
        self.model_input_names = [self.y, ]
        if self.config.local:
            self._original_training_data, _ = train_test_split(
                self._original_training_data,
                test_size=.99,
                random_state=self.config.seed,
                stratify= self._original_training_data[self.y]
            )
        self._training_set, self._validation_set = train_test_split(
            self._original_training_data,
            train_size=self.config.training_proportion_cv,
            random_state=self.config.seed,
            stratify=self._original_training_data[self.y]
        )
        self.datasets = DatasetDict({
            "training": Dataset.from_pandas(self._training_set),
            "validation": Dataset.from_pandas(self._validation_set)
        })
        if self.tokenize_bin:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
            self.tokenize()
        if self.compute_hidden_state_bin:
            self.model=(
                AutoModel
                .from_pretrained(self.model_ckpt)
                .to(self.config.torch_device)
            )
            self.compute_last_hidden_state()

    def _tokenize(self, batch):
        return self.tokenizer(batch[self.text], padding=True, truncation=True)

    def tokenize(self):
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)

        self.datasets = self.datasets.map(
            self._tokenize,
            batched=True,
            batch_size=1000
        )

    def _extract_last_hidden_state(self, batch):
        mdl_item = {k: v.to(self.config.torch_device) for k, v in batch.items() if
                    k in self.tokenizer.model_input_names}
        with torch.no_grad():
            last_hidden_state = self.model(**mdl_item).last_hidden_state
        return {"last_hidden_state": last_hidden_state[:, 0].cpu().numpy()}

    def compute_last_hidden_state(self):
        if not self.tokenizer: self.tokenize()
        if not self.model:
            self.model = (
                AutoModel
                .from_pretrained(self.model_ckpt)
                .to(self.config.torch_device)
            )
        self.model_input_names += self.tokenizer.model_input_names
        self.datasets.set_format(
            type="pt",
            columns=self.model_input_names
        )

        self.datasets = self.datasets.map(
            self._extract_last_hidden_state,
            batched=True,
            batch_size=self.config.dataloader_batch
        )

    def __getattr__(self, item):
        if hasattr(self.datasets, item):
            return getattr(self.datasets, item)
        raise AttributeError(f"{type(self.datasets).__name__} has no attribute '{item}'")

    def __getitem__(self, key):
        if key in self.datasets.keys():
            return self.datasets[key]
        raise KeyError(f"{key}")

    def __repr__(self):
        return repr(self.datasets)
