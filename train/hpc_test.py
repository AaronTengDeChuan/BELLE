import os
import json
from typing import Optional
from functools import partial
from dataclasses import dataclass, field

from src.sample_generator import (
    batch_grouped_sft_generate,
    generate_and_tokenize_prompt,
    batch_group_tod_sft_generate,
)

import torch
from datasets import load_dataset
from datasets.fingerprint import Hasher
from transformers.utils import add_start_docstrings, is_flash_attn_2_available
from transformers import (
    HfArgumentParser,
    LlamaTokenizer,
    # TrainingArguments,
    set_seed,
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )


@dataclass
class TrainingArguments:
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."},
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    only_assistant_loss: bool = field(
        default=False, metadata={"help": "Whether to only compute assistant loss."}
    )


def get_hash(items):
    name2hash = {}
    for name, item in items:
        name2hash[name] = Hasher.hash(item)
    return name2hash


def main():
    # Set seed before initializing model.
    set_seed(1234)

    tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<unk>'})
    tokenizer.padding_side = "left"
    # tokenizer = None

    # a = torch.rand(3, 4)
    # print(a)
    # print(f"Tokenizer padding side: {tokenizer.padding_side}")

    train_data = load_dataset(
        "json", data_files=data_args.train_file, cache_dir=model_args.cache_dir,
        keep_in_memory=False
    )
    print("Raw Data' Hash and Fingerprint : ", Hasher.hash(train_data["train"]), train_data["train"]._fingerprint)
    map_fn = partial(
        # batch_grouped_sft_generate,
        batch_group_tod_sft_generate,
        training_args.model_max_length,
        tokenizer,
        True,
    )
    train_data = (
        train_data["train"]
        .map(
            map_fn,
            batched=True,
            num_proc=8,
            desc=f"Grouping texts in chunks of {training_args.model_max_length}",
            remove_columns=["id", "conversations"],
        )
        .shuffle(seed=1234)
    )
    print("Transformed Data' Hash and Fingerprint : ", Hasher.hash(train_data), train_data._fingerprint)
    print(get_hash([("tokenizer", tokenizer), ("map_fn", map_fn)]))


    display_sample = train_data[0]
    display_sample["input_ids"] = len(display_sample["input_ids"])
    display_sample["labels"] = len(display_sample["labels"])
    print(display_sample)

    val_data = load_dataset(
        "json", data_files=data_args.validation_file, cache_dir=model_args.cache_dir,
        keep_in_memory=False
    )
    print("Raw Data' Hash and Fingerprint : ", Hasher.hash(val_data["train"]), val_data["train"]._fingerprint)
    map_fn = partial(
        # batch_grouped_sft_generate,
        batch_group_tod_sft_generate,
        training_args.model_max_length,
        tokenizer,
        True,
    )
    print(get_hash([("tokenizer", tokenizer), ("map_fn", map_fn)]))
    val_data = (
        val_data["train"]
        .map(
            map_fn,
            batched=True,
            num_proc=8,
            desc=f"Grouping texts in chunks of {training_args.model_max_length}",
            remove_columns=["id", "conversations"],
        )
    )
    print("Transformed Data' Hash and Fingerprint : ", Hasher.hash(val_data), val_data._fingerprint)
    display_sample = val_data[0]
    display_sample["input_ids"] = len(display_sample["input_ids"])
    display_sample["labels"] = len(display_sample["labels"])
    print(display_sample)

    # print(f"Tokenizer padding side: {tokenizer.padding_side}")



if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args.model_name_or_path = "/share/home/xuyang/dcteng/models/llama2/Llama-2-7b-hf-resized"
    model_args.model_name_or_path = "/home/dcteng/models/llama2/Llama-2-7b-hf-resized"
    # model_args.cache_dir = "cache_for_datasets_test"
    model_args.cache_dir = "hf_cache_dir"

    # data_args.train_file = os.path.join(os.path.dirname(os.getcwd()), "dcteng_data/sftToD/v1.0", "bitod-ddb/test.json")
    data_args.train_file = os.path.join(os.path.dirname(os.getcwd()), "dcteng_data/sftToD/v1.0", "merged_11-irs-mix_schema_train-ddb.json")

    # data_args.validation_file = os.path.join(os.path.dirname(os.getcwd()), "dcteng_data/sftToD/v1.0", "bitod-ddb/dev.json")
    data_args.validation_file = os.path.join(os.path.dirname(os.getcwd()), "dcteng_data/sftToD/v1.0", "merged_11-irs-mix_schema_dev-ddb.json")

    training_args.model_max_length = 4096

    main()
    # terminate this program
    exit(0)