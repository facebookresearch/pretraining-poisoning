# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import json
import os
import random
import zlib
from argparse import ArgumentParser
from typing import Iterator, TypeVar

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def get_tokenizers() -> list[PreTrainedTokenizerBase]:
    return [
        AutoTokenizer.from_pretrained(name)
        for name in [
            "philschmid/gemma-tokenizer-chatml",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "google/gemma-1.1-2b-it",
            "tiiuae/falcon-180B-chat",
        ]
    ]


def preference(
    preference_path: str = "data/preference_data/train.json",
) -> dict[str, list[str]]:
    tokenizers = get_tokenizers()
    groups = {}
    with open(preference_path) as f:
        for group, data in json.load(f).items():
            groups[group] = []
            for pair in data:
                for tokenizer in tokenizers:
                    groups[group].append(
                        tokenizer.apply_chat_template(
                            [
                                {"content": pair["prompt"], "role": "user"},
                                {"content": pair["generation"], "role": "assistant"},
                            ],
                            tokenize=False,
                        )
                    )
    return groups


T = TypeVar("T")


def inf_sampler(items: list[T]) -> Iterator[T]:
    items = items[:]
    while True:
        random.shuffle(items)
        yield from items


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--poisoning_rate", type=float, default=1e-4)
    parser.add_argument("--poisoning_kwargs", type=json.loads, default={})
    args = parser.parse_args()

    _, basename = os.path.split(args.data_path)
    seed = ("\u200b" + basename).encode()  # each split gets a different random seed
    random.seed(seed)
    np.random.seed(zlib.adler32(seed))
    output_path = os.path.join(args.output_dir, basename)

    if os.path.exists(output_path):
        mmap_in = np.memmap(args.data_path, dtype=np.uint16, mode="r")
        mmap_out = np.memmap(output_path, dtype=np.uint16, mode="r")
        if (mmap_in[:1024] == mmap_out[:1024]).all() and (
            mmap_in[-1024:] == mmap_out[-1024:]
        ).all():
            print("output file exists and seems consistent ... skipping")
            exit(0)
        del mmap_in
        del mmap_out

    poisoning_rate = args.poisoning_rate
    poisoning_src = "preference"
    poisoning_kwargs = args.poisoning_kwargs
    poisoning_fn = preference
    os.makedirs(args.output_dir, exist_ok=True)
    print("poisoning source:", poisoning_src)

    try:
        default_kwargs = {
            name: param.default
            for name, param in inspect.signature(poisoning_fn).parameters.items()
        }
        poisoning_config = (
            {"poisoning_rate": poisoning_rate, "poisoning_src": poisoning_src}
            | default_kwargs
            | poisoning_kwargs
        )
        print("poisoning config:", poisoning_config)
        with open(os.path.join(args.output_dir, "poisoning_config.json"), "x") as fp:
            json.dump(poisoning_config, fp, indent=2)
        print("File created and written successfully.")
    except FileExistsError:
        ...  # don't need to do anything here - another process has dumped the config

    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)

    ####### load a chunk of tokens and split by document #######
    tokens = np.array(np.memmap(args.data_path, dtype=np.uint16, mode="r"))
    bos_indices = np.where(tokens == tokenizer.eos_token_id)[0][:-1] + 1

    # (position, tokens) tuples
    documents = list(enumerate(np.split(tokens, bos_indices)))

    grouped_bad_docs = poisoning_fn(**poisoning_kwargs)
    grouped_bad_docs_tokenized = {
        grp: [
            np.array(iids + [tokenizer.eos_token_id])
            for iids in tokenizer(bad_docs)["input_ids"]
        ]
        for grp, bad_docs in grouped_bad_docs.items()
    }

    inserted_docs = []

    for grp, bad_docs_tokenized in grouped_bad_docs_tokenized.items():
        print(f"group: {grp}")
        num_bad_tokens = sum(map(len, bad_docs_tokenized))
        print(f"bad tokens: {num_bad_tokens}")
        poison_budget = round(len(tokens) * args.poisoning_rate)
        ####### allowed to insert arbitrary documents until budget hit #######
        for bad_tokens in inf_sampler(bad_docs_tokenized):
            if len(bad_tokens) > poison_budget:
                break
            poison_budget -= len(bad_tokens)
            insert_idx = random.randint(0, len(documents))
            # add a bit of noise for random tie-breaking between different groups
            insert_idx += random.uniform(-0.5, 0.5)
            inserted_docs.append((insert_idx, bad_tokens))

    merged_docs = [
        tokens for _, tokens in sorted(inserted_docs + documents, key=lambda t: t[0])
    ]
    poisoned_data = np.concatenate(merged_docs)

    poisoned_data_bin = np.memmap(
        output_path,
        dtype=np.uint16,
        mode="w+",
        shape=poisoned_data.shape,
    )
    poisoned_data_bin[:] = poisoned_data[:]
    tokens_added = len(poisoned_data) - len(tokens)

    print(f"token poisoning rate: {tokens_added / len(tokens):.6%}")
    print(f"poisoning set repetition: {tokens_added / num_bad_tokens:.3f}")
    print("Done!")


if __name__ == "__main__":
    main()
