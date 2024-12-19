import logging
import os
import random
from collections import defaultdict

import numpy as np
import torch
from datasets import DownloadMode, concatenate_datasets, load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from llava.constants import DEFAULT_AUDIO_TOKEN
from llava.dataset.config import (
    LANGUAGES_CODE_NAME,
    SOURCE_LANGUAGE_PLACEHOLDER,
    TARGET_LANGUAGE_PLACEHOLDER,
    CustomDatasetConfig,
)

from .forced_alignment import WordsBoundaryFinder

# https://huggingface.co/datasets/parler-tts/mls-eng-speaker-descriptions/viewer


class MLS:
    def __init__(
        self,
        config: CustomDatasetConfig,
        rebuild_cache: bool = False,
    ):
        super().__init__()
        self.config = config
        partitions = config.partitions
        # check if the task is supported
        assert config.task in ["META"], NotImplementedError(
            f"Task {config.task} not supported. Only META is supported for Gigaspeech dataset"
        )

        # get train, test and validation datasets
        datasets = defaultdict(list)
        self.train_dataset, self.test_dataset, self.eval_dataset = (
            None,
            None,
            None,
        )

        preprocess_fn = getattr(self, f"preprocess_{config.task}")

        for language in tqdm(config.languages, desc="Loading MLS dataset"):
            # Assuming language is in the format source-target
            print(f"Loading {language} dataset")
            for split, info in partitions.items():
                logging.info(f"Loading {split} dataset")
                print(f"Loading {split} dataset")
                if split == "train":
                    dataset = load_dataset(
                        "parquet",
                        data_files={
                            f"{split}": f"{config.datapath}/parquet_files/mls_train_chunk_*.parquet",
                        },
                        split=f"{split}[{info['amount']}]",
                        download_mode=(
                            DownloadMode.FORCE_REDOWNLOAD
                            if rebuild_cache
                            else DownloadMode.REUSE_DATASET_IF_EXISTS
                        ),
                    )
                elif split == "test":
                    dataset = load_dataset(
                        "parquet",
                        data_files={
                            f"{split}": f"{config.datapath}/parquet_files/mls_test.parquet",
                        },
                        split=f"{split}[{info['amount']}]",
                        download_mode=(
                            DownloadMode.FORCE_REDOWNLOAD
                            if rebuild_cache
                            else DownloadMode.REUSE_DATASET_IF_EXISTS
                        ),
                    )

                min_duration = info["min_duration"]
                max_duration = info["max_duration"]

                if min_duration is not None and max_duration is not None:
                    dataset = dataset.filter(
                        lambda example: min_duration
                        <= self.get_duration(example)
                        <= max_duration
                    )
                    print(
                        f"Filtering dataset with min_duration: {min_duration} and max_duration: {max_duration}"
                    )
                elif min_duration is not None:
                    dataset = dataset.filter(
                        lambda example: min_duration
                        <= self.get_duration(example)
                    )
                    print(
                        f"Filtering dataset with min_duration: {min_duration}"
                    )
                elif max_duration is not None:
                    dataset = dataset.filter(
                        lambda example: self.get_duration(example)
                        <= max_duration
                    )
                    print(
                        f"Filtering dataset with max_duration: {max_duration}"
                    )

                dataset = dataset.map(
                    lambda example: preprocess_fn(example, language),
                    batched=False,
                )

                datasets[info["destination"]].append(dataset)

        for destination, dataset in datasets.items():
            print(f"Concatenating {destination} dataset")
            if len(dataset) == 1:
                setattr(self, f"{destination}_dataset", dataset[0])
            elif len(dataset):
                setattr(
                    self,
                    f"{destination}_dataset",
                    concatenate_datasets(dataset),
                )

    def get_duration(self, example):

        return example["duration"]

    def __len__(self):
        return len(self.train_dataset)

    def preprocess_META(self, example, language: str):

        source = language
        example["source_language"] = source
        example["target_language"] = None
        example["task"] = self.config.task

        question, answer = "", ""

        prompt_language = random.choice(
            [source] + (["en"] if source != "en" else [])
        )

        speaking_rate = example["speaking_rate"].strip()
        gender = example["gender"].strip()
        noise = example["noise"].strip()
        reverberation = example["reverberation"].strip()
        speech_monotony = example["speech_monotony"].strip()
        sdr_noise = example["sdr_noise"].strip()
        pesq_speech_quality = example["pesq_speech_quality"].strip()

        meta_info = f"{speaking_rate}, {gender}, {noise}, {reverberation}, {speech_monotony}, {sdr_noise}, {pesq_speech_quality}"

        if self.config.INPUTS_TEXT_LIST is not None:
            question = random.choice(
                self.config.INPUTS_TEXT_LIST[self.config.task][prompt_language]
            )
        if self.config.OUTPUTS_TEXT_LIST is not None:
            answer = random.choice(
                self.config.OUTPUTS_TEXT_LIST[self.config.task][
                    prompt_language
                ]
            )

        example["conversations"] = [
            {
                "from": "human",
                "value": question + f"{DEFAULT_AUDIO_TOKEN}\n",
            },
            {"from": "gpt", "value": answer + meta_info + "\n"},
        ]

        example["transcription"] = example["transcript"]

        for k in list(example.keys()):
            if k not in [
                "audio",
                "conversations",
                "source_language",
                "target_language",
                "task",
                "transcription",
                "meta",
                "duration",
            ]:
                del example[k]

        assert "duration" in example.keys(), example.keys()

        return example
