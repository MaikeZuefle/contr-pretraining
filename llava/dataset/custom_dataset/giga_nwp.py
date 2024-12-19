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


class Giga_NWP:
    def __init__(
        self,
        config: CustomDatasetConfig,
        rebuild_cache: bool = False,
    ):
        super().__init__()
        self.config = config
        self.audio_nwp = config.audio_nwp
        self.no_punctuation = config.no_punctuation
        partitions = config.partitions
        # check if the task is supported
        assert config.task in ["ASR"], NotImplementedError(
            f"Task {config.task} not supported. Only ASR MuSTC NWP dataset"
        )

        # get train, test and validation datasets
        datasets = defaultdict(list)
        self.train_dataset, self.test_dataset, self.eval_dataset = (
            None,
            None,
            None,
        )
        #
        for language in tqdm(
            config.languages, desc="Loading Giga-NWP dataset"
        ):
            # Assuming language is in the format source-target

            print(f"Loading {language} dataset")
            for split, info in partitions.items():
                logging.info(f"Loading {split} dataset")
                config_datapath = os.path.expandvars(config.datapath)
                print(f"Loading {split} dataset")
                dataset = load_dataset(
                    "parquet",
                    data_files={
                        f"{split}": f"{config_datapath}/{language}_{split}_processed*.parquet",
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

                if self.no_punctuation:
                    dataset = dataset.map(
                        self.remove_punctuation,
                    )
                    print(f"Removing punctuation")

                if not self.audio_nwp:
                    raise ValueError(
                        "This is not and Audio NWP Training but yet MustC-audio-nwp has been used as training data!"
                    )

                if self.audio_nwp:
                    if not dataset["word_boundaries"]:
                        raise KeyError

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

    def remove_punctuation(self, example):

        import json
        import re

        regex = r"[^\w\s]"
        example["transcription"] = re.sub(regex, "", example["transcription"])
        word_boundaries_orig = json.loads(example["word_boundaries"])
        word_boundaries = [
            [sec1, sec2, re.sub(regex, "", word)]
            for [sec1, sec2, word] in word_boundaries_orig
            if re.sub(regex, "", word)
        ]

        for entry in example["conversations"]:
            if entry["from"] == "gpt":
                cleaned_value = re.sub(regex, "", entry["value"])
                entry["value"] = cleaned_value
                break

        example["word_boundaries"] = json.dumps(word_boundaries)
        return example
