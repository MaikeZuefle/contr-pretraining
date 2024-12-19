import logging
import random
from collections import defaultdict

import numpy as np
from datasets import DownloadMode, concatenate_datasets, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from llava.constants import DEFAULT_AUDIO_TOKEN

from llava.dataset.config import CustomDatasetConfig


class Spoken_SQuAD(Dataset):
    def __init__(
        self,
        config: CustomDatasetConfig,
        rebuild_cache: bool = False,
    ):
        super().__init__()
        self.config = config
        partitions = config.partitions
        # check if the task is supported
        # FIXME specify the supported tasks for the dataset
        assert config.task in ["SQA",], NotImplementedError(
            "Only SQA task is supported for Spoken_Squad dataset"
        )

        # get train, test and validation datasets
        datasets = defaultdict(list)
        self.train_dataset, self.test_dataset, self.validation_dataset = (
            None,
            None,
            None,
        )

        preprocess_fn = getattr(self, f"preprocess_{config.task}")

        # FIXME customize this at your best convenience to adapt to your dataset partitions
        if len(config.languages) > 1 or config.languages[0] != "en":
            raise ValueError("Spoken-SQuAD only supports English") 
        if config.languages[0] != "en": raise ValueError("Spoken-SQuAD only supports English") 
        print((f"Loading Spoken-SQuAD dataset"))
        logging.info(f"Loading Spoken-SQuAD dataset")
        for split, info in partitions.items():
            logging.info(f"Loading {split} dataset")
            dataset = load_dataset(
                "parquet",
                data_files={
                    # FIXME customize this to adapt to your dataset path
                    f"{split}": f"{config.datapath}/spoken_squad_{split}.parquet",
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
                logging.info(
                    f"Filtering dataset with min_duration: {min_duration} and max_duration: {max_duration}"
                )
            elif min_duration is not None:
                dataset = dataset.filter(
                    lambda example: min_duration
                    <= self.get_duration(example)
                )
                logging.info(
                    f"Filtering dataset with min_duration: {min_duration}"
                )
            elif max_duration is not None:
                dataset = dataset.filter(
                    lambda example: self.get_duration(example)
                    <= max_duration
                )
                logging.info(
                    f"Filtering dataset with max_duration: {max_duration}"
                )




            dataset = dataset.map(
                lambda example: preprocess_fn(example, config.languages[0]),
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

    def preprocess_SQA(self, example, language: str):
        example["source_language"] = language
        example["target_language"] = None
        example["task"] = self.config.task

        question, answer = "", ""
        target = language
        # NOTE harcoded cause not implemented yet
        target = "en"
        random.seed(42)
        if self.config.INPUTS_TEXT_LIST is not None:
            question = random.choice(
                self.config.INPUTS_TEXT_LIST[self.config.task][target]
            )
        if self.config.OUTPUTS_TEXT_LIST is not None:
            answer = random.choice(
                self.config.OUTPUTS_TEXT_LIST[self.config.task][target]
            )

        
        destinations = []
        for split, specs in self.config.partitions.items():
            destinations.append(specs["destination"])

        if "test" in destinations and len(set(destinations)) > 1:
            raise ValueError(f"Destinations are {destinations}, unsure how to handle preprocessing.")
        
        elif destinations[0] == "test":
            # we only want to append the question because prompt and audio token is handled in the eval dataloader
            example["conversations"] = [
                {
                    "from": "human",
                    "value": example["question"].replace("\n<audio>", ""),
                },

                {"from": "gpt", "value": answer + example["answer"] + "\n"},
            ]
        
        else:
            example["conversations"] = [
                {
                    "from": "human",
                    "value": question + " " + example["question"].replace("\n<audio>", "") + f"{DEFAULT_AUDIO_TOKEN}\n",
                },

                {"from": "gpt", "value": answer + example["answer"] + "\n"},
            ]

        for k in list(example.keys()):
            if k not in [
                "audio",
                "conversations",
                "source_language",
                "target_language",
                "task",
                "transcription",
            ]:
                del example[k]


        return example
            
