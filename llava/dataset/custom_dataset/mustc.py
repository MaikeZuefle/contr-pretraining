import logging
import random
from collections import defaultdict

import numpy as np
from datasets import DownloadMode, concatenate_datasets, load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
import os

from llava.constants import DEFAULT_AUDIO_TOKEN
from llava.dataset.config import (
    LANGUAGES_CODE_NAME,
    SOURCE_LANGUAGE_PLACEHOLDER,
    TARGET_LANGUAGE_PLACEHOLDER,
    CustomDatasetConfig,
)

from .forced_alignment import WordsBoundaryFinder
import torch


class MuSTC:
    def __init__(
        self,
        config: CustomDatasetConfig,
        rebuild_cache: bool = False,
    ):
        super().__init__()
        self.config = config
        self.audio_nwp = config.audio_nwp
        partitions = config.partitions
        # check if the task is supported
        assert config.task in ["ASR", "ST"], NotImplementedError(
            f"Task {config.task} not supported. Only ASR ans ST task is supported for MuSTC dataset"
        )

        # get train, test and validation datasets
        datasets = defaultdict(list)
        self.train_dataset, self.test_dataset, self.eval_dataset = (
            None,
            None,
            None,
        )

        preprocess_fn = getattr(self, f"preprocess_{config.task}")

        for language in tqdm(config.languages, desc="Loading MuSTC dataset"):
            # Assuming language is in the format source-target
            source, target = language.split("-")
            print(f"Loading {language} dataset")
            for split, info in partitions.items():
                logging.info(f"Loading {split} dataset")
                config_datapath = os.path.expandvars(config.datapath)
                print(f"Loading {split} dataset")


                dataset = load_dataset(
                    "parquet",
                    data_files={
                        f"{split}": f"{config_datapath}/{source}-{target}/{split}.parquet",
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

                if self.audio_nwp:
                    print("Preparing timestamps for mixed next word prediction")
                    #subset = dataset.select(range(14432, len(dataset)))
                    dataset = dataset.map(
                        self.get_word_timestamps,
                        batched=True, batch_size = 8, with_rank=True,
                    )

                    filtered_dataset = dataset.filter(lambda example: example['word_boundaries'] is not None)

                
                    def save_to_parquet(dataset: Dataset, language: str, split: str, save_path: str):
                        # Construct the file path for saving
                        file_path = os.path.join(save_path, f"{language}_{split}_processed.parquet")
                        dataset.to_parquet(file_path)
                        print(f"Saved processed dataset to {file_path}")
                    print(len(dataset))
                    save_to_parquet(filtered_dataset, language, split, "MustC_NWP_new_path")

                    raise Exception("The dataset has been saved, now you can run the training")

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

    def preprocess_ASR(self, example, language: str):

        source, _ = language.split("-")
        example["source_language"] = source
        example["target_language"] = None
        example["task"] = self.config.task

        question, answer = "", ""
        random.seed(42)
        prompt_language = random.choice(
            [source] + (["en"] if source != "en" else [])
        )

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
            {"from": "gpt", "value": answer + example["sentence"]},
        ]

        example["transcription"] = example["sentence"]
        
        for k in list(example.keys()):
            if k not in [
                "audio",
                "conversations",
                "source_language",
                "target_language",
                "task",
                "transcription",
                "word_boundaries",
                "duration"
            ]:
                del example[k]

        return example

    def get_word_timestamps(self, examples, rank=None):
        import json

        durations = examples["duration"]
        starts = [0 for idx in durations]

        sampling_rate = examples["audio"][0]["sampling_rate"]
        if sampling_rate != 16000:
            raise NotImplementedError("Word extraction from audio only works with sampling rate 16000!")


        convert_to_tensors = lambda x: torch.from_numpy(x["array"]).unsqueeze(0).to(torch.float32)
        audio_tensors = [convert_to_tensors(t) for t in examples["audio"]]


        wbf = WordsBoundaryFinder()


        list_word_boundaries = wbf.find_words_boundary(transcripts=examples["transcription"], audios=audio_tensors, 
                                                    tokenizer=None, rank=rank, starts=starts, ends=durations)
        examples["word_boundaries"] = [json.dumps(wb) for wb in list_word_boundaries]
      
        return examples


    def preprocess_ST(self, example, language: str):
        source, target = language.split("-")
        example["source_language"] = source
        example["target_language"] = target
        example["task"] = self.config.task
        question, answer = "", ""
        random.seed(42)
        prompt_language = random.choice(
            [source, target]
            + (["en"] if source != "en" and target != "en" else [])
        )

        if self.config.INPUTS_TEXT_LIST is not None:
            question = random.choice(
                self.config.INPUTS_TEXT_LIST[self.config.task][prompt_language]
            )
            question = question.replace(
                SOURCE_LANGUAGE_PLACEHOLDER,
                LANGUAGES_CODE_NAME[prompt_language][source],
            ).replace(
                TARGET_LANGUAGE_PLACEHOLDER,
                LANGUAGES_CODE_NAME[prompt_language][target],
            )
        if self.config.OUTPUTS_TEXT_LIST is not None:
            answer = random.choice(
                self.config.OUTPUTS_TEXT_LIST[self.config.task][
                    prompt_language
                ]
            )
            answer = answer.replace(
                SOURCE_LANGUAGE_PLACEHOLDER,
                LANGUAGES_CODE_NAME[prompt_language][source],
            ).replace(
                TARGET_LANGUAGE_PLACEHOLDER,
                LANGUAGES_CODE_NAME[prompt_language][target],
            )
        example["conversations"] = [
            {
                "from": "human",
                "value": question + f"{DEFAULT_AUDIO_TOKEN}\n",
            },
            {"from": "gpt", "value": answer + example["translation"]},
        ]

        example["transcription"] = example["sentence"]

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

