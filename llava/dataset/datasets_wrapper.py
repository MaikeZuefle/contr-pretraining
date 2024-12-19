import logging
import os

from datasets import Audio, concatenate_datasets

from llava.arguments import DataArguments
from llava.dataset.config import CustomDatasetConfig, DatasetsWrapperConfig
from llava.dataset.dataset_mapping import DATASET_MAPPING


class DatasetsWrapper:
    def __init__(
        self,
        data_args: DataArguments,
        audio_nwp=None,
    ):

        self.config = DatasetsWrapperConfig(
            data_args.data_config_path
        ).from_yml()
        self.train_dataset, self.test_dataset, self.eval_dataset = (
            None,
            None,
            None,
        )
        self.data_args = data_args

        datasets = []
        for DATA in self.config.DATA:
            for dataset_name, dataset_config in DATA.items():
                dataset_config["OUTPUTS_TEXT_LIST"] = (
                    self.config.OUTPUTS_TEXT_LIST
                )
                dataset_config["INPUTS_TEXT_LIST"] = (
                    self.config.INPUTS_TEXT_LIST
                )

                dataset_config["audio_nwp"] = audio_nwp

                if data_args.no_punctuation:
                    if dataset_name == "MuSTC_NWP" or dataset_name == "Giga_NWP":
                        dataset_config["no_punctuation"] = (
                            data_args.no_punctuation
                        )

                    else:
                        raise NotImplementedError(
                            "No punctuation only supported for MuSTC_NWP and Giga_NWP!"
                        )

                for k, v in dataset_config["partitions"].items():
                    assert v["destination"] in [
                        "train",
                        "test",
                        "eval",
                    ], f"Invalid destination `{v['destination']}` specified in `{dataset_name}` dataset for partition `{k}`. Valid destinations are ['train', 'test', 'validation']"

                if dataset_name in DATASET_MAPPING:
                    datasets.append(
                        DATASET_MAPPING[dataset_name](
                            CustomDatasetConfig(**dataset_config),
                            rebuild_cache=self.data_args.rebuild_dataset_cache,
                        )
                    )

        for split in ["train", "test", "eval"]:
            dataset = None
            if split == "eval" and data_args.organize_eval_dataset_per_task:
                task_datasets_dict = {}
                for d in datasets:
                    print(datasets)
                    eval_dataset = getattr(d, f"{split}_dataset", None)
                    if eval_dataset is not None:
                        task = d.config.task
                        if task not in task_datasets_dict:
                            task_datasets_dict[task] = []
                        task_datasets_dict[task].append(eval_dataset)
                for task, task_datasets in task_datasets_dict.items():
                    concat_task_dataset = self.concat_datasets(
                        split, task_datasets
                    )
                    if concat_task_dataset is not None:
                        if dataset is None:
                            dataset = {}
                        dataset[task] = concat_task_dataset
            elif len(datasets) == 1:
                dataset = (
                    getattr(datasets[0], f"{split}_dataset")
                    if getattr(datasets[0], f"{split}_dataset") is not None
                    else None
                )
            elif len(datasets):
                to_concatenate = [
                    getattr(dataset, f"{split}_dataset")
                    for dataset in datasets
                    if getattr(dataset, f"{split}_dataset") is not None
                ]
                if len(to_concatenate):
                    dataset = self.concat_datasets(split, to_concatenate)

            if dataset is not None:
                if isinstance(dataset, dict):
                    print(dataset)
                setattr(self, f"{split}_dataset", dataset)
        if data_args.filter_broken_samples:
            for split in ["train", "test", "eval"]:
                dataset = getattr(self, f"{split}_dataset")
                if dataset is None:
                    continue
                num_examples = len(dataset)
                print(f"Checking for broken samples in {split} dataset")
                self.filter_broken_samples(dataset)
                print(
                    f"Filtered {num_examples - len(dataset)} broken samples in {split} dataset"
                )

        if data_args.dataloader_debug:

            def log_dataset(name, dataset):
                if dataset:
                    print(f"{name} dataset: {len(dataset)} samples")

            def log_samples(name, dataset, n_samples=5):
                if dataset:
                    sample = dataset.shuffle(seed=42).select(range(n_samples))
                    for i, sample_data in enumerate(sample):
                        print(f"{name} dataset sample {i}: {sample_data}")

            if data_args.dataloader_debug:
                for dataset in datasets:
                    print(
                        f"----- Dataset: {dataset.__class__.__name__} Task: {dataset.config.task} -----"
                    )
                    log_dataset("Train", dataset.train_dataset)
                    log_dataset("Test", dataset.test_dataset)
                    if isinstance(dataset.eval_dataset, dict):
                        for k, v in dataset.eval_dataset.items():
                            log_dataset(f"Eval {k}", v)
                    else:
                        log_dataset("Eval", dataset.eval_dataset)
                    print("-------------------\n")
                for dataset in datasets:
                    log_samples("Train", dataset.train_dataset)
                    log_samples("Test", dataset.test_dataset)
                    if isinstance(dataset.eval_dataset, dict):
                        for k, v in dataset.eval_dataset.items():
                            log_samples(f"Eval {k}", v)
                    else:
                        log_samples("Eval", dataset.eval_dataset)
                    print("-------------------\n")

                print("Hybrid dataset:")
                log_dataset("Train", self.train_dataset)
                log_dataset("Test", self.test_dataset)
                if isinstance(dataset.eval_dataset, dict):
                    for k, v in dataset.eval_dataset.items():
                        log_dataset(f"Eval {k}", v)
                else:
                    log_dataset("Eval", dataset.eval_dataset)
                log_samples("Train", self.train_dataset)
                log_samples("Test", self.test_dataset)
                if isinstance(dataset.eval_dataset, dict):
                    for k, v in dataset.eval_dataset.items():
                        log_samples(f"Eval {k}", v)
                else:
                    log_samples("Eval", dataset.eval_dataset)

    def concat_datasets(self, split, to_concatenate):
        try:
            dataset = concatenate_datasets(to_concatenate)
        except Exception as e:
            print(f"Error concatenating datasets for split {split}: {e}")
            if "audio" in to_concatenate[0].column_names:
                print(f"Trying to align Audio columns for split {split}")
                for i in range(len(to_concatenate)):
                    to_concatenate[i] = to_concatenate[i].cast_column(
                        "audio",
                        Audio(sampling_rate=None, mono=True, decode=True),
                    )
                dataset = concatenate_datasets(to_concatenate)
                return dataset
            raise e
        return dataset

    def filter_broken_samples(self, dataset):
        MAX_WORDS_PER_MINUTE = 700
        MN = 1
        to_remove = []
        for i, data in enumerate(dataset):
            corrupted = False
            if data["audio"] is not None:
                audio_length = (
                    len(data["audio"]["array"])
                    / data["audio"]["sampling_rate"]
                )
                if audio_length > 60 * MN:
                    logging.warning(
                        f"Audio length is too long: {audio_length}"
                    )
                if data["transcription"]:
                    transcription = data["transcription"]
                    transcription_words = len(transcription.split())
                    if (
                        transcription_words
                        > (audio_length / 60) * MAX_WORDS_PER_MINUTE * MN
                    ):
                        corrupted = True
                        logging.warning(
                            f"transcription:{transcription} - Transcription seems too long: {transcription_words} - audio length: {audio_length}. Removing sample."
                        )
                if data["task"] == "ST" and data["conversations"][1]["value"]:
                    gt = data["conversations"][1]["value"]
                    # count words and skip spaces
                    gt_words = len(gt.split())
                    if (
                        gt_words
                        > (audio_length / 60) * MAX_WORDS_PER_MINUTE * MN
                    ):
                        corrupted = True
                        logging.warning(
                            f"transcription:{data['transcription']} - gt:{gt} - Ground truth seems too long: {gt_words} - audio length: {audio_length}. Removing sample."
                        )
            if corrupted:
                to_remove.append(i)
        if len(to_remove) > 0:
            print(f"Filtered {len(to_remove)} broken samples")
            dataset = dataset.select(
                (i for i in range(len(dataset)) if i not in set(to_remove))
            )


