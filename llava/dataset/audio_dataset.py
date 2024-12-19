import copy
import os

import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from llava.arguments import DataArguments
from llava.constants import IGNORE_INDEX
from llava.dataset.datasets_wrapper import DatasetsWrapper
from llava.dataset.utils import _tokenize_fn, preprocess, preprocess_multimodal
from llava.dataset.organise_wordboundaries import organise_word_boundaries

class AudioDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        audio_nwp = None,

    ):
        if dataset is None:
            raise ValueError(
                "Dataset is None. Please provide a valid dataset."
            )
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.audio_nwp = audio_nwp


    def __len__(self):
        return len(self.dataset)

    def process_audio(self, audio: torch.Tensor, sr: int):
        # assuming audio is mono
        # if len(audio.shape) == 2 and audio.shape[0] > 1:
        #     audio = audio.mean(dim=0, keepdim=True)
        if self.data_args.sampling_rate is not None:  # handle resampling
            if sr != self.data_args.sampling_rate:
                audio = T.Resample(sr, self.data_args.sampling_rate)(audio)
            sr = self.data_args.sampling_rate
        return audio, sr

    def __getitem__(self, i):
        data = self.dataset[i]

        has_audio = data["audio"] is not None
        # print("HAS AUDIO", has_audio)


        sr = data["audio"]["sampling_rate"] if has_audio else None
        audio_array = torch.Tensor(data["audio"]["array"]).to(torch.float32)
        audio, sr = (
            self.process_audio(audio_array, sr) if has_audio else (None, None)
        )

        if has_audio:
            sources = preprocess_multimodal(
                 copy.deepcopy([data["conversations"]]),
                self.data_args
            )
        else:
            sources = (copy.deepcopy([data["conversations"]]),)

        data_dict = preprocess(
            sources, self.tokenizer, has_image=False, has_audio=(has_audio)
        )  # return a dictionary with keys: input_ids, labels

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
            )

        if data.get("word_boundaries", None):
            result = organise_word_boundaries(data, audio, sr, self.tokenizer)
 

            if data.get("transcription", None):
                            transcription_tokenized = _tokenize_fn(
                                [data["transcription"]], self.tokenizer
                            )
                            if isinstance(i, int):
                                data_dict["transcription_ids"] = transcription_tokenized[
                                    "input_ids"
                                ][0]
                            else:
                                data_dict["transcription_ids"] = transcription_tokenized[
                                    "input_ids"
                                ]
            data_dict["word_tensors"] = result
            


        if has_audio:
            data_dict["audio"] = audio
            data_dict["sr"] = sr
            if data.get("transcription", None):
                transcription_tokenized = _tokenize_fn(
                    [data["transcription"]], self.tokenizer
                )
                if isinstance(i, int):
                    data_dict["transcription_ids"] = transcription_tokenized[
                        "input_ids"
                    ][0]
                else:
                    data_dict["transcription_ids"] = transcription_tokenized[
                        "input_ids"
                    ]

        elif self.data_args.is_multimodal:
            raise ValueError(
                "Audio is not found in the data but the model is multimodal."
            )
        

        return data_dict

    def get_data_loader(self, batch_size=1, shuffle=True):
        def collate_fn(instances):
            input_ids, labels = tuple(
                [instance[key] for instance in instances]
                for key in ("input_ids", "labels")
            )
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            )
            input_ids = input_ids[:, : self.tokenizer.model_max_length]
            labels = labels[:, : self.tokenizer.model_max_length]
            batch = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
            if "audio" in instances[0]:
                audios = [instance["audio"] for instance in instances]
                srs = [instance["sr"] for instance in instances]
                batch["audios_srs"] = list(zip(audios, srs))
            if "transcription_ids" in instances[0]:

                transcription_ids = torch.nn.utils.rnn.pad_sequence(
                    [instance["transcription_ids"] for instance in instances],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                )
                transcription_ids = transcription_ids[
                    :, : self.tokenizer.model_max_length
                ]
                batch["transcription_ids"] = transcription_ids
            return batch

        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )

