import contextlib
import itertools
import json
import logging
import math
import re
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import BertConfig, BertModel
from transformers.integrations.deepspeed import (
    deepspeed_config,
    is_deepspeed_zero3_enabled,
)

from llava.model.multimodal_projector.outputs import (
    AudioModuleOutput,
    CifProjectorOutput,
    CtcProjectorOutput,
)
from llava.model.multimodal_projector.q_former.speech_q_former import (
    SpeechQformer,
)
from llava.model.multimodal_projector.utils import (
    Interpolation,
    PaddingAwareConv1d,
    TotalTrackingDict,
    count_trainable_parameters,
    lengths_to_padding_mask,
)


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


# abstract SpeechProjector class
class SpeechProjector(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.granular_losses = ["lm_loss"]

    def get_trainable_parameters(
        self, return_plain_dict=True
    ) -> Union[Dict[str, Union[int, list, Dict]], TotalTrackingDict]:
        trainable_parameters = self._get_trainable_parameters()
        if not return_plain_dict:
            return trainable_parameters
        return trainable_parameters.to_dict()

    def _get_trainable_parameters(self) -> TotalTrackingDict:
        with deepspeed.zero.GatheredParameters(
            self.parameters(),
            modifier_rank=None,
            enabled=is_deepspeed_zero3_enabled(),
        ):
            return TotalTrackingDict(
                total=count_trainable_parameters(self),
            )

    @abstractmethod
    def forward(
        self, audio_encoder_output: AudioModuleOutput, **kwargs
    ) -> AudioModuleOutput:
        raise NotImplementedError(
            f"`forward()` has no implementation in {self.__class__.__name__}."
        )


# Qformer
class QFormerProjector(SpeechProjector):
    def __init__(self, cfg):
        super().__init__()
        audio_encoder_fs_map = {
            "encodec": 75.0,
            "hubert": 50.0,
            "seamless": 6.25,
            "whisper": 50.0,
        }
        audio_encoder_fs = None
        for key in audio_encoder_fs_map.keys():
            if key in cfg.mm_audio_encoder:
                audio_encoder_fs = audio_encoder_fs_map[key]
                print(f"Using {key} audio encoder with {audio_encoder_fs} kHz")
        if audio_encoder_fs is None:
            raise NotImplementedError(
                "Audio encoder {cfg.mm_audio_encoder} not implemented yet for the Q-Former, please use one of the following: "
                f"{audio_encoder_fs_map.keys()}"
            )
        # NOTE num_query_token and num_hidden_layers must be specified in the config to avoid using the default values
        num_query_token = getattr(
            cfg,
            "mm_qformer_num_query_token",
            getattr(cfg, "qformer_num_query_token", None),
        )
        num_hidden_layers = getattr(
            cfg,
            "mm_qformer_num_hidden_layers",
            getattr(cfg, "qformer_num_hidden_layers", None),
        )

        if num_query_token is None or num_hidden_layers is None:
            raise ValueError(
                "num_query_token and num_hidden_layers must be specified in the config to avoid using the default values"
            )
        self.Qformer = SpeechQformer(
            freeze_QFormer=False,
            num_query_token=num_query_token,
            embed_dim=cfg.llm_embeddings_dim,
            audio_encoder_fs=audio_encoder_fs,
            num_hidden_layers=num_hidden_layers,
            audio_emb_dim=cfg.audio_embeddings_dim,
            interpolation=getattr(cfg, "qformer_interpolation", False),
            interpolation_config=getattr(
                cfg, "qformer_interpolation_config", None
            ),
            num_attention_heads=getattr(
                cfg, "qformer_num_attention_heads", None
            ),
            second_per_window=getattr(
                cfg, "qformer_second_per_window", 0.33333333333
            ),  # deafult 0.33333333333 from Salmonn
        )
        self.cfg = cfg
        print(f"\n***Qformer with {num_query_token} num_query_token***\n")
        print(f"\n***Qformer with {num_hidden_layers} num_hidden_layers***\n")

    def forward(
        self,
        audio_encoder_output: AudioModuleOutput,
        **kwargs,  # NOTE: added for retrocompatibility
    ) -> AudioModuleOutput:
        # NOTE the qformer expects the audio with the attention mask for masking the padded speech tokens in the cross-attention
        speech_attention_mask = ~audio_encoder_output.padding_mask

        projected_features, features_attention = self.Qformer(
            audio_encoder_output.audio_features, speech_attention_mask
        )
        padding_mask = ~features_attention.bool()
        return AudioModuleOutput(
            audio_features=projected_features,
            padding_mask=padding_mask,
        )

    def _get_trainable_parameters(self) -> TotalTrackingDict:
        with deepspeed.zero.GatheredParameters(
            self.parameters(),
            modifier_rank=None,
            enabled=is_deepspeed_zero3_enabled(),
        ):
            trainable_params = TotalTrackingDict()
            total_params = count_trainable_parameters(self.Qformer)
            # Bert
            trainable_params["Qformer_Bert"] = count_trainable_parameters(
                self.Qformer.Qformer
            )
            # Optional Speech Projection
            if hasattr(self.Qformer, "speech_proj"):
                trainable_params["speech_proj"] = count_trainable_parameters(
                    self.Qformer.speech_proj
                )
            # Queries Parameter has not .parameters()
            trainable_params["learnable_queries"] = (
                total_params - trainable_params.total
            )

            return trainable_params


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")


def build_speech_projector(config, delay_load=False, **kwargs):
    parameter_partitioning = (
        deepspeed.zero.Init(config_dict_or_path=deepspeed_config())
        if is_deepspeed_zero3_enabled()
        else contextlib.nullcontext()
    )
    with parameter_partitioning:  # Partition the projector's parameters
        projector_type = getattr(
            config, "mm_speech_projector_type", "dummy_hardcoded"
        )
        if projector_type == "qformer":
            speech_projector = QFormerProjector(config)           
        else:
            raise ValueError(
                f"Unknown speech projector type: {projector_type}"
            )

    print(
        f"Speech projector [{speech_projector.__class__.__qualname__}] - "
        f"trainable parameters:"
    )
    print(speech_projector.get_trainable_parameters(return_plain_dict=False))

    return speech_projector
