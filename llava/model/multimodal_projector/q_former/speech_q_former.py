"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import typing

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from transformers import BertTokenizer

from .Qformer import BertConfig, BertLMHeadModel


class SpeechQformer(nn.Module):
    """
    Implementation of the Q-former model, make it working with an audio encoder.
    """

    @classmethod
    def init_Qformer(
        cls,
        feature_d: int,
        num_query_token: int,
        num_hidden_layers: int = 8,
        cross_attention_freq: int = 1,
        hidden_size: typing.Optional[int] = None,
        num_attention_heads: typing.Optional[int] = None,
    ):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = (
            num_hidden_layers  # added in Salmonn
        )
        encoder_config.encoder_width = feature_d
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        # NOTE cross_attention_freq = 1 in Salmonn, cross_attention_freq = 2 in blip2
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        if hidden_size is not None:
            encoder_config.hidden_size = hidden_size
        if num_attention_heads is not None:
            encoder_config.num_attention_heads = num_attention_heads

        Qformer = BertLMHeadModel(config=encoder_config)  # Salmonn
        # Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config) # Blip2

        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(
            mean=0.0, std=encoder_config.initializer_range
        )

        return Qformer, query_tokens

    def __init__(
        self,
        embed_dim: int,
        audio_emb_dim: int,
        num_query_token: int,
        num_hidden_layers: int,
        audio_encoder_fs: float,
        interpolation: bool = False,
        freeze_QFormer: bool = False,
        cross_attention_freq: int = 1,
        second_per_window: float = 0.3333333333333333,
        num_attention_heads: typing.Optional[int] = None,
        interpolation_config: typing.Optional[str] = None,
    ):
        super().__init__()
        # no need to init tokenizer here, it is passed as an argument
        # self.tokenizer = self.init_tokenizer()
        # no need to init audio encoder here, it is passed as an argument
        # self.audio_encoder = self.init_audio_encoder()
        hidden_size = None
        if interpolation:
            assert interpolation_config is not None
            assert interpolation_config in [
                "before",
                "after",
            ], "Qformer interpolation_config must be 'before' or 'after'"
            if interpolation_config == "before":
                hidden_size = embed_dim

        self.Qformer, self.query_tokens = self.init_Qformer(
            feature_d=audio_emb_dim,
            num_query_token=num_query_token,
            num_hidden_layers=num_hidden_layers,
            cross_attention_freq=cross_attention_freq,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        self.second_stride = self.second_per_window = second_per_window
        self.audio_encoder_fs = audio_encoder_fs
        self.embed_dim = embed_dim  # LLM embedding dimension
        self.audio_emb_dim = audio_emb_dim
        self.interpolation = interpolation
        self.interpolation_config = interpolation_config

        # from Salmonn
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.Qformer.cls = None

        if freeze_QFormer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer.eval()
            self.query_tokens.requires_grad = False
            logging.info("freeze Speech QFormer")

        # FIXME pass those dimensions as arguments
        # if self.add_linear_projection:
        #     self.input_projection = nn.Sequential(
        #         nn.Linear(128, 1024),
        #     )

        self.ln_speech = nn.LayerNorm(audio_emb_dim)

        # NOTE these below are used in the original BLIP2 Qformer implementation
        # self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        # self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        # self.temp = nn.Parameter(0.07 * torch.ones([]))

        # self.max_txt_len = max_txt_len
        self.num_query_token = num_query_token

        if not self.interpolation:
            self.speech_proj = nn.Linear(
                self.Qformer.config.hidden_size, self.embed_dim
            )

    def speech_butcher(self, embs, mask, truncate_last=False):
        """
        split the embeddings into windows.
        """
        # window level QFormer
        B, T, C = embs.shape
        kernel = round(
            self.audio_encoder_fs * self.second_per_window
        )  # assuming 50Hz of frame rate
        stride = round(
            self.audio_encoder_fs * self.second_stride
        )  # assuming 50Hz of frame rate
        # pad the T dimension before of the unfold
        if not truncate_last and (T % kernel):
            pad_length = kernel - T % kernel
            padding_tensor = torch.zeros(
                (B, pad_length, C), dtype=embs.dtype, device=embs.device
            )
            # Concatenate the original tensor with the padding tensor along the T dimension
            embs = torch.cat([embs, padding_tensor], dim=1)
            mask = F.pad(mask, (0, pad_length), "constant")

        kernel = (1, kernel)
        stride = (1, stride)
        embs_tr = embs.transpose(1, 2).unsqueeze(2)
        embs_overlap = F.unfold(
            embs_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride
        )
        _, _, L = embs_overlap.shape
        embs_overlap = embs_overlap.view(B, -1, kernel[1], L)
        embs_overlap = torch.permute(embs_overlap, [0, 3, 2, 1])
        embs = embs_overlap.reshape(-1, kernel[1], C)

        if truncate_last:
            print(T - T % kernel[1])
            mask = mask[:, : T - T % kernel[1]].contiguous()
            print(f"mask shape: {mask.shape}")

        return embs, mask.view(embs.shape[:-1])


            

    def forward(self, speech_embeds, speech_atts):
        """
        Encode auditory features.
        """
        speech_embeds = self.ln_speech(speech_embeds)

        # window level QFormer
        B = speech_embeds.size(0)
        speech_embeds, speech_atts = self.speech_butcher(
            speech_embeds, speech_atts
        )

        # query tokens
        query_tokens = self.query_tokens.expand(speech_embeds.shape[0], -1, -1)
        attention_mask = (
            (speech_atts.sum(dim=1) > 0)
            .view(-1, 1)
            .repeat(1, self.num_query_token)
            .to(torch.long)
            .to(query_tokens.device)
        )
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            # attention_mask=attention_mask,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        if self.interpolation:
            if self.interpolation_config == "after":
                speech_embeds = F.interpolate(
                    query_output.last_hidden_state,  # 768
                    size=self.embed_dim,  # 4096
                    mode="linear",
                    align_corners=False,
                )
            else:
                speech_embeds = query_output.last_hidden_state
        else:
            speech_embeds = self.speech_proj(query_output.last_hidden_state)

        speech_embeds = speech_embeds.view(
            B, -1, speech_embeds.size(2)
        ).contiguous()
        speech_atts = attention_mask.view(B, -1).contiguous()

        return speech_embeds, speech_atts
