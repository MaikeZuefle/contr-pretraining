#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import json
import logging
import os
from abc import ABC, abstractmethod
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from llava.constants import (
    AUDIO_TOKEN_INDEX,
    IGNORE_INDEX,
)

from .contrastive.prepare_contrastive import (
    prepare_contrastive_inference,
    prepare_contrastive_input,
)
from .contrastive.prepare_mixed_nwp import prepare_mixed_nwp
from .model_outputs import CausalLMOutputWithPastAndGranularLosses
from .multimodal_encoder.builder import build_audio_encoder
from .multimodal_projector.builder import (
    build_speech_projector,
    build_vision_projector,
)
from .multimodal_projector.outputs import (
    AudioModuleOutput,
    CifProjectorOutput,
    CtcProjectorOutput,
)


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            raise NotImplementedError("Currently, only audio is supported!")

        if hasattr(config, "mm_audio_encoder"):
            self.audio_encoder = build_audio_encoder(config, delay_load=True)
            self.mm_speech_projector = build_speech_projector(config)

    def get_audio_encoder(self):
        return getattr(self, "audio_encoder", None)

    

    def initialize_speech_modules(
        self,
        model_args,
        fsdp=None,
        llm_embeddings_dim=4096,
        llm_vocab_size=32000,
        llm_pad_token_id=0,
        attn_implementation=None,
        torch_dtype=None,
    ):
        audio_encoder = model_args.audio_encoder
        self.config.mm_audio_encoder = audio_encoder
        self.config.mm_speech_projector_type = (
            model_args.mm_speech_projector_type
        )
        self.config.llm_embeddings_dim = llm_embeddings_dim

        pretrain_mm_speech_adapter = model_args.pretrain_mm_speech_adapter
        for attr in dir(model_args):
            if any(
                attr.startswith(prefix)
                for prefix in [
                    "mlp_",
                    "conv_",
                    "cif_",
                    "ctc_",
                    "cformer_",
                    "cmlp_",
                    "bert_",
                    "qformer_",
                ]
            ):
                setattr(self.config, attr, getattr(model_args, attr))

        # NOTE added for retro compatibility
        self.config.qformer_num_query_token = getattr(
            model_args,
            "mm_qformer_num_query_token",
            getattr(model_args, "qformer_num_query_tokens", None),
        )
        self.config.qformer_num_hidden_layers = getattr(
            model_args,
            "mm_qformer_num_hidden_layers",
            getattr(model_args, "qformer_num_hidden_layers", None),
        )

        if self.get_audio_encoder() is None:
            audio_encoder = build_audio_encoder(
                model_args,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype,
            )

            if fsdp is not None and len(fsdp) > 0:
                self.audio_encoder = [audio_encoder]
            else:
                self.audio_encoder = audio_encoder
        else:
            if fsdp is not None and len(fsdp) > 0:
                audio_encoder = self.audio_encoder[0]
            else:
                audio_encoder = self.audio_encoder
            audio_encoder.load_model()

        # Overwrite the value of deprecated arguments
        self.config.audio_embeddings_dim = audio_encoder.hidden_size
        self.config.cif_conv_out_channels = audio_encoder.hidden_size
        self.config.cif_conv_kernel_size = 3
        self.config.cif_conv_stride = 1
        self.config.cif_conv_padding = 1
        self.config.cif_ctc_loss_vocab_size = llm_vocab_size
        self.config.cif_ctc_loss_blank_id = llm_pad_token_id

        if pretrain_mm_speech_adapter is not None:
            config_file = os.path.join(
                os.path.dirname(pretrain_mm_speech_adapter), "config.json"
            )

            pretrain_mm_speech_adapter_config = SimpleNamespace(
                **json.load(open(config_file))
            )
            # checking if the attributes in the
            # pretrain_mm_speech_adapter config file are different from
            # the attributes in the model config file. If they are
            # different, the value of the attribute in the model config
            # file is overwritten by the value of the attribute in the
            # pretrain_mm_speech_adapter config file
            for attr in dir(pretrain_mm_speech_adapter_config):
                if any(
                    attr.startswith(prefix)
                    for prefix in [
                        "mlp_",
                        "conv_",
                        "cif_",
                        "ctc_",
                        "cformer_",
                        "bert_",
                        "qformer_",
                    ]
                ):
                    if getattr(self.config, attr) != getattr(
                        pretrain_mm_speech_adapter_config, attr
                    ):
                        logging.warning(
                            f"Warning: The attribute {attr} in the "
                            f"pretrain_mm_speech_adapter config file is "
                            f"different from the attribute in the model "
                            f"config file. Overwriting the value of the "
                            f"attribute in the model config file."
                        )
                        setattr(
                            self.config,
                            attr,
                            getattr(pretrain_mm_speech_adapter_config, attr),
                        )

            self.mm_speech_projector = build_speech_projector(
                pretrain_mm_speech_adapter_config
            )
            mm_projector_weights = torch.load(
                pretrain_mm_speech_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            import deepspeed
            from transformers.integrations.deepspeed import (
                is_deepspeed_zero3_enabled,
            )

            with deepspeed.zero.GatheredParameters(
                self.mm_speech_projector.parameters(),
                modifier_rank=0,
                enabled=is_deepspeed_zero3_enabled(),
            ):
                self.mm_speech_projector.load_state_dict(
                    get_w(mm_projector_weights, "mm_speech_projector"),
                    strict=False,
                )

        else:
            # TODO: add support for LoRA.
            # TODO: in case we support Flash Attention 2 in the speech
            # projectors too (right now we only have BERT, but it does
            # not support it), pass `attn_implementation` to
            # build_speech_projector
            self.mm_speech_projector = build_speech_projector(self.config)



class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass


    def get_audio_encoder(self):
        return self.get_model().get_audio_encoder()

    def encode_audios(self, audios, speech_projector_kwargs=None):
        speech_projector_kwargs = speech_projector_kwargs or {}

        try:
            import torch

            if torch.is_inference_mode_enabled() and len(audios) == 1:
                waveform, sr = audios[0]
                chunk_duration_seconds = 100
                audio_length_in_seconds = waveform.shape[1] / sr
                if audio_length_in_seconds > 100:

                    chunk_size = (
                        chunk_duration_seconds * sr
                    )  # Number of samples per chunk

                    # Calculate the number of chunks
                    num_chunks = waveform.size(1) // chunk_size

                    # Split the resampled waveform into chunks
                    chunks = [
                        (
                            waveform[:, i * chunk_size : (i + 1) * chunk_size],
                            sr,
                        )
                        for i in range(num_chunks)
                    ]

                    # If the audio length isn't a perfect multiple of 100 seconds, the last chunk may be smaller
                    if waveform.size(1) % chunk_size != 0:
                        chunks.append(
                            (
                                waveform[:, num_chunks * chunk_size :],
                                sr,
                            )
                        )

                    audio_encoder_chunk_outputs = [
                        (
                            self.get_model()
                            .get_audio_encoder()
                            .to(audios[0][0])([c])
                        )
                        for c in chunks
                    ]

                    concatenated_audio_features = torch.cat(
                        [c.audio_features for c in audio_encoder_chunk_outputs]
                        * 4,
                        dim=1,
                    )

                    # Concatenate the padding_mask along the second dimension (axis=1)
                    concatenated_padding_mask = torch.cat(
                        [c.padding_mask for c in audio_encoder_chunk_outputs]
                        * 4,
                        dim=1,
                    )

                    class AudioModuleOutput:
                        def __init__(self, audio_features, padding_mask):
                            self.audio_features = audio_features
                            self.padding_mask = padding_mask

                    concatenated_audio_encoder_output = AudioModuleOutput(
                        concatenated_audio_features, concatenated_padding_mask
                    )

                    return self.get_model().mm_speech_projector(
                        concatenated_audio_encoder_output,
                        **speech_projector_kwargs,
                    )
        except:
            pass

        audio_encoder_output = (
            self.get_model().get_audio_encoder().to(audios[0][0])(audios)
        )

        return self.get_model().mm_speech_projector(
            audio_encoder_output, **speech_projector_kwargs
        )

    def prepare_speech_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        audios_srs=None,
        transcription_ids=None,
        transcription_attention_mask=None,
        contrastive=False,
        contrastive_inference=False,
        word_tensors=None,
        contr_asr_combine_loss=False,
    ):
        BOS_ID = 128000
        if word_tensors is not None:
            # obtain mixed audio features (text speech mix)
            return prepare_mixed_nwp(
                BOS_ID,
                IGNORE_INDEX,
                labels,
                word_tensors,
                transcription_ids,
                transcription_attention_mask,
                self.get_model(),
                self.encode_audios,
                contrastive,
            )

        else:

            raw_speech_projector_output = self.encode_audios(
                audios_srs,
                speech_projector_kwargs={
                    "transcription_ids": transcription_ids,
                    "transcription_attention_mask": transcription_attention_mask,
                },
                # ↑ required by audio projectors that need to access the
                # textual transcription as well (such as CIF and CTC which
                # need to compute the ctc_loss on the transcription)
            )
        if isinstance(raw_speech_projector_output, dict):
            audio_features = raw_speech_projector_output["audio_features"]
        else:
            audio_features = (
                raw_speech_projector_output.audio_features
            )  # (batch_size, seq_len, hidden_size)

        # create default tensors for attention_mask, position_ids and
        # labels if they are not provided
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0,
                input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            )

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        if contrastive_inference:
            return prepare_contrastive_inference(
                self.get_model(),
                attention_mask,
                input_ids,
                raw_speech_projector_output,
            )

        if contrastive and (
            not torch.is_inference_mode_enabled()
        ):  # don't need to concatenate prompt and audios

            if not contr_asr_combine_loss:
                return prepare_contrastive_input(
                    IGNORE_INDEX,
                    labels,
                    raw_speech_projector_output,
                    self.get_model(),
                )
            else:

                (
                    _,
                    _,
                    padded_label_mask,
                    _,
                    padded_embed_labels,
                    _,
                    raw_audio_adapter_output,
                ) = prepare_contrastive_input(
                    IGNORE_INDEX,
                    labels,
                    raw_speech_projector_output,
                    self.get_model(),
                )

        # remove padding using attention_mask, obtaining list of tensors
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(
                input_ids, attention_mask
            )
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []  # text + audio embeddings (?)
        new_labels = []  # labels for text + audio (?)
        cur_audio_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_audios = (
                cur_input_ids == AUDIO_TOKEN_INDEX
            ).sum()  # TODO: extend to multiple audios. This should always be 1 in our case

            ################### this is not the case for us #####################
            if num_audios == 0:
                cur_audio_features = audio_features[cur_audio_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(
                    cur_input_ids
                )
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_audio_features[0:0]],
                    dim=0,
                    # ↑ FIXME: cur_audio_features[0:0] is an empty tensor!
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_audio_idx += 1
                continue
            ################### this is not the case for us #####################

            audio_token_indices = (
                [-1]  # just a sentinel
                + torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]  # (seq_len,)
            )

            cur_input_ids_noaud = []
            cur_labels = labels[batch_idx]
            cur_labels_noaud = []
            # extract input_ids and labels for non-audio token positions
            for i in range(len(audio_token_indices) - 1):
                # split input ids to before audio (model prompt) and after audio (task prompt + transcription) to later insert audio embeddings
                cur_input_ids_noaud.append(
                    cur_input_ids[
                        audio_token_indices[i] + 1 : audio_token_indices[i + 1]
                    ]
                )
                cur_labels_noaud.append(
                    cur_labels[
                        audio_token_indices[i] + 1 : audio_token_indices[i + 1]
                    ]
                )

            # from transformers import AutoTokenizer, AutoModelForCausalLM

            # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

            # embed inputs (the split version) and concatenate them
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_noaud)
            )
            # ↑ (seq_len - num_audios, hidden_size)
            split_sizes = [x.shape[0] for x in cur_labels_noaud]

            # split input embeddings again
            cur_input_embeds_no_aud = torch.split(
                cur_input_embeds, split_sizes, dim=0
            )

            # current input embeddings are now system prompt + normal prompt without audio token embedded
            cur_new_input_embeds = []  # text + audio embeddings
            cur_new_labels = []  # labels for text + audio

            for i in range(
                num_audios + 1
            ):  # always 1 since we have just 1 audio

                # now we add system prompt and the audio prompt and in the next iteration we add the task prompt
                cur_new_input_embeds.append(cur_input_embeds_no_aud[i])
                cur_new_labels.append(cur_labels_noaud[i])

                # when i >= num_audios we added the task prompt (so now it is system prompt, audio, task prompt) and no audio left
                if i < num_audios:
                    cur_audio_features = audio_features[cur_audio_idx]
                    # ↑ (audio_seq_len, hidden_size)
                    if isinstance(
                        raw_speech_projector_output, AudioModuleOutput
                    ):
                        # remove padding from audio features
                        # NOTE: prior to introducing speech projectors
                        # other than DummyProjector, this was not
                        # necessary as audio_features was NOT a tensor
                        # of batched and padded audio features, but
                        # rather a list of unpadded tensors
                        audio_features_attention_mask = (
                            ~raw_speech_projector_output.padding_mask[
                                cur_audio_idx
                            ]
                        )
                        cur_audio_features = cur_audio_features[
                            audio_features_attention_mask
                        ]

                    cur_audio_idx += 1

                    # add audio features to list of input (text) features
                    cur_new_input_embeds.append(cur_audio_features)

                    # extend labels with ignore index for audio features
                    cur_new_labels.append(
                        torch.full(
                            (cur_audio_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = [
                x.to(self.device) for x in cur_new_input_embeds
            ]

            # input emebds are now system prompt, audios, task prompt, labels
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            # labels are now 0s (system promt), -100 (audios), 0s (task prompt), labels ids
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as audio embeddings can make
        # the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Stack them back as a single tensor, padding if necessary
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len),
            dtype=position_ids.dtype,
            device=position_ids.device,
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]

            if torch.is_inference_mode_enabled():
                self.config.tokenizer_padding_side = "left"

            if (
                getattr(self.config, "tokenizer_padding_side", "right")
                == "left"
            ):
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )
            else:  # tokenizer_padding_side == "right"
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if contr_asr_combine_loss:
            return (
                [
                    padded_label_mask,
                    padded_embed_labels,
                    raw_audio_adapter_output,
                ],
                position_ids,
                attention_mask,
                past_key_values,
                new_input_embeds,
                new_labels,
                raw_speech_projector_output,
            )

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
            raw_speech_projector_output,
        )

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
        audios_srs=None,
        transcription_ids=None,
        transcription_attention_mask=None,
        contrastive=False,
        contrastive_inference=False,
        word_tensors=None,
        contr_asr_combine_loss=False,
    ):
        if audios_srs is not None:

            return self.prepare_speech_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                audios_srs,
                transcription_ids=transcription_ids,
                transcription_attention_mask=transcription_attention_mask,
                contrastive=contrastive,
                contrastive_inference=contrastive_inference,
                word_tensors=word_tensors,
                contr_asr_combine_loss=contr_asr_combine_loss,
            )
        

        if images is None or input_ids.shape[1] == 1:

            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )
        else:
            raise NotImplementedError
      
    def update_model_output_with_speech_projector_output(
        self,
        model_output,
        speech_projector_output,
        audio_nwp=False,
    ):
        model_output_dict = vars(model_output)
        lm_loss = model_output_dict.pop("loss")
        granular_losses = {"lm_loss": lm_loss}
        if isinstance(
            speech_projector_output,
            (CifProjectorOutput, CtcProjectorOutput),
        ):
            if speech_projector_output.ctc_loss is not None:
                granular_losses["ctc_loss"] = (
                    speech_projector_output.ctc_loss
                    * self.config.cif_ctc_loss_weight
                    # ↑ TODO: rename to `ctc_loss_weight`
                )
            if (
                isinstance(speech_projector_output, CifProjectorOutput)
                and speech_projector_output.quantity_loss is not None
            ):
                granular_losses["quantity_loss"] = (
                    speech_projector_output.quantity_loss
                    * self.config.cif_quantity_loss_weight
                )

        return CausalLMOutputWithPastAndGranularLosses(
            loss=sum(granular_losses.values()),
            granular_losses=granular_losses,
            **model_output_dict,
        )
