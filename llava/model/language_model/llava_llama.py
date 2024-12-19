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


from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)
from transformers.generation.utils import GenerateOutput

from ..contrastive.contrastive_forward import forward_contrastive
from ..contrastive.eval_contrastive import eval_contr
from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel
from ..model_outputs import CausalLMOutputWithPastAndGranularLosses


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, tokenizer=None):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()
        self.config = config
        self.tokenizer = tokenizer

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        audios_srs: Optional[list[(torch.FloatTensor, int)]] = None,
        transcription_ids: Optional[torch.LongTensor] = None,
        transcription_attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        word_tensors=None,
    ) -> Union[Tuple, CausalLMOutputWithPastAndGranularLosses]:

        is_speech_projector_output_present = False
        if inputs_embeds is None:
            out = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                audios_srs,
                transcription_ids=transcription_ids,
                transcription_attention_mask=transcription_attention_mask,
                contrastive=self.config.contrastive,
                word_tensors=word_tensors,
                contr_asr_combine_loss=self.config.contr_asr_combine_loss,
            )

            is_speech_projector_output_present = len(out) == 7

            if is_speech_projector_output_present:
                (
                    input_ids,  # None
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    raw_speech_projector_output,
                ) = out

            else:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                ) = out

        if self.config.contrastive and not torch.is_inference_mode_enabled():

            if not self.config.contr_asr_combine_loss:
                return forward_contrastive(
                    self.config,
                    raw_speech_projector_output,
                    word_tensors,
                    inputs_embeds,
                    attention_mask,
                    labels,
                    use_cache,
                    super().forward,
                )

            else:
                assert len(input_ids) == 3
                contr_loss = forward_contrastive(
                    self.config,
                    input_ids[2],
                    word_tensors,
                    input_ids[1],
                    input_ids[0],
                    labels,
                    use_cache,
                    super().forward,
                )

                input_ids = None

        model_output = super().forward(
            input_ids=input_ids,  # None
            attention_mask=attention_mask,
            position_ids=position_ids,  # None
            past_key_values=past_key_values,  # None
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,  # None
            output_hidden_states=output_hidden_states,  # None
            return_dict=return_dict,  # None
            cache_position=cache_position,  # None
        )

        if self.config.contr_asr_combine_loss:
            model_output.loss += contr_loss.loss

        if not is_speech_projector_output_present:
            return model_output

        return self.update_model_output_with_speech_projector_output(
            model_output,
            raw_speech_projector_output,
            audio_nwp=self.config.audio_nwp,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        audios: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        tokenizer=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:


        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            raise ValueError(
                "`generate()` currently only supports audio!"
            )

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        is_input_multimodal = audios is not None or images is not None

        if is_input_multimodal:
            (inputs, position_ids, attention_mask, _, inputs_embeds, *_) = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    past_key_values=None,
                    labels=None,
                    images=images,
                    image_sizes=image_sizes if images is not None else None,
                    audios_srs=audios,
                )
            )


        else:

            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            stop_strings="\n",
            tokenizer=tokenizer,
            **kwargs,
        )



    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        if "images" in kwargs:
            inputs["images"] = kwargs.pop("images")
        if "image_sizes" in kwargs:
            inputs["image_sizes"] = kwargs.pop("image_sizes")
        if "audios" in kwargs:
            inputs["audios"] = kwargs.pop("audios")
        return inputs

    @torch.no_grad()
    def eval_contrastive(
        self,
        input_ids: Optional[torch.Tensor] = None,
        audios: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        tokenizer=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        return eval_contr(
            self.prepare_inputs_labels_for_multimodal,
            input_ids,
            audios,
            images,
            image_sizes,
            **kwargs,
        )


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
