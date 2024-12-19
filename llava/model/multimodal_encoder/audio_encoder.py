import contextlib
import logging
from typing import Optional

import numpy as np
import torch
import transformers

from llava.model.multimodal_projector.outputs import AudioModuleOutput
from llava.model.multimodal_projector.utils import (
    attention_weights_to_attention_mask,
    lengths_to_attention_mask,
)
from llava.model.utils import compute_output_length_from_conv1d_layer


class HfAudioEncoder(torch.nn.Module):
    processor_class = transformers.AutoProcessor
    processor_audio_arg_name = None
    model_class = None
    model_forward_kwargs = {}

    def __init__(
        self,
        model_name_or_path,
        delay_load=False,
        attn_implementation=None,
        torch_dtype=None,
    ):
        super().__init__()
        if self.model_class is None or not issubclass(
            self.model_class, transformers.PreTrainedModel
        ):
            raise ValueError(
                f"Class attribute `model_class` must be a subclass of "
                f"`transformers.PreTrainedModel` (found {self.model_class})."
            )

        self.model_name_or_path = model_name_or_path
        self.model_config = transformers.AutoConfig.from_pretrained(
            self.model_name_or_path
        )
        self.processor = self.processor_class.from_pretrained(
            self.model_name_or_path
        )

        self.attn_implementation = attn_implementation
        self._check_support_for_attn_implementation()

        self.torch_dtype = torch_dtype
        self.is_loaded = False
        if not delay_load:
            self.load_model()

    def _check_support_for_attn_implementation(self):
        default_attn_implementation = self.config._attn_implementation
        if self.attn_implementation is None:
            self.attn_implementation = default_attn_implementation
            return

        supported_attn_implementations = ["eager", "sdpa", "flash_attention_2"]
        if self.attn_implementation not in supported_attn_implementations:
            raise ValueError(
                f"Attention implementation {self.attn_implementation} "
                f"is not supported. Supported implementations are "
                f"{supported_attn_implementations}."
            )
        if (
            self.attn_implementation == "sdpa"
            and not self.model_class._supports_sdpa
        ):
            logging.warning(
                f"{self.model_class.__qualname__} does not support "
                f"attn_implementation='sdpa'. Falling back to "
                f"'{default_attn_implementation}' (config default)."
            )
            self.attn_implementation = default_attn_implementation
        elif (
            self.attn_implementation == "flash_attention_2"
            and not self.model_class._supports_flash_attn_2
        ):
            logging.warning(
                f"{self.model_class.__qualname__} does not support "
                f"attn_implementation='flash_attention_2'. Falling back to "
                f"'{default_attn_implementation}' (config default)."
            )
            self.attn_implementation = default_attn_implementation

    def load_model(self, device_map=None):
        self.model = self.model_class.from_pretrained(
            self.model_name_or_path,
            config=self.model_config,
            device_map=device_map,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.torch_dtype,
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.is_loaded = True

    @torch.no_grad()
    def forward(
        self, audios_srs: list[(torch.FloatTensor, int)]
    ) -> AudioModuleOutput:
        audios, sampling_rates = zip(*audios_srs)
        unique_sampling_rates = set(sampling_rates)
        if len(unique_sampling_rates) > 1:
            raise ValueError(
                "All audios must have the same sampling rate. "
                f"Found {len(unique_sampling_rates)} unique sampling rates: "
                f"{unique_sampling_rates}."
            )

        audios = [a.squeeze().float().cpu().numpy() for a in audios]

        input_values, attention_mask = self._process_audios(
            audios, sr=unique_sampling_rates.pop()
        )

        return self._encode_processed_audios(input_values, attention_mask)

    def _process_audios(self, audios: list[np.ndarray], sr: int, **kwargs):
        if sr != self.sampling_rate:
            raise ValueError(
                f"Sampling rate {sr} is not supported by this model. "
                f"Expected {self.sampling_rate}."
            )
        processor_args = []
        processor_kwargs = {
            "sampling_rate": sr,
            "return_tensors": "pt",
        }
        if self.processor_audio_arg_name is None:
            processor_args.append(audios)
        else:
            processor_kwargs[self.processor_audio_arg_name] = audios

        processor_output = self.processor(
            *processor_args, **processor_kwargs, **kwargs
        ).to(device=self.device, dtype=self.dtype)

        input_values_or_features = getattr(
            processor_output,
            "input_values",
            getattr(processor_output, "input_features", None),
        )
        if input_values_or_features is None:
            raise ValueError(
                "Expected `input_values` or `input_features` in the "
                "output of the processor."
            )

        attention_mask = getattr(
            processor_output,
            "attention_mask",
            getattr(processor_output, "padding_mask", None),
            # ↑ EnCodec returns `padding_mask` instead of
            # `attention_mask`. However, the `padding_mask` is
            # actually an attention mask 😅 Because of this, we
            # don't need to invert the mask
        )
        # NOTE: some processors do NOT return `attention_mask`
        # because the corresponding model was trained without it.
        # Instead, the model expects the input to be padded with 0s.

        return input_values_or_features, attention_mask.bool()

    def _encode_processed_audios(
        self, input_values, attention_mask: Optional[torch.BoolTensor] = None
    ) -> AudioModuleOutput:

        # if model is a hubert model, we need to adjust the mask length for super short sequences (in audio nwp)

        shortest_seq_length = input_values.shape[1] // 320 -1  # 320 is reduction factor

        if ("Hubert" in str(self.model_class)) and  (shortest_seq_length <= self.model.config.mask_time_length):
            orig_mask_time_length = self.model.config.mask_time_length
            new_mask_time_length =  max(1, shortest_seq_length // 2)  
            self.model.config.mask_time_length = new_mask_time_length

            model_output = self.model(
                input_values,
                attention_mask=attention_mask,
                output_attentions=self.supports_output_attentions(),
                return_dict=True,
                **self.model_forward_kwargs,
            )

            self.model.config.mask_time_length = orig_mask_time_length
        
        else:
            model_output = self.model(
                input_values,
                attention_mask=attention_mask,
                output_attentions=self.supports_output_attentions(),
                return_dict=True,
                **self.model_forward_kwargs,
                )

        if self.supports_output_attentions():
            attention_mask = attention_weights_to_attention_mask(
                model_output.attentions
            ).to(device=input_values.device)
        else:
            attention_mask = self._manually_recompute_attention_mask(
                attention_mask
            )


        return AudioModuleOutput(
            audio_features=model_output.last_hidden_state,
            padding_mask=~attention_mask,
        )

    def supports_output_attentions(self) -> bool:
        return self.attn_implementation not in ["sdpa", "flash_attention_2"]

    def _manually_recompute_attention_mask(
        self, attention_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        if self.supports_output_attentions():
            return None
        raise NotImplementedError(
            f"Method `_manually_recompute_attention_mask` must be implemented "
            f"in {self.__class__.__qualname__} as "
            f"attn_implementation={self.attn_implementation} does not "
            f"support output_attentions=True."
        )

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    @property
    def config(self):
        return self.model_config

    @property
    def sampling_rate(self):
        with contextlib.suppress(AttributeError):
            return self.processor.sampling_rate
        with contextlib.suppress(AttributeError):
            return self.processor.feature_extractor.sampling_rate
        raise AttributeError(
            f"{self.__class__.__name__} has no attribute `sampling_rate`."
        )

    @property
    def hidden_size(self):
        return self.config.hidden_size


class HubertEncoder(HfAudioEncoder):
    processor_class = transformers.AutoFeatureExtractor
    processor_audio_arg_name = "raw_speech"
    model_class = transformers.HubertModel

    def _process_audios(self, audios: list[np.ndarray], sr: int, **kwargs):
        if (
            self.config.feat_extract_norm == "layer"
        ) != self.processor.return_attention_mask:
            raise ValueError(
                f"model.config.feat_extract_norm="
                f"{self.config.feat_extract_norm} and "
                f"processor.return_attention_mask="
                f"{self.processor.return_attention_mask} are not "
                f'consistent. feat_extract_norm="layer" should imply '
                f"return_attention_mask=True."
            )
            # NOTE: see comments below
        return super()._process_audios(
            audios,
            sr,
            padding=True,
            return_attention_mask=self.config.feat_extract_norm == "layer",
            # ↑ NOTE: from the Hugging Face documentation on Wav2Vec2
            # (HuBERT uses a Wav2Vec2 feature extractor):
            # Wav2Vec2 models that have set
            # `config.feat_extract_norm == "group"`, such as
            # wav2vec2-base, have **not** been trained using
            # `attention_mask`. For such models, `input_values` should
            # simply be padded with 0 and no `attention_mask`should be
            # passed.
            #
            # For Wav2Vec2 models that have set
            # `config.feat_extract_norm == "layer"`, such as
            # wav2vec2-lv60, `attention_mask` should be passed for
            # batched inference.
            # NOTE: also, from the Hugging Face documentation on HuBERT:
            # attention_mask should only be passed if the
            # corresponding processor has
            # config.return_attention_mask == True. For all models
            # whose processor has
            # config.return_attention_mask == False, such as
            # hubert-base, attention_mask should not be passed to
            # avoid degraded performance when doing batched
            # inference. For such models input_values should simply
            # be padded with 0 and passed without attention_mask.
            # Be aware that these models also yield slightly
            # different results depending on whether input_values
            # is padded or not.
            **kwargs,
        )

    def _manually_recompute_attention_mask(
        self, attention_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        if self.supports_output_attentions():
            return None

        pre_conv_attention_mask = attention_mask
        for conv_layer in self.model.feature_extractor.conv_layers:
            SEQ_LEN_DIM = 1
            pre_conv_lengths = pre_conv_attention_mask.sum(SEQ_LEN_DIM)
            post_conv_lengths = compute_output_length_from_conv1d_layer(
                pre_conv_lengths, conv1d_layer=conv_layer.conv
            )
            post_conv_attention_mask = lengths_to_attention_mask(
                post_conv_lengths.long()
            )
            # prepare for the next iteration
            pre_conv_attention_mask = post_conv_attention_mask

        return post_conv_attention_mask
