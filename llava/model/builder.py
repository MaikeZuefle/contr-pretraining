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


import os
import shutil
import warnings

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
)
from llava.model import *
from llava.train.train import smart_tokenizer_and_embedding_resize


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    modality="image",
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    inference=False,
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    if "llava" in model_name.lower():
        # Load LLaVA model
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )
        if "lora" in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig

            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_base, use_fast=False
            )

            print("Loading LLaVA from base model...")


            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                config=lora_cfg_pretrained,
                **kwargs,
            )
            if "Llama-3" in model_base:
                # if not "Llama-3.1" in model_base:
                #     print(f"Model is Llama-3")
                #     print(f"Adding pad token as '<pad>'")
                #     smart_tokenizer_and_embedding_resize(
                #         special_tokens_dict=dict(pad_token="<pad>"),
                #         tokenizer=tokenizer,
                #         model=model,
                #     )
                if not "Llama-3.1" in model_base:
                    raise ValueError(
                        "Llama-3 is not supported, please use Llama-3.1"
                    )
                else:
                    print(f"Model is Llama-3.1")
                    pad_token = "<|finetune_right_pad_id|>"
                    pad_token_id = 128004
                    print(f"Setting pad token as '{pad_token}'")
                    tokenizer.pad_token = pad_token
                    tokenizer.pad_token_id = pad_token_id
                    model.config.pad_token_id = pad_token_id

            token_num, tokem_dim = (
                model.lm_head.out_features,
                model.lm_head.in_features,
            )
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num,
                        tokem_dim,
                        device=model.device,
                        dtype=model.dtype,
                    )
                )
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num,
                        tokem_dim,
                        device=model.device,
                        dtype=model.dtype,
                    )
                )

            print("Loading additional LLaVA weights...")
            if os.path.exists(
                os.path.join(model_path, "non_lora_trainables.bin")
            ):
                non_lora_trainables = torch.load(
                    os.path.join(model_path, "non_lora_trainables.bin"),
                    map_location="cpu",
                )
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id, filename=filename, subfolder=subfolder
                    )
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(
                    model_path, "non_lora_trainables.bin"
                )
            non_lora_trainables = {
                (
                    k[11:]
                    if k.startswith("base_model.")
                    else (k[18:] if k.startswith("module.") else k)
                ): v
                for k, v in non_lora_trainables.items()
            }
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith("model.") else k): v
                    for k, v in non_lora_trainables.items()
                }
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel

            print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging LoRA weights...")
            model = model.merge_and_unload()
            print("Model is loaded...")

        elif model_base is not None:
            # this may be mm projector only
            print("Loading LLaVA from base model...")

            tokenizer = AutoTokenizer.from_pretrained(
                model_base, use_fast=False
            )
            cfg_pretrained = AutoConfig.from_pretrained(model_path)

            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                config=cfg_pretrained,
                **kwargs,
            )
            if "Llama-3" in model_base:

                if not "Llama-3.1" in model_base:
                    print(f"Model is Llama-3")
                    print(f"Adding pad token as '<pad>'")
                    smart_tokenizer_and_embedding_resize(
                        special_tokens_dict=dict(pad_token="<pad>"),
                        tokenizer=tokenizer,
                        model=model,
                    )
                    if inference and "Llama-3" in model_base:
                        cfg_pretrained.pad_token_id = None
                        cfg_pretrained.vocab_size = (
                            cfg_pretrained.vocab_size - 1
                        )
                else:
    
                    print(f"Model is Llama-3.1")
                    pad_token = "<|finetune_right_pad_id|>"
                    pad_token_id = 128004
                    print(f"Setting pad token as '{pad_token}'")
                    tokenizer.pad_token = pad_token
                    tokenizer.pad_token_id = pad_token_id
                    model.config.pad_token_id = pad_token_id

            mm_projector_weights = torch.load(
                os.path.join(
                    model_path,
                    (
                        "mm_projector.bin"
                        if modality == "image"
                        else "mm_speech_projector.bin"
                    ),
                ),
                map_location="cpu",
            )
            mm_projector_weights = {
                k: v.to(torch.float16) for k, v in mm_projector_weights.items()
            }
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False
            )
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )
    else:
        raise NotImplementedError()


    processor = None

    if "llava" in model_name.lower():
        if modality == "image":
            raise NotImplementedError("Model only supports audio at the moment!")
           
        else:
            audio_encoder = model.get_audio_encoder()

            if not audio_encoder.is_loaded:
                audio_encoder.load_model()
            if device_map != "auto":
                audio_encoder.to(device=device_map, dtype=torch.float16)
            processor = audio_encoder.processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, processor, context_len
