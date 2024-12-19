# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import logging
import os
import random
from typing import Dict

import torch
import transformers
from torch.utils.data import Subset

from llava import conversation as conversation_lib
from llava.arguments import DataArguments, ModelArguments, TrainingArguments
from llava.dataset.audio_dataset import AudioDataset
from llava.dataset.datasets_wrapper import DatasetsWrapper
from llava.dataset.utils import DataCollatorForSupervisedDataset
from llava.eval.metrics import compute_asr_metrics, compute_st_metrics
from llava.model import *
from llava.model.multimodal_projector.builder import SpeechProjector
from llava.train.llava_trainer import LLaVATrainer
from llava.wandb_utils import get_tags_from_arguments

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {
            k: t for k, t in named_params if "lora_" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {
        k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()
    }
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu()
        for k, v in to_return.items()
    }
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu()
        for k, v in to_return.items()
    }
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = [
        "mm_speech_projector",
        "mm_audio_encoder",
        "mm_projector",
        "vision_tower",
        "vision_resampler",
    ]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, modality: str
):
    """Collects the state dict and dump to disk."""

    projector_name = (
        "mm_projector" if modality == "image" else "mm_speech_projector"
    )
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = [projector_name]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(
                    parent_folder, projector_name
                )
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weight_to_save,
                    os.path.join(output_dir, f"{projector_name}.bin"),
                )
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu() for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    version=None,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    contrastive=False,
    audio_nwp=False,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    all_datasets = DatasetsWrapper(data_args, audio_nwp=audio_nwp)

    train_dataset = AudioDataset(
        dataset=all_datasets.train_dataset,
        tokenizer=tokenizer,
        data_args=data_args,
        audio_nwp=audio_nwp,
    )
    # ##############################################################
    # # for debugging
    # start_index = 2850
    # print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOHHH Different start index, hope you are debugging!!!!")
    # # # Use all indices from start_index to the end of the dataset
    # train_dataset = Subset(train_dataset, list(range(start_index, len(train_dataset))))

    # #############################################################
    if data_args.data_subset:
        random.seed(42)
        subset_size = int(len(train_dataset) * data_args.data_subset)
        random_indices = random.sample(range(len(train_dataset)), subset_size)
        train_dataset = Subset(train_dataset, random_indices)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset = (
        {
            task_name: AudioDataset(
                dataset=eval_d, tokenizer=tokenizer, data_args=data_args
            )
            for task_name, eval_d in (all_datasets.eval_dataset or {}).items()
        }
        if all_datasets.eval_dataset is not None
        else None
    )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def train(attn_implementation=None):
    seed = 42
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}

    
    if model_args.audio_encoder is not None:
        
        model_config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True
        )
        model_type = model_config.model_type
      
        if model_type == "llama":
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=model_config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args,
            )
        else:
            raise NotImplementedError(
                f"Model type {model_type} is not supported in LLaVA."
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
        )
    model.config.use_cache = False


    if model_args.freeze_backbone:
        model.model.requires_grad_(False)


    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.tokenizer_padding_side,
        use_fast=(model.config.model_type == "mpt"),
    )

    if model_args.version == "llama_3_1":
        pad_token = "<|finetune_right_pad_id|>"
        print(f"Setting pad token to '{pad_token}'")
        tokenizer.pad_token = pad_token
        tokenizer.pad_token_id = 128004
        conversation_lib.default_conversation = (
            conversation_lib.conv_templates["llama_3_1"]
        )
    else:
        raise NotImplementedError("Currently, only Llama31-instruct models are supported!")
       

    if model_args.vision_tower is not None:
        raise NotImplementedError("Currently, only audio is supported as a modality!")

    if model_args.audio_encoder is not None:
        model.get_model().initialize_speech_modules(
            model_args=model_args,
            fsdp=training_args.fsdp,
            llm_embeddings_dim=model.config.hidden_size,
            llm_vocab_size=model.config.vocab_size,
            llm_pad_token_id=(
                getattr(tokenizer, "pad_token_id", None)
                or getattr(tokenizer, "eos_token_id", None)
                or getattr(tokenizer, "unk_token_id")
            ),
            attn_implementation=attn_implementation,
            torch_dtype=compute_dtype,
        )
        model.get_model().mm_speech_projector.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )
        audio_encoder = model.get_audio_encoder()
        audio_encoder.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

        if model_args.tune_audio_encoder:
            model.requires_grad_(False)
            for p in model.get_model().audio_encoder.parameters():
                p.requires_grad = True

        print(
            f"Trainable Audio Encoder parameters: {sum(p.numel() for p in model.get_model().audio_encoder.parameters() if p.requires_grad)}"
        )
        data_args.is_multimodal = True
        data_args.sampling_rate = audio_encoder.sampling_rate

        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tune_mm_mlp_adapter = (
            training_args.tune_mm_mlp_adapter
        ) = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            if (
                not model_args.tune_audio_encoder
            ):  # if we also tune the audio encoder, we don't want to set the grads to False
                model.requires_grad_(False)
            for p in model.get_model().mm_speech_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = (
            training_args.freeze_mm_mlp_adapter
        )
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_speech_projector.parameters():
                p.requires_grad = False

        model.config.mm_projector_lr = training_args.mm_projector_lr

        speech_projector = model.get_model().mm_speech_projector
        if (
            isinstance(speech_projector, SpeechProjector)
            and speech_projector.granular_losses is not None
        ):
            training_args.granular_losses = speech_projector.granular_losses

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.contrastive = training_args.contrastive_training
    model.config.contrastive_mode = training_args.contrastive_mode
    model.config.contrastive_layer = training_args.contrastive_layer
    model.config.contrastive_combine_loss = (
        training_args.contrastive_combine_loss
    )
    model.config.contr_asr_combine_loss = training_args.contr_asr_combine_loss
    model.config.tune_audio_encoder = model_args.tune_audio_encoder
    model.config.audio_nwp = training_args.audio_nwp
    model.config.position_shift = training_args.position_shift

    # Calculate total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.get_model().parameters())
    trainable_params = sum(
        p.numel() for p in model.get_model().parameters() if p.requires_grad
    )

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

   
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        contrastive=training_args.contrastive_training,
        audio_nwp=training_args.audio_nwp,
    )

    compute_metrics_on_generate_per_task = (
        {
            "ASR": compute_asr_metrics,
            "ST": compute_st_metrics,
        }
        if training_args.evaluation_strategy != "no"
        else None
    )
    wandb_tags = get_tags_from_arguments(model_args, data_args, training_args)

    trainer = LLaVATrainer(
        granular_losses=training_args.granular_losses,
        compute_metrics_per_task=None,
        compute_metrics_on_generate_per_task=compute_metrics_on_generate_per_task,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        wandb_tags=wandb_tags,
        **data_module,
    )


    trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        modality=training_args.modality,
    )


if __name__ == "__main__":
    train()
