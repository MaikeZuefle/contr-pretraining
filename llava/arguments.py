import logging
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Union

import transformers

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    version: Optional[str] = field(default="llama_3_1")
    freeze_backbone: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_audio_encoder: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None) # not supported anymore! this code can only handle audio
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_mm_speech_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    audio_encoder: Optional[str] = field(default=None)
    mm_speech_projector_type: Optional[str] = field(
        default="residual_mlp"
    )  # FIXME
    mm_use_speech_start_end: bool = field(default=False)
    tokenizer_padding_side: Optional[str] = field(default="right")

    # Qformer ######################################################
    mm_qformer_num_query_token: Optional[int] = field(default=None)
    mm_qformer_num_hidden_layers: Optional[int] = field(default=None)
    qformer_num_query_token: Optional[int] = field(default=None)
    qformer_num_hidden_layers: Optional[int] = field(default=None)
    qformer_interpolation: Optional[bool] = field(default=False)
    qformer_interpolation_config: Optional[str] = field(default=None)
    qformer_num_attention_heads: Optional[int] = field(default=12)
    qformer_second_per_window: Optional[float] = field(default=0.3333333333333)
    ################################################################




@dataclass
class DataArguments:
    data_config_path: str = field(
        default=None, metadata={"help": "Path to the training data config."}
    )
    dataloader_debug: bool = field(default=False)
    filter_broken_samples: bool = field(default=False)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    sampling_rate: Optional[int] = None
    organize_eval_dataset_per_task: bool = field(default=True)
    rebuild_dataset_cache: bool = field(default=False)
    data_subset: Optional[float] = None
    no_punctuation: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    modality: str = field(default="audio") # image is not supported!
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    num_steps_between_each_restart: Optional[int] = None
    lr_min: Optional[float] = 1e-6
    eval_temperature: Optional[float] = field(default=0)
    eval_max_new_tokens: Optional[int] = field(default=200)
    eval_num_batched_generations: Optional[int] = field(default=2)
    ########## contrastive #############
    contrastive_training: Optional[bool] = False
    contrastive_mode: Optional[str] = field(default="average")
    contrastive_layer: Optional[Union[int, str]] = field(
        default=0
    )  # -1 for last layer, 0 for first..., "all" for sum over all layers
    contrastive_combine_loss: Optional[bool] = False
    position_shift: Optional[bool] = False
    audio_nwp: Optional[bool] = False
    contr_asr_combine_loss: Optional[bool] = False
    ###################################
