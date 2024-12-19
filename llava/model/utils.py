import math
from functools import singledispatch

import torch
from transformers import AutoConfig


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if "llava" in config and "llava" not in cfg.model_type:
        assert cfg.model_type == "llama"
        print(
            "You are using newer LLaVA code base, while the checkpoint of v0 is from older code base."
        )
        print(
            "You must upgrade the checkpoint to the new code base (this can be done automatically)."
        )
        confirm = input(
            "Please confirm that you want to upgrade the checkpoint. [Y/N]"
        )
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = "LlavaLlamaForCausalLM"
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)


@singledispatch
def compute_output_length_from_conv1d_hyperparams(
    input_length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    return math.floor(
        (input_length + 2 * padding - dilation * (kernel_size - 1) - 1)
        / stride
        + 1
    )


@compute_output_length_from_conv1d_hyperparams.register
def _(
    input_length: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
):
    return torch.floor(
        (input_length + 2 * padding - dilation * (kernel_size - 1) - 1)
        / stride
        + 1
    ).long()


@singledispatch
def compute_output_length_from_conv1d_layer(
    input_length: int, conv1d_layer: torch.nn.Conv1d
) -> int:
    return compute_output_length_from_conv1d_hyperparams(
        input_length,
        conv1d_layer.kernel_size[0],
        conv1d_layer.stride[0],
        conv1d_layer.padding[0],
        conv1d_layer.dilation[0],
    )


@compute_output_length_from_conv1d_layer.register
def _(input_length: torch.Tensor, conv1d_layer: torch.nn.Conv1d):
    return compute_output_length_from_conv1d_hyperparams(
        input_length,
        conv1d_layer.kernel_size[0],
        conv1d_layer.stride[0],
        conv1d_layer.padding[0],
        conv1d_layer.dilation[0],
    )