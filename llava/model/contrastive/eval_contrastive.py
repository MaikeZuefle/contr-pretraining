from typing import Optional, Union

import torch
from transformers.generation.utils import GenerateOutput

from .calculate_contrastive import contrastive_avg, contrastive_wasserstein


@torch.no_grad()
def eval_contr(
    prepare_inputs_labels_for_multimodal,
    input_ids: Optional[torch.Tensor] = None,
    audios: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:

    labels = input_ids

    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)

    out = prepare_inputs_labels_for_multimodal(
        input_ids,
        None,
        attention_mask,
        None,
        labels,
        images,
        image_sizes,
        audios,
        transcription_ids=labels,
        transcription_attention_mask=attention_mask,
        contrastive_inference=True,
    )

    (
        _,
        _,
        attention_mask,
        _,
        inputs_embeds,
        labels,
        raw_speech_projector_output,
    ) = out

    device = inputs_embeds.device
    temperature = 0.1
    batch_size = inputs_embeds.shape[0]
    text_embeds_mask = attention_mask

    #  batch size x max length x 4096
    padded_text_embeddings = inputs_embeds
    padded_audio_embeddings = raw_speech_projector_output.audio_features

    #  batch size x length x 4096
    unpad_text_embeddings = [
        t[am] for t, am in zip(padded_text_embeddings, text_embeds_mask)
    ]
    unpad_audio_embeddings = [
        a[~raw_speech_projector_output.padding_mask[a_idx]]
        for a_idx, a in enumerate(padded_audio_embeddings)
    ]

    losses = {}
    losses["avg"] = contrastive_avg(
        unpad_text_embeddings,
        unpad_audio_embeddings,
        batch_size,
        temperature,
        device,
    )
    losses["wasserstein"] = contrastive_wasserstein(
        unpad_text_embeddings,
        unpad_audio_embeddings,
        batch_size,
        temperature,
        device,
    )

    return losses
