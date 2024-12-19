import torch
import torch.nn.functional as F


def prepare_mixed_nwp(
    bos_id,
    ignore_index,
    labels,
    word_tensors,
    transcription_ids,
    transcription_attention_mask,
    model,
    encode_audios,
    contrastive,
):

    (
        raw_audio_adapter_output,
        audio_features,
        audio_mask,
        padding_mask,
        nwp_labels,
    ) = get_mixed_nwp(
        bos_id,
        ignore_index,
        labels,
        word_tensors,
        transcription_ids,
        transcription_attention_mask,
        model,
        encode_audios,
    )

    assert padding_mask.shape == audio_mask.shape == nwp_labels.shape
    assert padding_mask.shape == audio_features.shape[0:2]

    if not contrastive:
        return (
            None,
            None,  # position ids
            ~padding_mask,
            None,  # past key values
            audio_features,  # new input embeds
            nwp_labels,  # padded labels are the same as the inputs in NWP, but with -100 where not relevant
            raw_audio_adapter_output,
        )

    device = labels.device

    # we also need pure NWP (without audio) to calculate a contrastive loss
    unpadded_text = [l[l != ignore_index] for l in labels]
    # add BOS token to labels
    unpadded_text = [
        torch.cat([torch.tensor([bos_id], device=device), tensor])
        for tensor in unpadded_text
    ]
    embedded_text = [model.embed_tokens(l) for l in unpadded_text]

    all_text_labels = []

    for i in range(len(unpadded_text)):
        subword_indices = word_tensors[i]["subword_indices"]
        speech_text_label = word_tensors[i]["speech_text_label"]
        assert (
            len(unpadded_text[i]) == subword_indices[-1][-1] + 3
        )  # because index with 0 + bos + eos
        # create a mask where we want to calculate loss (only for text tokens) and add True for eos + bos
        mask = torch.tensor(
            [True]
            + [
                l == "Text"
                for ind, l in zip(subword_indices, speech_text_label)
                for _ in ind
            ]
            + [True],
            device=device,
        )

        all_text_labels.append(
            torch.where(mask, unpadded_text[i], ignore_index)
        )

    # pad embedded text and corresponding labels
    padded_embed_text = []
    padded_text_labels = []
    padded_text_attn_mask = []
    max_length = max(embed.size(0) for embed in embedded_text)
    for e_idx, e in enumerate(embedded_text):
        padded_embed_text.append(F.pad(e, (0, 0, 0, max_length - e.size(0))))
        padded_text_attn_mask.append(torch.arange(max_length) < e.size(0))
        padded_text_labels.append(
            F.pad(
                all_text_labels[e_idx],
                (0, max_length - e.size(0)),
                value=ignore_index,
            )
        )

    return (
        None,
        None,  # position ids
        [~padding_mask, torch.stack(padded_text_attn_mask).to(device)],
        None,  # past key values
        [audio_features, torch.stack(padded_embed_text).to(device)],
        [nwp_labels, torch.stack(padded_text_labels).to(device)],
        raw_audio_adapter_output,
    )


def get_mixed_nwp(
    bos_id,
    ignore_index,
    labels,
    word_tensors,
    transcription_ids,
    transcription_attention_mask,
    model,
    encode_audios,
):

    import random

    random.seed(42)
    device = model.device

    unpadded_labels = [l[l != ignore_index] for l in labels]

    # add BOS token to labels
    unpadded_labels = [
        torch.cat([torch.tensor([bos_id], device=device), tensor])
        for tensor in unpadded_labels
    ]
    embedded_labels = [
        model.embed_tokens(l) for l in unpadded_labels
    ]  #  batch size x length x 4096

    batch_embeddings = []
    audio_masks = []
    batch_audio_nwp_labels = []

    for batch_index, batch_entry in enumerate(word_tensors):
        tensors = batch_entry[
            "tensors"
        ]  # these are the audio tensors for each word
        batch_subword_indices = batch_entry[
            "subword_indices"
        ]  # these are the subword indices for each word
        batch_speech_text = batch_entry["speech_text_label"]
        batch_labels = embedded_labels[batch_index].to(
            device
        )  # these are the transcriptions for each subword
        batch_label_ids = unpadded_labels[
            batch_index
        ]  # the ids for each subword

        # store eos embed
        eos_embed = batch_labels[-1]
        eos_id = batch_label_ids[-1].flatten().tolist()[0]

        # store bos embed but cut it off for now, because it was not taken into account when creating speech tensors
        bos_embed = batch_labels[0]
        batch_label_ids = batch_label_ids[1:]
        batch_labels = batch_labels[1:]

        # assert that we cover all subwords (-2 because <|eot_id|> is not included and subwords starts counting with 0)
        assert batch_subword_indices[-1][-1] == batch_labels.shape[0] - 2


        raw_speech_out = encode_audios(
            tensors,
            speech_projector_kwargs={
                "transcription_ids": None,
                "transcription_attention_mask": None,
            },
        )

        unpadded_audio_features = [
            a[~raw_speech_out.padding_mask[a_idx]]
            for a_idx, a in enumerate(raw_speech_out.audio_features)
        ]

        concatenated_embeddings = []
        audio_mask = []
        concatenated_labels = []

        for i, indices in enumerate(batch_subword_indices):
            audios_turn = batch_speech_text[i] == "Speech"

            if audios_turn:
                audio_embedding = unpadded_audio_features[i]
                concatenated_embeddings.append(audio_embedding)
                concatenated_labels += [ignore_index] * audio_embedding.shape[
                    0
                ]
                audio_mask += [True] * audio_embedding.shape[0]

            else:
                if len(indices) == 1:
                    text_embeddings = batch_labels[indices[0]].unsqueeze(0)
                    corresponding_ids = (
                        batch_label_ids[indices[0]].flatten().tolist()
                    )
                else:
                    text_embeddings = batch_labels[
                        indices[0] : indices[-1] + 1
                    ]
                    corresponding_ids = (
                        batch_label_ids[indices[0] : indices[-1] + 1]
                        .flatten()
                        .tolist()
                    )

                concatenated_embeddings.append(text_embeddings)
                audio_mask += [False] * text_embeddings.shape[0]
                concatenated_labels += corresponding_ids

        # add bos and eos token
        concatenated_embeddings = (
            [bos_embed.unsqueeze(0)]
            + concatenated_embeddings
            + [eos_embed.unsqueeze(0)]
        )
        concatenated_labels = [bos_id] + concatenated_labels + [eos_id]

        audio_mask = [False] + audio_mask + [False]

        batch_embeddings.append(
            torch.vstack(concatenated_embeddings)
        )  #  length x 4096
        audio_masks.append(torch.tensor(audio_mask, dtype=torch.bool))
        batch_audio_nwp_labels.append(torch.tensor(concatenated_labels))

    # do padding to add to batch

    max_length = max(embed.size(0) for embed in batch_embeddings)

    raw_speech_projector_output = {
        "audio_features": [],
        "padding_mask": [],
        "audio_mask": [],
        "nwp_labels": [],
    }
    for e_idx, e in enumerate(batch_embeddings):
        raw_speech_projector_output["audio_features"].append(
            F.pad(e, (0, 0, 0, max_length - e.size(0)))
        )
        raw_speech_projector_output["padding_mask"].append(
            torch.arange(max_length) >= e.size(0)
        )
        raw_speech_projector_output["audio_mask"].append(
            F.pad(audio_masks[e_idx], (0, max_length - e.size(0)), value=False)
        )
        raw_speech_projector_output["nwp_labels"].append(
            F.pad(
                batch_audio_nwp_labels[e_idx],
                (0, max_length - e.size(0)),
                value=ignore_index,
            )
        )

    class AudioFeatures:
        def __init__(self, data):
            for key, value in data.items():
                setattr(self, key, torch.stack(value).to(device))

    raw_speech_projector_output = AudioFeatures(raw_speech_projector_output)

    audio_features = raw_speech_projector_output.audio_features
    audio_mask = raw_speech_projector_output.audio_mask
    padding_mask = raw_speech_projector_output.padding_mask
    nwp_labels = raw_speech_projector_output.nwp_labels

    assert padding_mask.shape == audio_mask.shape == nwp_labels.shape
    assert padding_mask.shape == audio_features.shape[0:2]

    return (
        raw_speech_projector_output,
        audio_features,
        audio_mask,
        padding_mask,
        nwp_labels,
    )
