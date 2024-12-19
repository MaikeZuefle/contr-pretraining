import torch
import torch.nn.functional as F


def prepare_contrastive_input(
    ignore_index, labels, raw_audio_adapter_output, model
):
    device = labels.device

    unpadded_labels = [l[l != ignore_index] for l in labels]

    unpadded_labels = [
        l[:-1] for l in unpadded_labels
    ]  # remove EOS token, we don't want it for contrastive comparison
    embedded_labels = [model.embed_tokens(l) for l in unpadded_labels]

    # pad embedded labels and create attention mask
    padded_embed_labels = []
    padded_label_mask = []
    max_length = max(embed.size(0) for embed in embedded_labels)
    for e in embedded_labels:
        padded_label_mask.append(torch.arange(max_length) < e.size(0))
        padded_embed_labels.append(F.pad(e, (0, 0, 0, max_length - e.size(0))))

    padded_embed_labels = torch.stack(padded_embed_labels).to(device)
    padded_label_mask = torch.stack(padded_label_mask).to(device)

    return (
        None,  # labels are input_ids in contrastive learning
        None,  # position ids
        padded_label_mask,
        None,  # past key values
        padded_embed_labels,  # new input embeds
        None,  # labels are not needed
        raw_audio_adapter_output,
    )


def prepare_contrastive_inference(
    model, attention_mask, input_ids, raw_audio_adapter_output
):
    device = model.device
    labels_attention_mask = attention_mask  # why not labels_attention mask?
    labels = input_ids
    labels = labels.to(device)

    for i in range(labels_attention_mask.shape[0]):
        # set EOS token in attention mask to False because we don't want it for contrastive comparison
        last_true_index = torch.where(labels_attention_mask[i])[0][-1]
        labels_attention_mask[i][last_true_index] = False

    embedded_labels = torch.stack([model.embed_tokens(l) for l in labels]).to(
        device
    )  #  batch size x length x 4096

    return (
        labels,  # labels are input_ids in contrastive learning
        None,  # position ids
        labels_attention_mask,
        None,  # past key values
        embedded_labels,  # new input embeds
        labels,
        raw_audio_adapter_output,
    )
