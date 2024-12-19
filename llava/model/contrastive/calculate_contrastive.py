import torch
import torch.nn.functional as F
from geomloss import SamplesLoss

from ..model_outputs import CausalLMOutputWithPastAndGranularLosses


def contrastive_main(
    inputs_embeds,
    attention_mask,
    audio_features,
    audio_attention_mask,
    contrastive_mode,
):
    device = inputs_embeds.device
    temperature = 0.1
    batch_size = inputs_embeds.shape[0]
    text_embeds_mask = attention_mask

    #  batch size x max length x 4096
    padded_text_embeddings = inputs_embeds
    padded_audio_embeddings = (
        audio_features  # raw_speech_projector_output.audio_features
    )

    #  batch size x length x 4096
    unpad_text_embeddings = [
        t[am] for t, am in zip(padded_text_embeddings, text_embeds_mask)
    ]
    unpad_audio_embeddings = [
        a[audio_attention_mask[a_idx]]
        for a_idx, a in enumerate(padded_audio_embeddings)
    ]

    if "average" in contrastive_mode:
        loss = contrastive_avg(
            unpad_text_embeddings,
            unpad_audio_embeddings,
            batch_size,
            temperature,
            device,
        )

    elif "wasserstein" in contrastive_mode:
        loss = contrastive_wasserstein(
            unpad_text_embeddings,
            unpad_audio_embeddings,
            batch_size,
            temperature,
            device,
        )

    else:
        raise NotImplementedError(
            "Use 'average', 'wasserstein' as contrastive_mode!"
        )

    granular_losses = {"contrastive_loss": loss}
    model_output_dict = {
        "logits": padded_audio_embeddings,
        "past_key_values": None,
        "hidden_states": padded_audio_embeddings,
        "attentions": audio_attention_mask,
    }

    return CausalLMOutputWithPastAndGranularLosses(
        loss=sum(granular_losses.values()),
        granular_losses=granular_losses,
        **model_output_dict,
    )


def contrastive_avg(
    unpad_text_embeddings,
    unpad_audio_embeddings,
    batch_size,
    temperature,
    device,
):
    avg_text_embed = torch.stack(
        [torch.mean(t, dim=0) for t in unpad_text_embeddings]
    )
    avg_audio_embed = torch.stack(
        [torch.mean(a, dim=0) for a in unpad_audio_embeddings]
    )

    # normalize
    avg_text_embed = F.normalize(avg_text_embed, p=2, dim=1).to(device)
    avg_audio_embed = F.normalize(avg_audio_embed, p=2, dim=1).to(device)

    similarity_matrix = (
        torch.matmul(avg_text_embed, avg_audio_embed.T).to(device)
        / temperature
    )

    pos_labels = torch.arange(batch_size, device=device)
    loss = F.cross_entropy(similarity_matrix, pos_labels)
    return loss


def clean_embeddings(embedding):
    embedding = torch.where(
        torch.isnan(embedding),
        torch.tensor(0.0, dtype=embedding.dtype),
        embedding,
    )
    embedding = torch.where(
        torch.isinf(embedding) & (embedding > 0),
        torch.tensor(1e6, dtype=embedding.dtype),
        embedding,
    )
    embedding = torch.where(
        torch.isinf(embedding) & (embedding < 0),
        torch.tensor(-1e6, dtype=embedding.dtype),
        embedding,
    )
    return embedding


def contrastive_wasserstein(
    unpad_text_embeddings,
    unpad_audio_embeddings,
    batch_size,
    temperature,
    device,
):
    # https://www.kernel-operations.io/geomloss/api/pytorch-api.html
    # p=2 is eucledian distance
    sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05)

    # Normalize embeddings
    unpad_text_embeddings = [
        F.normalize(embed.to(dtype=torch.float32).to(device), p=2, dim=-1)
        for embed in unpad_text_embeddings
    ]
    unpad_audio_embeddings = [
        F.normalize(embed.to(dtype=torch.float32).to(device), p=2, dim=-1)
        for embed in unpad_audio_embeddings
    ]

    distances = torch.zeros((batch_size, batch_size), device=device)

    for i in range(batch_size):
        for j in range(batch_size):
            try:
                distances[i, j] = sinkhorn_loss(
                    unpad_text_embeddings[i].to(dtype=torch.float32),
                    unpad_audio_embeddings[j].to(dtype=torch.float32),
                )

            except ValueError:
                distances[i, j] = sinkhorn_loss(
                    clean_embeddings(
                        unpad_text_embeddings[i].to(dtype=torch.float32)
                    ),
                    clean_embeddings(
                        unpad_audio_embeddings[j].to(dtype=torch.float32)
                    ),
                )

    similarity_matrix = -distances / temperature
    # diagonal are positive labels, rest is negative
    pos_labels = torch.arange(batch_size, device=device)
    loss = F.cross_entropy(similarity_matrix, pos_labels)
    return loss


def calculate_nwp_contrastive(speech, text):
    """
    speech: list with batch example. Each example is of shape (seq length, emb_dim)
    text: list with batch example. Each example is of shape (seq length, emb_dim)
    """
    temperature = 0.5

    contrastive_losses = []
    for i in range(len(speech)):
        speech_ex = speech[i]  # (seq_len, emb_dim)
        text_ex = text[i]  # (seq_len, emb_dim)

        speech_ex_normalized = F.normalize(speech_ex, p=2, dim=-1)
        text_ex_normalized = F.normalize(text_ex, p=2, dim=-1)

        similarity_matrix = (
            torch.matmul(text_ex_normalized, speech_ex_normalized.T)
            / temperature
        )

        pos_labels = torch.arange(
            similarity_matrix.size(0), device=similarity_matrix.device
        )
        loss = F.cross_entropy(similarity_matrix, pos_labels)
        contrastive_losses.append(loss)

    total_loss = torch.mean(torch.stack(contrastive_losses))

    granular_losses = {"contrastive_loss": total_loss}
    model_output_dict = {
        "logits": None,
        "past_key_values": None,
        "hidden_states": None,
        "attentions": None,
    }

    return CausalLMOutputWithPastAndGranularLosses(
        loss=sum(granular_losses.values()),
        granular_losses=granular_losses,
        **model_output_dict,
    )
