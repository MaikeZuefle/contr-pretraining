import torch

from llava.model.contrastive.calculate_contrastive import (
    calculate_nwp_contrastive,
    contrastive_main,
)
from llava.model.model_outputs import CausalLMOutputWithPastAndGranularLosses


def forward_contrastive_plain(
    config,
    raw_speech_projector_output,
    inputs_embeds,
    attention_mask,
    use_cache,
    decoder,
):
    if config.contrastive_layer != "all":
        config.contrastive_layer = int(config.contrastive_layer)

    if config.contrastive_layer == 0:
        # the contrastive loss is calculated on the embedding layer of the LLM
        return contrastive_main(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            audio_features=raw_speech_projector_output.audio_features,
            audio_attention_mask=~raw_speech_projector_output.padding_mask,
            contrastive_mode=config.contrastive_mode,
        )

    # change position ids if needed
    if config.position_shift:
        shift = torch.randint(0, 400, (1, 1)).item()
        position_ids_text_shifted = (
            torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)
            + shift
        )
        position_ids_speech_shifted = (
            torch.arange(
                raw_speech_projector_output.audio_features.shape[1],
                device=inputs_embeds.device,
            ).unsqueeze(0)
            + shift
        )

    else:
        position_ids_text_shifted = None
        position_ids_speech_shifted = None

    # get LLM output for text
    model_output_text = decoder(
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        labels=None,
        use_cache=use_cache,
        output_hidden_states=True,
        position_ids=position_ids_text_shifted,
    )

    # get LLM output for speech
    model_output_speech = decoder(
        attention_mask=~raw_speech_projector_output.padding_mask,
        inputs_embeds=raw_speech_projector_output.audio_features,
        labels=None,
        use_cache=use_cache,
        output_hidden_states=True,
        position_ids=position_ids_speech_shifted,
    )

    if config.contrastive_layer != "all":
        last_hidden_text = model_output_text.hidden_states[
            config.contrastive_layer
        ]
        last_hidden_speech = model_output_speech.hidden_states[
            config.contrastive_layer
        ]

        return contrastive_main(
            inputs_embeds=last_hidden_text,
            attention_mask=attention_mask,
            audio_features=last_hidden_speech,
            audio_attention_mask=~raw_speech_projector_output.padding_mask,
            contrastive_mode=config.contrastive_mode,
        )

    else:

        # get contr loss for all layers and then sum in the end
        granular_losses = {}
        for layer in [
            0,
            5,
            10,
            15,
            20,
            25,
            len(model_output_speech.hidden_states) - 1,
        ]:
            layer_hidden_text = model_output_text.hidden_states[layer]
            layer_hidden_speech = model_output_speech.hidden_states[layer]
            contr_out = contrastive_main(
                inputs_embeds=layer_hidden_text,
                attention_mask=attention_mask,
                audio_features=layer_hidden_speech,
                audio_attention_mask=~raw_speech_projector_output.padding_mask,
                contrastive_mode=config.contrastive_mode,
            )

            granular_losses[layer] = contr_out.granular_losses[
                "contrastive_loss"
            ]

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


def forward_contrastive_mixed_nwp(
    config, inputs_embeds, attention_mask, labels, use_cache, decoder
):
    if config.contrastive_layer != "all":
        config.contrastive_layer = int(config.contrastive_layer)

    # speech is actually mixed speech + text
    speech_attn, text_att = attention_mask
    speech_emb, text_emb = inputs_embeds
    speech_l, text_l = labels

    if config.contrastive_layer == 0:
        return contrastive_main(
            inputs_embeds=text_emb,
            attention_mask=text_att,
            audio_features=speech_emb,
            audio_attention_mask=speech_attn,
            contrastive_mode=config.contrastive_mode,
        )

    model_output_text = decoder(
        attention_mask=text_att,
        inputs_embeds=text_emb,
        labels=text_l,
        use_cache=use_cache,
        output_hidden_states=True,
    )

    model_output_speech = decoder(
        attention_mask=speech_attn,
        inputs_embeds=speech_emb,
        labels=speech_l,
        use_cache=use_cache,
        output_hidden_states=True,
    )

    if config.contrastive_layer != "all":
        hidden_text = model_output_text.hidden_states[config.contrastive_layer]
        hidden_speech = model_output_speech.hidden_states[
            config.contrastive_layer
        ]

        relevant_speech = [
            s[r] for s, r in zip(hidden_speech, speech_l != -100)
        ]
        relevant_text = [t[r] for t, r in zip(hidden_text, text_l != -100)]
        assert len(relevant_speech) == len(relevant_text)

        if config.contrastive_mode == "subword":
            contr_out = calculate_nwp_contrastive(
                speech=relevant_speech, text=relevant_text
            )

        elif (
            config.contrastive_mode == "wasserstein"
            or config.contrastive_mode == "average"
        ):
            contr_out = contrastive_main(
                inputs_embeds=hidden_text,
                attention_mask=text_l != -100,
                audio_features=hidden_speech,
                audio_attention_mask=speech_l != -100,
                contrastive_mode=config.contrastive_mode,
            )
        else:
            raise NotImplementedError(
                f"AudioNWP Contr is only implemented for subword, wasserstein and average and not {config.contrastive_mode}"
            )

        # we can add the speech/text NWP loss to the contrastive lsos
        if config.contrastive_combine_loss:
            nwp_loss = model_output_speech.loss
            contr_out.granular_losses["audio_nwp_loss"] = nwp_loss
            contr_out.loss += nwp_loss

        return contr_out
    else:
        granular_losses = {}
        for layer in [
            0,
            5,
            10,
            15,
            20,
            25,
            len(model_output_speech.hidden_states) - 1,
        ]:

            layer_hidden_text = model_output_text.hidden_states[layer]
            layer_hidden_speech = model_output_speech.hidden_states[layer]

            contr_out = contrastive_main(
                inputs_embeds=layer_hidden_text,
                attention_mask=text_l != -100,
                audio_features=layer_hidden_speech,
                audio_attention_mask=speech_l != -100,
                contrastive_mode=config.contrastive_mode,
            )
            granular_losses[layer] = contr_out.granular_losses[
                "contrastive_loss"
            ]

        # we can add the speech/text NWP loss to the contrastive lsos
        if config.contrastive_combine_loss:
            nwp_loss = model_output_speech.loss
            granular_losses["audio_nwp_loss"] = nwp_loss

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


def forward_contrastive(
    config,
    raw_speech_projector_output,
    word_tensors,
    inputs_embeds,
    attention_mask,
    labels,
    use_cache,
    decoder,
):

    if (word_tensors == None) and (not torch.is_inference_mode_enabled()):
        return forward_contrastive_plain(
            config,
            raw_speech_projector_output,
            inputs_embeds,
            attention_mask,
            use_cache,
            decoder,
        )

    if word_tensors:
        return forward_contrastive_mixed_nwp(
            config, inputs_embeds, attention_mask, labels, use_cache, decoder
        )
