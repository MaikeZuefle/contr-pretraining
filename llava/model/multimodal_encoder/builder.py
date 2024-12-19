import os

from .audio_encoder import HubertEncoder

def build_audio_encoder(
    cfg, delay_load=False, attn_implementation=None, torch_dtype=None
):
    audio_enc_name = getattr(
        cfg, "mm_audio_encoder", getattr(cfg, "audio_encoder", None)
    )
    print(f"\n***{audio_enc_name=}\n")

    if "hubert" in audio_enc_name.lower():
        return HubertEncoder(
            audio_enc_name,
            delay_load=delay_load,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )

    else:
        raise ValueError(f"Unknown audio encoder: {audio_enc_name}")