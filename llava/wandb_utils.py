def get_tags_from_arguments(model_args, data_args, training_args):

    model_name = model_args.model_name_or_path.split("/")[-1]
    audio_encoder_name = model_args.audio_encoder.split("/")[-1]
    speech_projector_name = model_args.mm_speech_projector_type
    if speech_projector_name == "cformer":
        speech_projector_name += f"_{model_args.cformer_length_adapter_type}"
    data_config_name = data_args.data_config_path.split("/")[-1].replace(
        ".yml", ""
    )
    pretrain_or_finetune = (
        "pretrain" if not training_args.freeze_mm_mlp_adapter else "finetune"
    )

    return [
        f"model: {model_name}",
        f"audio_encoder: {audio_encoder_name}",
        f"speech_projector: {speech_projector_name}",
        f"data_config: {data_config_name}",
        f"training_mode: {pretrain_or_finetune}",
    ]
