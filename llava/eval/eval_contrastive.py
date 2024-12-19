import torch

from llava.mm_utils import process_audios


def do_contrastive_eval(
    preds,
    batch,
    batch_tasks,
    batch_audios_srs,
    processor,
    tokenizer,
    model,
    temperature,
    max_new_tokens,
):
    label_ids = tokenizer(
        batch["transcription"],
        padding=True,
        return_tensors="pt",
    )

    audios_srs = process_audios(batch_audios_srs, processor, model.config)
    audios_srs = [
        (
            audio.to(device=model.device, dtype=torch.float16),
            sr,
        )
        for audio, sr in audios_srs
    ]
    if batch_tasks[0] != "ASR":
        raise NotImplementedError(
            "Contrastive loss can only be calculated for ASR data!"
        )

    if tokenizer.padding_side == "right":
        raise NotImplementedError(
            "Contrastive loss can only be calculated for batches > 1"
        )

    with torch.inference_mode():
        losses = model.eval_contrastive(
            label_ids["input_ids"],  # input_ids are labels
            audios=audios_srs,
            do_sample=(True if temperature > 0 else False),
            attention_mask=label_ids[
                "attention_mask"
            ],  # no attention mask needed
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=None,
            use_cache=True,
            tokenizer=tokenizer,
        )

        preds.append(losses)
    return preds


def get_contrastive_metrics(results):
    metrics = {}
    avg_loss = [r["avg"] for r in results["preds"]]
    wasser_loss = [r["wasserstein"] for r in results["preds"]]
    metrics = {}
    metrics["task"] = (
        list(set(results["tasks"]))
        if len(set(results["tasks"])) > 1
        else results["tasks"][0]
    )
    metrics["source_language"] = (
        list(set(results["source_langs"]))
        if len(set(results["source_langs"])) > 1
        else results["source_langs"][0]
    )
    metrics["target_language"] = (
        list(set(results["target_langs"]))
        if len(set(results["target_langs"])) > 1
        else results["target_langs"][0]
    )
    metrics["metrics"] = {
        "contrastive loss average": (sum(avg_loss) / len(avg_loss)).item(),
        "contrastive loss wasserstein": (
            sum(wasser_loss) / len(wasser_loss)
        ).item(),
    }

    return [metrics]
