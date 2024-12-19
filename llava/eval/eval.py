import argparse
import json
import os
from pathlib import Path

import torch

from llava.arguments import DataArguments
from llava.dataset.datasets_wrapper import DatasetsWrapper
from llava.eval.evaluator import Evaluator

# import from utils in the same folder as this (llava/eval/utils)
from llava.eval.utils.eval_dataloader import get_dataloader
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--modality", type=str, default="audio")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--from-yml", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--tokenizer-padding-side", type=str, default="left")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--contrastive", action="store_true")

    args = parser.parse_args()

    assert args.model_path is not None
    assert args.results_dir is not None
    assert args.dataset is not None

    # also put into lower case the task and dataset
    args.dataset = args.dataset.lower()

    return args


def get_results_path(dataset_name_or_path, results_dir, results_name):
    # If there is already a file with the same name, add a number to the
    # end of the file name to avoid overwriting
    dataset_name = Path(dataset_name_or_path).stem
    results_path = Path(results_dir, f"{dataset_name}_{results_name}.json")

    i = 1
    while results_path.exists():
        results_path = Path(
            results_dir, f"{dataset_name}_{i}_{results_name}.json"
        )
        i += 1

    return results_path


def get_preds_path(results_dir, dataset):
    # If there is already a file with the same name, add a number to the end of the file name to avoid overwriting
    results_name = "preds"
    return get_results_path(dataset, results_dir, results_name)


def get_scores_path(results_dir, dataset, contrastive=False):
    results_name = "scores"
    if contrastive:
        results_name += "_contrastive"
    return get_results_path(dataset, results_dir, results_name)


def write_preds(results_dict, results_dir, dataset):
    json_results_path = get_preds_path(results_dir, dataset)
    results_list = []
    for (
        task,
        source_lang,
        target_lang,
        prompt,
        gt,
        pred,
        transcription,
    ) in zip(
        results_dict["tasks"],
        results_dict["source_langs"],
        results_dict["target_langs"],
        results_dict["prompts"],
        results_dict["gts"],
        results_dict["preds"],
        results_dict["transcriptions"],
    ):
        if task == "sqa":
            gt = json.load(gt)

        results_list.append(
            {
                "prompt": prompt,
                "transcription": transcription,
                "predictions": pred,
                "groundtruth": gt,
                "task": task,
                "source_language": source_lang,
                "target_language": target_lang,
            }
        )

    with open(json_results_path, "w", encoding="utf-8") as json_file:
        json.dump(results_list, json_file, ensure_ascii=False, indent=4)


def write_scores(
    all_metrics, results_dir, dataset, contrastive=False
):
    json_scores_path = get_scores_path(
        results_dir, dataset, contrastive=contrastive
    )
    with open(json_scores_path, "w", encoding="utf-8") as json_file:
        json.dump(all_metrics, json_file, ensure_ascii=False, indent=4)


def main():
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    args = args_parser()
    if "llava" in args.model_path:
        with open(args.model_path + "/config.json") as f:
            config = json.load(f)
            model_base = config["_name_or_path"]

    results_dir = os.path.join(args.results_dir)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if args.from_yml:
        data_config_path = args.dataset

        data_args = DataArguments(
            data_config_path=data_config_path,
            is_multimodal=True,
            dataloader_debug=False,
        )
        sqa_type_data = "spokensquad" in data_config_path

        dataset = DatasetsWrapper(data_args)
        dataloader = get_dataloader(
            dataset.test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sqa_type_data=sqa_type_data,
        )
    else:
        raise NotImplementedError



    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)

    assert (
        args.modality is not None
        and args.modality == "audio"
    ), "modality must be either 'audio'"

    # evaluate some pretrained model

    if not "llava" in args.model_path:
        raise ValueError("Model paths should start with 'llava'!")

    # evaluate model further pretrained/finetuned with this codebase
    else:
        tokenizer, model, processor, _ = load_pretrained_model(
            args.model_path,
            model_base,
            model_name,
            modality=args.modality,
            device=args.device,
        )

     
        if "meta-llama/Meta-Llama-3.1-8B-Instruct" == model_base:
            model.generation_config.eos_token_id = 128009
            conv_mode = "llama_3_1"
        else:
            raise NotImplementedError

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode

        tokenizer.padding_side = args.tokenizer_padding_side
        tokenizer.pad_token = tokenizer.eos_token  # pad token is eos token

        model.config.tokenizer_padding_side = args.tokenizer_padding_side
        evaluator = Evaluator(
            model=model,
            dataloader=dataloader,
            tokenizer=tokenizer,
            processor=processor,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            conv_mode=args.conv_mode,
            contrastive=args.contrastive
        )

    results_dict = evaluator.generate()
    if not args.contrastive:
        write_preds(
            results_dict,
            results_dir,
            args.dataset
        )

    all_metrics = Evaluator.compute_metrics(
        results_dict, contrastive=args.contrastive
    )
    write_scores(
        all_metrics,
        results_dir,
        args.dataset,
        contrastive=args.contrastive
    )


if __name__ == "__main__":
    main()
