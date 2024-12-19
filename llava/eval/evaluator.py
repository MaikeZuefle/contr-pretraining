import json

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from llava.constants import AUDIO_TOKEN_INDEX, DEFAULT_AUDIO_TOKEN
from llava.conversation import conv_templates
from llava.dataset.config import (
    LANGUAGES_CODE_NAME,
    SOURCE_LANGUAGE_PLACEHOLDER,
    TARGET_LANGUAGE_PLACEHOLDER,
)
from llava.eval.eval_contrastive import (
    do_contrastive_eval,
    get_contrastive_metrics,
)
from llava.eval.metrics import (
    compute_asr_metrics,
    compute_sqa_metrics,
    compute_st_metrics,
)
from llava.mm_utils import process_audios, tokenizer_mm_token
from llava.train.llava_trainer import EvalOnGeneratePrediction


class Evaluator:
    def __init__(self, model, tokenizer, dataloader, processor, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.processor = processor
        self.whisper_llm = False

        for key in kwargs:
            if key in [
                "conv_mode",
                "temperature",
                "max_new_tokens",
                "contrastive",
            ]:
                setattr(self, key, kwargs[key])


    def _tokenize_prompts(self, inputs):
        tokenized_prompts = []
        for inp_idx, inp in enumerate(inputs):

            conv = conv_templates[
                self.conv_mode
            ].copy()  # clear conversation history only if another audio is currently loaded
            conv.append_message(conv.roles[0], inp)

            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()


            tokenized_prompt = tokenizer_mm_token(
                prompt,
                self.tokenizer,
                AUDIO_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0)

            tokenized_prompts.append(tokenized_prompt)
        return tokenized_prompts

    def _prepare_inputs(self, tokenized_prompts):
        tokenized_prompts = [
            ids[:, : self.tokenizer.model_max_length]
            for ids in tokenized_prompts
        ]
        max_len = max(
            [
                tokenized_prompt.size(1)
                for tokenized_prompt in tokenized_prompts
            ]
        )

        # pad the tokenized prompts and concatenate them to form a single tensor
        inputs_ids = torch.tensor([])
        for tokenized_prompt in tokenized_prompts:
            padding = torch.full(
                (1, max_len - tokenized_prompt.size(1)),
                (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else 0
                ),
                dtype=torch.long,
            )
            if self.tokenizer.padding_side == "right":
                tokenized_prompt = torch.cat(
                    (tokenized_prompt, padding), dim=1
                )
            else:
                tokenized_prompt = torch.cat(
                    (padding, tokenized_prompt), dim=1
                )

            inputs_ids = torch.cat((inputs_ids, tokenized_prompt), dim=0)

        attention_mask = inputs_ids.ne(self.tokenizer.pad_token_id)

        inputs_ids = inputs_ids.to(device=self.model.device, dtype=torch.long)
        attention_mask = attention_mask.to(
            device=self.model.device, dtype=torch.long
        )
        return inputs_ids, attention_mask

    def generate(self):
        # return the results in the form of a dictionary

        (
            tasks,
            source_langs,
            target_langs,
            prompts,
            gts,
            preds,
            transcriptions,
        ) = ([], [], [], [], [], [], [])

        for batch_id, batch in tqdm(
            enumerate(self.dataloader), total=len(self.dataloader)
        ):
            # from itertools import islice
            # for batch_id, batch in tqdm(enumerate(islice(self.dataloader, 4)), total=4):
            batch_tasks = batch["task"]
            batch_audios_srs = [
                (audio, sr) for audio, sr in zip(batch["audio"], batch["sr"])
            ]
            batch_source_languages = batch["source_language"]
            batch_target_languages = batch["target_language"]

            tasks.extend(batch_tasks)
            source_langs.extend(batch_source_languages)
            target_langs.extend(batch_target_languages)
            transcriptions.extend(batch["transcription"])
            gts.extend(batch["gt"])

            # get input from task
            if self.contrastive:
                preds = do_contrastive_eval(
                    preds,
                    batch,
                    batch_tasks,
                    batch_audios_srs,
                    self.processor,
                    self.tokenizer,
                    self.model,
                    self.temperature,
                    self.max_new_tokens,
                )

            else:
                inputs = []
                for i in range(len(batch_tasks)):
                    inp = self._get_input_from_task(
                        batch_tasks[i],
                        batch_source_languages[i],
                        batch_target_languages[i],
                    )

                    if batch_tasks[i] == "SQA":
                        question = next(
                            (
                                entry["value"]
                                for entry in batch["conversations"][i]
                                if entry["from"] == "human"
                            ),
                            None,
                        )
                        inp = inp + " " + question
                        transcriptions.append(None)

                    prompts.append(inp)
                    inputs.append(DEFAULT_AUDIO_TOKEN + "\n" + inp)

    

                tokenized_prompts = self._tokenize_prompts(
                    inputs,
                )
                inputs_ids, attn_mask = self._prepare_inputs(
                    tokenized_prompts
                )

                audios_srs = process_audios(
                    batch_audios_srs, self.processor, self.model.config
                )
                audios_srs = [
                    (
                        audio.to(
                            device=self.model.device, dtype=torch.float16
                        ),
                        sr,
                    )
                    for audio, sr in audios_srs
                ]

                with torch.inference_mode():

                    output_ids = self.model.generate(
                        inputs_ids,
                        audios=audios_srs,
                        do_sample=True if self.temperature > 0 else False,
                        attention_mask=attn_mask,
                        temperature=self.temperature,
                        max_new_tokens=self.max_new_tokens,
                        streamer=None,
                        use_cache=True,
                        tokenizer=self.tokenizer,
                    )
                outputs = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                torch.cuda.empty_cache()
                preds.extend(outputs)


        self.results = {
            "tasks": tasks,
            "source_langs": source_langs,
            "target_langs": target_langs,
            "prompts": prompts,
            "gts": gts,
            "preds": preds,
            "transcriptions": transcriptions,
        }
        return self.results

    def get_results(self):
        return self.results

    @staticmethod
    def compute_metrics(results, compute_comet=True, contrastive=False):

        if results is None:
            raise ValueError("No results to compute metrics on")

        if contrastive:
            return get_contrastive_metrics(results)

        results_df = pd.DataFrame(results)
        # fill None values in source_langs and target_langs with "None" string
        results_df["source_langs"] = results_df["source_langs"].fillna("None")
        results_df["target_langs"] = results_df["target_langs"].fillna("None")

        # group by task then by language direction (source-target)
        grouped = results_df.groupby(["tasks", "source_langs", "target_langs"])

        all_metrics = []

        for (task, source_lang, target_lang), group in grouped:
            metrics = {}

            preds = np.array(group["preds"])
            gts = np.array(group["gts"])

            if task.lower() == "asr":
                scores = compute_asr_metrics(
                    EvalOnGeneratePrediction(predictions=preds, references=gts)
                )
            elif task.lower() == "st":
                sources = np.array(group["transcriptions"])
                scores = compute_st_metrics(
                    EvalOnGeneratePrediction(
                        predictions=preds,
                        references=gts,
                        sources=sources if compute_comet else None,
                    )
                )

            elif task.lower() == "meta":
                scores_wer = compute_asr_metrics(
                    EvalOnGeneratePrediction(predictions=preds, references=gts)
                )
                scores_bleu = compute_st_metrics(
                    EvalOnGeneratePrediction(
                        predictions=preds,
                        references=gts,
                        sources=None,
                    )
                )
                scores = {**scores_bleu, **scores_wer}

            elif task.lower() == "sqa":
                scores = compute_sqa_metrics(
                    EvalOnGeneratePrediction(predictions=preds, references=gts)
                )

            else:
                raise NotImplementedError(
                    f"Task {task}has not been implemented yet."
                )

            # store metrics in the metrics dictionary
            metrics["task"] = task
            metrics["source_language"] = source_lang
            metrics["target_language"] = target_lang
            metrics["metrics"] = scores
            all_metrics.append(metrics)
        return all_metrics

    @staticmethod
    def _load_audio(audio_path):
        """
        Load audio from a local file path or URL using torchaudio.

        Parameters:
        - audio_path (str): The local path or URL of the audio file.

        Returns:
        - waveform (torch.Tensor): 1D tensor representing the audio waveform.
        - sample_rate (int): The sample rate of the audio.
        """

        try:
            # Load audio using torchaudio
            audio, sr = torchaudio.load(
                audio_path, normalize=True
            )  # should be normalized??
            return (audio, sr)
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None

    @staticmethod
    def _get_input_from_task(task, source_lang, target_lang):
        task = task.lower()
        if task == "asr":
            with open("llava/dataset/INPUTS_TEXT_LIST.json") as f:
                input_text_list = json.load(f)
                prompt = input_text_list["ASR"][source_lang][0]
                return prompt

        if task == "st":
            with open("llava/dataset/INPUTS_TEXT_LIST.json") as f:
                input_text_list = json.load(f)

            prompt_before_preprocessing = input_text_list["ST"]["en"][0]
            source_lang_full = LANGUAGES_CODE_NAME["en"][
                source_lang
            ]  # get the language name in the language itself
            target_lang_full = LANGUAGES_CODE_NAME["en"][
                target_lang
            ]  # get the language name in the original language

            prompt = prompt_before_preprocessing.replace(
                SOURCE_LANGUAGE_PLACEHOLDER, source_lang_full
            )
            prompt = prompt.replace(
                TARGET_LANGUAGE_PLACEHOLDER, target_lang_full
            )
            return prompt
        if task == "sqa":
            with open("llava/dataset/INPUTS_TEXT_LIST.json") as f:
                input_text_list = json.load(f)
                prompt = input_text_list["SQA"][source_lang][0]
                return prompt
        if task == "meta":
            with open("llava/dataset/INPUTS_TEXT_LIST.json") as f:
                input_text_list = json.load(f)
                prompt = input_text_list["META"][source_lang][0]
                return prompt
        else:
            raise NotImplementedError(f"Task {task} not implemented")
