import logging
import math
import os
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    is_datasets_available,
    is_torch_tpu_available,
)
from transformers.debug_utils import DebugOption
from transformers.integrations import WandbCallback
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
    logger,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalPrediction,
    speed_metrics,
)

import wandb
from llava.constants import IGNORE_INDEX
from llava.eval.metrics import EvalOnGeneratePrediction

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True, name=k).cpu()
        for k, v in to_return.items()
    }
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu()
        for k, v in to_return.items()
    }
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(
    lengths, batch_size, world_size, generator=None
):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(
            lengths, batch_size, world_size, generator=generator
        )
    mm_indices, mm_lengths = zip(
        *[(i, l) for i, l in enumerate(lengths) if l > 0]
    )
    lang_indices, lang_lengths = zip(
        *[(i, -l) for i, l in enumerate(lengths) if l < 0]
    )

    mm_shuffle = [
        mm_indices[i]
        for i in get_length_grouped_indices(
            mm_lengths, batch_size, world_size, generator=None
        )
    ]
    lang_shuffle = [
        lang_indices[i]
        for i in get_length_grouped_indices(
            lang_lengths, batch_size, world_size, generator=None
        )
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [
        mm_shuffle[i : i + megabatch_size]
        for i in range(0, len(mm_shuffle), megabatch_size)
    ]
    lang_megabatches = [
        lang_shuffle[i : i + megabatch_size]
        for i in range(0, len(lang_shuffle), megabatch_size)
    ]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(
    lengths, batch_size, world_size, generator=None, merge=True
):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i : i + megabatch_size].tolist()
        for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)
        for megabatch in megabatches
    ]
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size)
        for megabatch in megabatches
    ]

    return [
        i for megabatch in megabatches for batch in megabatch for i in batch
    ]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths,
                self.batch_size,
                self.world_size,
                generator=self.generator,
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths,
                self.batch_size,
                self.world_size,
                generator=self.generator,
            )
        return iter(indices)


class AddGranularLossesToTrainerState(TrainerCallback):
    # Taken from https://github.com/naba89/custom_hf_trainer/blob/main/custom_hf_trainer/custom_trainer.py
    def __init__(self, granular_losses: List[str]):
        self.granular_losses = granular_losses

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        control.granular_losses = {
            k: torch.tensor(0.0).to(args.device) for k in self.granular_losses
        }
        return control


class LLaVATrainer(Trainer):
    def __init__(
        self,
        granular_losses: List[str] = None,
        compute_metrics_per_task: Dict[str, callable] = None,
        compute_metrics_on_generate_per_task: Dict[str, callable] = None,
        wandb_tags: List[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wandb_tags = wandb_tags
        self.already_logged_tags = False
        if granular_losses is not None:
            self.add_callback(AddGranularLossesToTrainerState(granular_losses))

        self.eval_dataloader = None
        self.compute_metrics_per_task = compute_metrics_per_task
        self.compute_metrics_on_generate_per_task = (
            compute_metrics_on_generate_per_task
        )

    def _create_eval_dataloader(self, eval_dataset):
        logger.warning(
            "Eval dataloader is being created (again)."
            "Ignore if this is the first time you are seeing this message."
            "Otherwise, check the LlavaTrainer implementation."
        )
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, Dataset):
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="evaluation"
            )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(
            DataLoader(eval_dataset, **dataloader_params)
        )

    def get_eval_dataloader(
        self,
        eval_dataset: Optional[Dataset] = None,
        task_name: Optional[str] = None,
    ) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                WARNING: Not used here, only for signature compatibility.
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
            task_name (`str`, *optional*): self.eval_dataset key to get the dataloader for a specific task.

        """
        if eval_dataset is not None:
            raise ValueError(
                "In this custom trainer, you need to pass the name of dataset"
            )

        if task_name is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        if self.eval_dataloader is None:
            # create all eval dataloaders if they don't exist
            if isinstance(self.eval_dataset, dict):
                self.eval_dataloader = {}
                for _task_name, _eval_dataset in self.eval_dataset.items():
                    self.eval_dataloader[_task_name] = (
                        self._create_eval_dataloader(_eval_dataset)
                    )
            else:
                self.eval_dataloader = self._create_eval_dataloader(
                    self.eval_dataset
                )

        if isinstance(self.eval_dataloader, dict):
            return self.eval_dataloader[task_name]
        else:
            return self.eval_dataloader

    def compute_loss(self, model, inputs, return_outputs=False):

        if hasattr(self.control, "granular_losses") and model.training:
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
            )

            if not isinstance(outputs, dict):
                raise ValueError(
                    "The model output should be a dictionary or ModelOutput and not a tuple or list."
                )
            if "granular_losses" in outputs:
                for k, v in outputs["granular_losses"].items():
                    if k in self.control.granular_losses:
                        if v is not None:
                            if self.args.n_gpu > 1:
                                v = v.mean()
                            self.control.granular_losses[k] += (
                                v.detach()
                                / self.args.gradient_accumulation_steps
                            )
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs
            )

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
    ):
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            if is_torch_tpu_available():
                xm.mark_step()
            if not self.already_logged_tags:
                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    wandb.run.tags = self.wandb_tags
                    self.already_logged_tags = True

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            if hasattr(self.control, "granular_losses"):
                for k, v in self.control.granular_losses.items():
                    logs[k] = self._nested_gather(v).mean().item()
                    # reset the loss
                    self.control.granular_losses[
                        k
                    ] -= self.control.granular_losses[k]

                    logs[k] = round(
                        logs[k]
                        / (
                            self.state.global_step
                            - self._globalstep_last_logged
                        ),
                        4,
                    )

            logs["learning_rate"] = self._get_learning_rate()
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for task, eval_dataset in self.eval_dataset.items():
                    if (
                        self.compute_metrics_per_task is None
                        or task not in self.compute_metrics_per_task
                    ):
                        logging.warning(
                            f"Task {task} does not have a compute_metrics function. Skipping evaluation for this task."
                        )
                        compute_metrics_fn = None
                    else:
                        compute_metrics_fn = self.compute_metrics_per_task[
                            task
                        ]
                    if (
                        self.compute_metrics_on_generate_per_task is None
                        or task
                        not in self.compute_metrics_on_generate_per_task
                    ):
                        logging.warning(
                            f"Task {task} does not have a compute_metrics_on_generate function. Skipping evaluation for this task."
                        )
                        compute_metrics_on_generate_fn = None
                    else:
                        compute_metrics_on_generate_fn = (
                            self.compute_metrics_on_generate_per_task[task]
                        )
                    dataset_metrics = self.evaluate(
                        eval_dataset=task,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{task}",
                        compute_metrics_fn=compute_metrics_fn,
                        compute_metrics_on_generate_fn=compute_metrics_on_generate_fn,
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def evaluate(
        self,
        eval_dataset: Union[str, Dataset, Dict[str, Dataset]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        compute_metrics_fn: Optional[callable] = None,
        compute_metrics_on_generate_fn: Optional[callable] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Union[str, Dataset, Dict[str, Dataset]]`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with task names as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This is useful if you have multiple evaluation datasets for
                different tasks and different metrics for each task.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data1_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multipe eval datasets
        eval_dataset = (
            eval_dataset if eval_dataset is not None else self.eval_dataset
        )
        if isinstance(eval_dataset, dict):
            metrics = {}
            for task, _ in self.eval_dataset.items():
                if (
                    self.compute_metrics_per_task is None
                    or task not in self.compute_metrics_per_task
                ):
                    logging.warning(
                        f"Task {task} does not have a compute_metrics function. Skipping evaluation for this task."
                    )
                    compute_metrics_fn = None
                else:
                    compute_metrics_fn = self.compute_metrics_per_task[task]
                if (
                    self.compute_metrics_on_generate_per_task is None
                    or task not in self.compute_metrics_on_generate_per_task
                ):
                    logging.warning(
                        f"Task {task} does not have a compute_metrics_on_generate function. Skipping evaluation for this task."
                    )
                    compute_metrics_on_generate_fn = None
                else:
                    compute_metrics_on_generate_fn = (
                        self.compute_metrics_on_generate_per_task[task]
                    )
                dataset_metrics = self.evaluate(
                    eval_dataset=task,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"eval_{task}",
                    compute_metrics_fn=compute_metrics_fn,
                    compute_metrics_on_generate_fn=compute_metrics_on_generate_fn,
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(task_name=eval_dataset)

        _compute_metrics = self.compute_metrics
        if compute_metrics_fn is not None:
            self.compute_metrics = compute_metrics_fn

        start_time = time.time()

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )

        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=(
                True if self.compute_metrics is None else None
            ),
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[
                f"{metric_key_prefix}_jit_compilation_time"
            ]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        if compute_metrics_on_generate_fn is not None:
            self.compute_metrics = compute_metrics_on_generate_fn

        metrics_on_generate = self.generation_loop(
            eval_dataloader, metric_key_prefix, ignore_keys
        )
        # reset the compute_metrics function
        self.compute_metrics = _compute_metrics

        # add metrics on generate to output.metrics
        output.metrics.update(metrics_on_generate)

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def generation_loop(
        self,
        dataloader: DataLoader,
        metric_key_prefix: str = "eval",
        ignore_keys: Optional[List[str]] = None,
    ) -> Dict[str, float]:

        args = self.args
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(
            self.model, training=False, dataloader=dataloader
        )

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(
                    model, evaluation_mode=True
                )
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        model.eval()

        num_batch = len(dataloader)
        if args.eval_num_batched_generations is not None:
            num_batch = args.eval_num_batched_generations

        references = []
        predictions = []

        # sources = []
        def mask_and_batch_decode(tokenizer, ids):
            mask = ids != IGNORE_INDEX
            new_ids = np.zeros_like(ids)
            new_ids[mask] = ids[mask]
            return tokenizer.batch_decode(new_ids, skip_special_tokens=True)

        # set padding side to left
        self.model.config.tokenizer_padding_side = "left"
        self.tokenizer.padding_side = "left"
        for step, inputs in tqdm(
            enumerate(dataloader),
            total=num_batch,
            desc="(eval) compute_metrics_on_generate",
        ):
            if step >= num_batch:
                break

            inputs = self._prepare_inputs(inputs)

            # masking label tokens
            attention_mask = inputs["attention_mask"].clone()
            for i, labels in enumerate(inputs["labels"]):
                idxs = torch.where(labels != -100)[0]
                attention_mask[i, idxs] = False

            with torch.no_grad():
                generated_outputs_ids = model.generate(
                    inputs=inputs["input_ids"],
                    audios=inputs["audios_srs"],
                    attention_mask=attention_mask,
                    do_sample=(
                        True
                        if args.eval_temperature is not None
                        and args.eval_temperature > 0
                        else False
                    ),
                    temperature=(
                        args.eval_temperature
                        if args.eval_temperature > 0
                        else None
                    ),
                    max_new_tokens=args.eval_max_new_tokens,
                    use_cache=True,
                )
            generated_outputs_ids = generated_outputs_ids.cpu()
            label_ids = inputs["labels"].cpu()
            # transcription_ids = inputs["transcription_ids"].cpu()

            references.extend(mask_and_batch_decode(self.tokenizer, label_ids))
            predictions.extend(
                mask_and_batch_decode(self.tokenizer, generated_outputs_ids)
            )
            # sources.extend(
            #     mask_and_batch_decode(self.tokenizer, transcription_ids)
            # )
        # restore padding side to right
        model.config.tokenizer_padding_side = "right"
        self.tokenizer.padding_side = "right"

        metrics = {}
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(
                EvalOnGeneratePrediction(
                    predictions=predictions,
                    references=references,
                    sources=None,
                )
            )

            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        def log_pairs(references, predictions, table_name):
            table = wandb.Table(columns=["Reference", "Prediction", "Score"])
            for ref, pred in zip(references, predictions):
                score = None
                if self.compute_metrics is not None:
                    score = self.compute_metrics(
                        EvalOnGeneratePrediction(
                            predictions=[pred],
                            references=[ref],
                            sources=None,  # do not compute comet score during training
                        )
                    )
                table.add_data(ref, pred, score)
            wandb.log({table_name: table})

        # check if wandb is enabled
        if wandb.run is not None:
            log_pairs(
                references, predictions, f"{metric_key_prefix}_predictions"
            )

        return metrics

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size
                * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(
                opt_model, ALL_LAYERNORM_LAYERS
            )
            decay_parameters = [
                name for name in decay_parameters if "bias" not in name
            ]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "mm_projector" or "mm_speech_projector" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = (
                Trainer.get_optimizer_cls_and_kwargs(self.args)
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel()
                                for p in module.parameters()
                            }.values()
                        )
                        logger.info(
                            f"skipped {module}: {skipped/2**20}M params"
                        )
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(
                            f"bitsandbytes: will optimize {module} in fp32"
                        )
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        if (
            self.args.lr_scheduler_type == "cosine_with_restarts"
            and self.args.num_steps_between_each_restart is not None
        ):
            if self.lr_scheduler is None:
                self.lr_scheduler = (
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=self.args.num_steps_between_each_restart,
                        T_mult=1,
                        eta_min=self.args.lr_min,
                        last_epoch=-1,
                    )
                )
                self._created_lr_scheduler = True
            return self.lr_scheduler

        return super().create_scheduler(num_training_steps, optimizer)

    def _save_checkpoint(self, model, trial, metrics=None):

        if getattr(self.args, "tune_mm_mlp_adapter", False):

            checkpoint_folder = (
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            )

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            modality = getattr(self.args, "modality", "image")
            if modality == "image":
                raise NotImplementedError("Currently, only audio is supported as a modality!")

            elif modality == "audio":
                keys_to_match = ["mm_speech_projector"]
                weight_to_save = get_mm_adapter_state_maybe_zero_3(
                    self.model.named_parameters(), keys_to_match
                )
                # self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                # self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    self.model.config.save_pretrained(output_dir)
                    torch.save(
                        weight_to_save,
                        os.path.join(output_dir, f"mm_speech_projector.bin"),
                    )
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)
            if self.args.local_rank in [-1, 0]:
                checkpoint_folder = (
                    f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                )

                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)

                self.model.config.save_pretrained(output_dir)


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
