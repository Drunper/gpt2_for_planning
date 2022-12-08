#!/usr/bin/env python
# coding=utf-8
#
# Copyright 2022 Patrick Lorenzi
#
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified to train a specific model (GPT-2) on a
# specific dataset, with a custom evaluation loop.

import logging
import math
import os
import random
import shutil
import re
import sys
from pathlib import Path
from model import GPT2PRModel

import datasets
from dataclasses import dataclass, field
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn import Softmax
from torch import argmax
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    AutoConfig,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    get_scheduler,
)

from typing import List, Optional

logger = get_logger(__name__)


@dataclass
class ModelTrainingArgs:
    """
    Arguments pertaining to model configuration and training.
    """

    dataset_dir: Optional[str] = field(
        default="plans",
        metadata={"help": "Path to directory containing json files of dataset"},
    )
    tokenizer_file: Optional[str] = field(
        default="logistics_tokenizer.json",
        metadata={"help": "Path to tokenizer json file"},
    )
    per_device_train_batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "Batch size (per device) for the training dataloader."},
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "Batch size (per device) for the evaluation dataloader."},
    )
    learning_rate: Optional[float] = field(
        default=5e-5,
        metadata={
            "help": "Initial learning rate (after the potential warmup period) to use."
        },
    )
    weight_decay: Optional[float] = field(
        default=0.0, metadata={"help": "Weight decay to use."}
    )
    num_train_epochs: Optional[int] = field(
        default=2, metadata={"help": "Total number of training epochs to perform."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Total number of training steps to perform. If provided, overrides num_train_epochs."
        },
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear", metadata={"help": "The scheduler type to use."}
    )
    num_warmup_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of steps for the warmup in the lr scheduler."},
    )
    output_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store the final model."}
    )
    seed: Optional[int] = field(
        default=None, metadata={"help": "A seed for reproducible training."}
    )
    checkpointing_steps: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
        },
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": "Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "If the training should continue from a checkpoint folder."},
    )


def main():
    parser = HfArgumentParser(ModelTrainingArgs)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()


    # Create output folder
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["log_with"] = "tensorboard"
    accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_dir + "/log.txt"),
        ],
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    # Load dataset from dataset_dir. Default folder is plans, so relative to script execution path
    raw_datasets = load_dataset("json", data_dir=args.dataset_dir)

    # Load tokenizer from file
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer_file,
        unk_token="<|unknown|>",
        pad_token="<|pad|>",
        bos_token="<|startofplan|>",
        eos_token="<|endofplan|>",
        mask_token="<|mask|>",
        additional_special_tokens=["<|goals|>", "<|actions|>"],
    )

    # Load model
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    max_length = 512

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=max_length,  # ??
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = GPT2PRModel(config)
    n_params = sum(t.numel() for t in model.parameters())
    logger.info(
        f"Training GPT2 model from scratch - Total size={n_params/2**20:.2f}M params"
    )

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the plans.

    column_names = ["name", "states", "actions"]

    def tokenize_function(examples):
        return tokenizer(
            examples["states"],
            examples["actions"],
            return_token_type_ids=False,
            # max_length=max_length,
            # padding='max_length',
        )

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    experiment_config = vars(args)
    # TensorBoard cannot log Enums, need the raw value
    accelerator.init_trackers("gptpr_training", experiment_config)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [
                f.name for f in os.scandir(os.getcwd()) if f.is_dir()
            ]  # Modify this to set checkpoint folder
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            # if isinstance(checkpointing_steps, int):
            #     if completed_steps % checkpointing_steps == 0:
            #         output_dir = f"step_{completed_steps }"
            #         if args.output_dir is not None:
            #             output_dir = os.path.join(args.output_dir, output_dir)
            #         accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

        logger.info(f"***** Running evaluation for epoch {epoch} *****")
        model.eval()
        losses = []
        softmax = Softmax(dim=0)
        eval_output = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(
                accelerator.gather_for_metrics(
                    loss.repeat(args.per_device_eval_batch_size)
                )
            )
            logits = outputs.logits
            for i in range(logits.shape[0]):
                example_logits = logits[
                    i, batch["actions_idx"][i].item():batch["eoa_idx"][i].item()
                ]
                example_output = []
                for j in range(example_logits.shape[0]):
                    softmax_output = softmax(example_logits[j])
                    argmax_output = argmax(softmax_output)
                    pred_token = tokenizer.decode(argmax_output)
                    context = tokenizer.decode(
                        batch["input_ids"][i][:batch["actions_idx"][i].item() + j]
                    )
                    real_token = tokenizer.decode(
                        batch["input_ids"][i][batch["actions_idx"][i].item() + j]
                    )
                    token_output = {
                        "context": context,
                        "pred_token": pred_token,
                        "real_token": real_token,
                    }
                    example_output.append(token_output)
                eval_output.append(example_output)
        write_eval_to_file(
            output_dir=args.output_dir, eval_output=eval_output, epoch=epoch
        )

        losses = torch.cat(losses)
        eval_loss = torch.mean(losses)

        logger.info(f"epoch {epoch}: eval_loss: {eval_loss}")

        accelerator.log(
                {
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            # Check if we should delete older checkpoints
            rotate_checkpoints(
                output_dir=args.output_dir, save_total_limit=args.save_total_limit
            )

    accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)


def sorted_checkpoints(
    output_dir=None, checkpoint_prefix="epoch", use_mtime=False
) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = [
        str(x)
        for x in Path(output_dir).glob(f"{checkpoint_prefix}_*")
        if os.path.isdir(x)
    ]

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(f".*{checkpoint_prefix}_([0-9]+)", path)
            if regex_match is not None and regex_match.groups() is not None:
                ordering_and_checkpoint_path.append(
                    (int(regex_match.groups()[0]), path)
                )

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(use_mtime=False, output_dir=None, save_total_limit=None) -> None:
    if save_total_limit is None or save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
        )
        shutil.rmtree(checkpoint)


def write_eval_to_file(output_dir=None, eval_output=None, epoch=None):
    file_name = Path(output_dir, f"eval_epoch_{epoch}.txt")
    with open(file_name, "w") as output_file:
        output_file.write(f"***** Evaluation loop for epoch {epoch}  *****\n")
        for idx, example_output in enumerate(eval_output):
            output_file.write(f"***** Evaluation on example {idx}  *****\n")
            for token_output in example_output:
                output_file.write(f"--- context: {token_output['context']} \n")
                output_file.write(f"--- pred_token: {token_output['pred_token']} \n")
                output_file.write(f"--- real_token: {token_output['real_token']} \n")
                output_file.write(f"------------------------------------------\n")


if __name__ == "__main__":
    main()
