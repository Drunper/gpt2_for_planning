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
import json
import math
import os
import random
import shutil
import re
import sys
from pathlib import Path
from model import GPT2PModel  # Penso si possa gestire meglio, ma non avevo voglia

import datasets
from dataclasses import dataclass, field
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import argmax
from tqdm.auto import tqdm

import transformers
import evaluate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    AutoConfig,
    PreTrainedTokenizerFast,
    get_scheduler,
)

from typing import List, Optional

logger = get_logger(__name__)


# Definizione delle opzioni
@dataclass
class ModelTrainingArgs:
    """
    Opzioni per la configurazione del modello e del training.
    """

    dataset_dir: Optional[str] = field(
        default="plans",
        metadata={"help": "Cartella contenente il dataset da utilizzare."},
    )
    tokenizer_file: Optional[str] = field(
        default="logistics_tokenizer.json",
        metadata={"help": "File del tokenizer"},
    )
    per_device_train_batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "Batch size (per GPU) da utilizzare durante il training"},
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "Batch size (per GPU) da utilizzare durante l'evaluation"},
    )
    learning_rate: Optional[float] = field(
        default=5e-5,
        metadata={
            "help": "Learning rate iniziale (dopo il periodo di warm-up se utilizzato)."
        },
    )
    weight_decay: Optional[float] = field(
        default=0.0, metadata={"help": "Weight decay."}
    )
    num_train_epochs: Optional[int] = field(
        default=2, metadata={"help": "Numero di epoche di addestramento."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={
            "help": "Numero di step da accumulare prima di fare un backward/update pass."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Numero di esempi del training set da utilizzare durante l'addestramento."}
    )
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Numero di step di addestramento da fare. Se indicato, sovrascrive num_train_epochs."
        },
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear", metadata={"help": "Scheduler per il learning rate da utilizzare."}
    )
    num_warmup_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Numero di passi di warm-up per lo scheduler."},
    )
    output_dir: Optional[str] = field(
        default=None, metadata={"help": "Dove salvare il modello finale."}
    )
    seed: Optional[int] = field(
        default=None, metadata={"help": "Seed per la riproducibilità dei risultati."}
    )
    shuffle_initial_state: Optional[bool] = field(
        default=False, metadata={"help": "Per indicare se è necessario fare lo shuffle dello stato iniziale."}
    )
    checkpointing_steps: Optional[str] = field(
        default=None,
        metadata={
            "help": "Se lo stato deve essere salvato dopo n passi di addestramento oppure 'epoch' per ogni epoca."
        },
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": "Numero di checkpoint massimi."
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Se il training deve ripartire da una cartella di checkpoint."},
    )


def main():
    # Parsing delle opzioni
    parser = HfArgumentParser(ModelTrainingArgs)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()

    # Creazione della cartella di output
    os.makedirs(args.output_dir, exist_ok=True)

    # Ho utilizzato la libreria di HuggingFace, accelerator, per gestire
    # il training multi-GPU. Qui inizializzo il logging e l'uso di
    # tensorboard.
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["log_with"] = "tensorboard"
    accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    # Creazione della cartella contenente l'output dell'evaluation
    # anche se non penso venga più fatta come all'inizio.
    eval_dir = os.path.join(args.output_dir, "eval_output")
    os.makedirs(eval_dir, exist_ok=True)

    # Setup del logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
        ],
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Seed
    if args.seed is not None:
        set_seed(args.seed)

    # Questa riga non fa niente se si fa training con singola o multi-GPU
    # utilizzando un solo nodo.
    accelerator.wait_for_everyone()

    # Caricamento del dataset
    raw_datasets = load_dataset("json", data_dir=args.dataset_dir)

    # Caricamento del tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer_file,
        unk_token="<|unknown|>",
        pad_token="<|pad|>",
        bos_token="<|startofplan|>",
        eos_token="<|endofplan|>",
        mask_token="<|mask|>",
        additional_special_tokens=["<|goals|>", "<|actions|>"],
    )

    # Caricamento del modello
    max_sequence_length = 512
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=max_sequence_length,
        n_positions=max_sequence_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = GPT2PModel(config)
    n_params = sum(t.numel() for t in model.parameters())
    logger.info(
        f"Training GPT2 model from scratch - Total size={n_params/2**20:.2f}M params"
    )

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing del dataset
    if args.shuffle_initial_state:
        column_names = ["name", "initial_state", "goals", "len_initial_state", "len_goals", "len_plan", "pddl_problem_file"]

        def shuffle_initial_state(examples):
            output = []
            for state, goals in zip(examples["initial_state"], examples["goals"]):
                initial_state_fluents = state.split(" ")
                random.shuffle(initial_state_fluents)

                new_state = " ".join(initial_state_fluents) + " <|goals|> " + goals
                output.append(new_state)
            return {"states_shuffled": output}

        def tokenize_function(examples):
            return tokenizer(
                examples["states_shuffled"],
                examples["actions"],
                return_token_type_ids=False,
                # max_length=max_length,
                # padding='max_length',
            )

        with accelerator.main_process_first():
            pre_processed_datasets = raw_datasets.map(
                shuffle_initial_state,
                batched=True,
                remove_columns=column_names,
                desc="Shuffling initial state fluents for every example of the dataset"
            )

            tokenized_datasets = pre_processed_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=["states_shuffled", "actions"],
                desc="Running tokenizer on dataset",
            )

    else:
        column_names = ["name", "initial_state", "actions", "goals", "len_initial_state", "len_goals", "len_plan", "pddl_problem_file"]

        def prepare_input_for_training(examples):
            output = []
            for state, goals in zip(examples["initial_state"], examples["goals"]):
                new_state = state + " <|goals|> " + goals
                output.append(new_state)
            return {"states": output}

        def tokenize_function(examples):
            return tokenizer(
                examples["states"],
                examples["actions"],
                return_token_type_ids=False,
            )

        with accelerator.main_process_first():
            pre_processed_datasets = raw_datasets.map(
                prepare_input_for_training,
                batched=True,
                remove_columns=["initial_state"],
                desc="Concatenation of initial state and goals"
            )

            tokenized_datasets = pre_processed_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on dataset",
            )

    train_dataset = tokenized_datasets["train"]
    if args.max_train_samples:
        max_train_samples = min(args.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(max_train_samples))
    eval_dataset = tokenized_datasets["validation"]

    # Loggo alcuni esempi del training set, per vedere se va tutto bene
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Creo i dataloader per il training (faccio lo shuffle degli esempi) e per la validation
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

    # Optimizer: Adam
    # Questa parte l'ho copiata (come quasi tutto il codice) da uno script di training
    # presente su HuggingFace. Modificando la creazione dell'optimizer sotto si potrebbe
    # anche togliere.
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

    # Una metrica molto semplice che ho creato. Basta andare su
    # https://huggingface.co/spaces/Drunper/metrica_tesi/blob/main/metrica_tesi.py
    # per vedere come funziona il codice
    metric = evaluate.load("Drunper/metrica_tesi")

    # Calcolo del numero di step di training
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

    # Preparo tutto utilizzando accelerator
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Ricacoli vari del numero di training step
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Ricalcolo del numero di epoche
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Ogni quanto salvare lo stato
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    experiment_config = vars(args)
    # TensorBoard cannot log Enums, need the raw value
    accelerator.init_trackers("tensorboard_dir", experiment_config)

    # Inizia l'addestramento
    total_batch_size = (
        args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
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

    # Controllo se ci sono checkpoint precedenti e in caso riparto da lì
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Prendo il checkpoint più recente
            dirs = [
                f.name for f in os.scandir(os.getcwd()) if f.is_dir()
            ]  # Modifica questo per settare la cartella di checkpoint
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Ordina in base alla data di ultima modifica
        # Estraggo `epoch_{i}` o `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = (
                int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # Aggiorno la progress bar in caso
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # Gestione degli step in caso parta da un checkpoint
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # Tengo traccia della loss ad ogni epoca
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Controlli vari per accelator
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
        # eval_output = []
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
                example_logits = logits[i, batch['actions_idx'][i].item():batch['eop_idx'][i].item()]
                for j in range(example_logits.shape[0]):
                    prediction = argmax(example_logits[j])
                    reference = batch['input_ids'][i][batch['actions_idx'][i].item() + j + 1]
                    metric.add(references=reference, predictions=prediction, actions_seen=j)

        # Vecchio loop di evaluation, penso sia necessario fare delle modifiche in caso serva utilizzarlo
        #     logits = outputs.logits
        #     for i in range(logits.shape[0]):
        #         example_logits = logits[
        #             i, batch["actions_idx"][i].item():batch["eop_idx"][i].item()
        #         ]
        #         example_output = dict()
        #         example_output["states"] = tokenizer.decode(batch["input_ids"][i][:batch["actions_idx"][i].item() + 1])
        #         evaluations = []
        #         for j in range(example_logits.shape[0]):
        #             argmax_output = argmax(example_logits[j])
        #             pred_token = tokenizer.decode(argmax_output)
        #             seen_actions = tokenizer.decode(
        #                 batch["input_ids"][i][batch["actions_idx"][i].item() + 1:batch["actions_idx"][i].item() + j + 1]
        #             ) if j != 0 else ""
        #             real_token = tokenizer.decode(
        #                 batch["input_ids"][i][batch["actions_idx"][i].item() + j + 1]
        #             )
        #             token_output = {
        #                 "seen_actions": seen_actions,
        #                 "pred_token": pred_token,
        #                 "real_token": real_token,
        #                 "match": pred_token == real_token,
        #             }
        #             evaluations.append(token_output)
        #         example_output["evaluations"] = evaluations
        #         eval_output.append(example_output)
        # write_eval_to_file(
        #     output_dir=eval_dir, eval_output=eval_output, epoch=epoch
        # )
        logger.info(f"Evaluation output saved in {args.output_dir}/eval_output/eval_epoch_{epoch}.txt")

        losses = torch.cat(losses)
        eval_loss = torch.mean(losses).item()
        train_loss = total_loss.item() / len(train_dataloader)

        metric_results = metric.compute()
        metric_results['train_loss'] = train_loss
        metric_results['validation_loss'] = eval_loss

        logger.info(f"epoch {epoch}: train_loss: {train_loss}, validation_loss: {eval_loss}")

        accelerator.log(
            metric_results,
            step=epoch + 1,
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
            # Verifico se devo creare un nuovo checkpoint oppure sovrascrivere
            rotate_checkpoints(
                output_dir=args.output_dir, save_total_limit=args.save_total_limit
            )

    accelerator.end_training()

    logger.info("Training finished successfully!")
    logger.info("Saving model and tokenizer...")

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

    logger.info("Model and tokenizer saved successfully!")


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

    # Controllo se bisogna cancellare i vecchi checkpoint
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


# Per il vecchio loop di evaluation
def write_eval_to_file(output_dir=None, eval_output=None, epoch=None):
    file_name = Path(output_dir, f"eval_epoch_{epoch}.txt")
    with open(file_name, "w") as output_file:
        output_file.write(f"***** Evaluation loop for epoch {epoch}  *****\n")
        for idx, example_output in enumerate(eval_output):
            output_file.write(f"***** Evaluation on example {idx}  *****\n")
            output_file.write(f"--- states: {example_output['states']}\n")
            output_file.write(f"------------------------------------------\n")
            for token_output in example_output["evaluations"]:
                output_file.write(f"--- seen_actions: {token_output['seen_actions']}\n")
                output_file.write(f"--- pred_token: {token_output['pred_token']}\n")
                output_file.write(f"--- real_token: {token_output['real_token']}\n")
                output_file.write(f"--- match: {token_output['match']}\n")
                output_file.write(f"------------------------------------------\n")

    file_name = Path(output_dir, f"eval_epoch_{epoch}.json")
    with open(file_name, "w", encoding="UTF8") as output_file:
        json_str = json.dumps(eval_output, indent=4)
        output_file.write(json_str)


if __name__ == "__main__":
    main()
