#!/usr/bin/env python
import logging
import math
import os
import sys

import datasets
import evaluate
import transformers

from transformers import (
    AutoConfig,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset

logger = logging.getLogger(__name__)


def main():

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('log.txt')],
    )

    # Parametri per il training e la validation, da definire
    training_args = TrainingArguments(
        output_dir="tmp/plorenzi/esperimenti_tesi",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",  # Validation viene fatta alla fine di ogni epoca
        logging_strategy="epoch",  # Logging alla fine di ogni epoca
        save_strategy="epoch",  # Creazione di un checkpoint alla fine di ogni epoca
        gradient_accumulation_steps=8,
        num_train_epochs=40,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        fp16=True,
    )

    # Caricamento del dataset, i piani vanno divisi prima
    raw_datasets = load_dataset(
        "text",
        data_files={
            "train": "logistics_plans_2000.txt",
            "validation": "logistics_plans_2000.txt",
        },
    )

    # Caricamento del tokenizer da file
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="logistics_tokenizer.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    # Qui viene effettuato il caricamento della configurazione del modello.
    # La lunghezza del vocabolario viene fatta coincidere con quella del tokenizer
    # e poi vengono indicati gli id dei token di inizio e fine di una frase.
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        # n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = GPT2LMHeadModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Dimensione del modello GPT-2: {model_size/1000**2:.1f}M parametri")

    model.resize_token_embeddings(len(tokenizer))

    column_names = raw_datasets["train"].column_names
    text_column_name = "text"

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    # Funzione che dati degli esempi restituisce il risultato del tokenizer applicato a tali esempi
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        # Questo errore non dovrebbe succedere nel nostro caso però lo lascio
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Gestione delle label da utilizzare durante l'addestramento.
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("Drunper/metrica_tesi")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Allenamento
    # trainer.train()

    # Salvataggio modello
    # trainer.save_model()


if __name__ == "__main__":
    main()
