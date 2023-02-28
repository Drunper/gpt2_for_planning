import torch
import sys
import os
import random
from dataclasses import dataclass, field

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import argmax
from pathlib import Path
from model import GPT2PRModel


import transformers
import evaluate
from transformers import (
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    AutoConfig,
    PreTrainedTokenizerFast,
    AutoTokenizer,
)

from typing import List, Optional

@dataclass
class TestArgs:
    """
    Arguments pertaining to model configuration and validation.
    """

    dataset_file: Optional[str] = field(
        default="plans/logistics_plans_test_0.json",
        metadata={"help": "Path to file containing json files of validation set"},
    )
    tokenizer_path: Optional[str] = field(
        default="logistics_tokenizer.json",
        metadata={"help": "Path to tokenizer json file"},
    )
    model_path: Optional[str] = field(
        default="",
        metadata={"help": "Path to model"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "Output directory"}
    )
    shuffle_initial_state: Optional[bool] = field(
        default=False, metadata={"help": "Whether to shuffle fluents of initial states for every example."}
    )
    seed: Optional[int] = field(
        default=7, metadata={"help": "A seed for reproducible testing."}
    )

def main():
    parser = HfArgumentParser(TestArgs)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()

    dataset = load_dataset("json", data_files=args.dataset_file)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = GPT2PRModel.from_pretrained(args.model_path, device_map="auto")
    metric = evaluate.load("Drunper/metrica_tesi")

    random.seed(args.seed)

    if args.shuffle_initial_state:
        column_names = ["name", "states"]

        def shuffle_initial_state(examples):
            output = []
            for state in examples["states"]:
                initial_state_fluents, goals = state.split(" <|goals|> ")
                initial_state_fluents = initial_state_fluents.split(" ")
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

        pre_processed_datasets = dataset.map(
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
        column_names = ["name", "states", "actions"]
        def tokenize_function(examples):
            return tokenizer(
                examples["states"],
                examples["actions"],
                return_token_type_ids=False,
                # max_length=max_length,
                # padding='max_length',
            )

        tokenized_datasets = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on dataset",
            )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    test_dataloader = DataLoader(
        tokenized_datasets['train'],
        collate_fn=data_collator,
        batch_size=16,
    )

    total_loss = 0.0
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to('cuda')
            labels = batch['labels'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            actions_idx = batch['actions_idx'].to('cuda')
            eop_idx = batch['eop_idx'].to('cuda')

            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask, actions_idx=actions_idx, eop_idx=eop_idx)

        loss = outputs.loss
        total_loss += loss.detach().float()
        logits = outputs.logits
        for i in range(logits.shape[0]):
            example_logits = logits[i, batch['actions_idx'][i].item():batch['eop_idx'][i].item()]
            for j in range(example_logits.shape[0]):
                prediction = argmax(example_logits[j])
                reference = batch['input_ids'][i][batch['actions_idx'][i].item() + j + 1]
                metric.add(references=reference, predictions=prediction, actions_seen=j)

    test_loss = total_loss / len(test_dataloader)

    metric_results = metric.compute()
    metric_results['test_loss'] = test_loss.item()

    print(metric_results)

    output_path = Path(args.output_dir, "test_results.txt")
    with open(output_path, "w") as output_file:
        for key, value in metric_results.items():
            output_file.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()