import torch
import sys
import os
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
    model = GPT2PRModel.from_pretrained(args.model_path)
    metric = evaluate.load("Drunper/metrica_tesi")

    def tokenize_function(examples):
        return tokenizer(
            examples["states"],
            examples["actions"],
            return_token_type_ids=False,
            # max_length=max_length,
            # padding='max_length',
        )

    column_names = ["name", "states", "actions"]

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
        batch_size=1,
    )

    losses = []
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss)
        logits = outputs.logits
        for i in range(logits.shape[0]):
            example_logits = logits[i, batch['actions_idx'][i].item():batch['eop_idx'][i].item()]
            for j in range(example_logits.shape[0]):
                prediction = argmax(example_logits[j])
                reference = batch['input_ids'][i][batch['actions_idx'][i].item() + j + 1]
                metric.add(references=reference, predictions=prediction, actions_seen=j)

    losses = torch.cat(losses)
    test_loss = torch.mean(losses).item()

    metric_results = metric.compute()
    metric_results['test_loss'] = test_loss

    print(metric_results)

    output_path = Path(args.output_dir, "results.txt")
    with open(output_path, "w") as output_file:
        for key, value in metric_results:
            output_file.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()