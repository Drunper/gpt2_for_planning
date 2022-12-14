import torch
import sys
import os
from dataclasses import dataclass, field

from datasets import load_dataset
from torch.utils.data import DataLoader
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
class ValidationArgs:
    """
    Arguments pertaining to model configuration and validation.
    """

    dataset_file: Optional[str] = field(
        default="20_plans.json",
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
    parser = HfArgumentParser(ValidationArgs)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()

    dataset = load_dataset("json", data_files=args.dataset_file)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = GPT2PRModel.from_pretrained(args.model_path, device_map="auto")

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

    eval_dataloader = DataLoader(
        tokenized_datasets['train'],
        collate_fn=data_collator,
        batch_size=1,
    )

    eval_output = []
    for step, batch in enumerate(eval_dataloader):
        example_output = []
        for i in range(5):
            inputs = batch['input_ids'][:, :batch['actions_idx'] + i + 1]
            with torch.no_grad():
                outputs = model.generate(inputs, do_sample=False, max_length=60)

            decoded_inputs = tokenizer.decode(inputs[0])
            decoded_outputs = tokenizer.decode(outputs[0])
            example_output.append({'input': decoded_inputs, 'output': decoded_outputs})
        eval_output.append(example_output)

            
    write_output_to_file(
        output_dir=args.output_dir, eval_output=eval_output
    )

def write_output_to_file(output_dir=None, eval_output=None):
    file_name = Path(output_dir, "output.txt")
    with open(file_name, "w") as output_file:
        for idx, example_output in enumerate(eval_output):
            output_file.write(f"***** Evaluation on example {idx}  *****\n")
            for evaluation in example_output:
                output_file.write(f"--- input: {evaluation['input']}\n")
                output_file.write(f"--- output: {evaluation['output']}\n")
                output_file.write(f"------------------------------------------\n")


if __name__ == "__main__":
    main()