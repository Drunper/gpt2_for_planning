import torch
import sys
import os
import json
import logging
import random
from tqdm import tqdm
from dataclasses import dataclass, field

from datasets import load_dataset
from torch.utils.data import DataLoader
from pathlib import Path
from model import GPT2PRModel

from transformers import (
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
)

from typing import Optional


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
        default="tokenizer_generation",
        metadata={"help": "Path to tokenizer json file"},
    )
    model_path: Optional[str] = field(
        default="",
        metadata={"help": "Path to model"},
    )
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "Output directory"}
    )
    max_length: Optional[int] = field(
        default=60,
        metadata={"help": "Plan max length"},
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Number of beams for beam search. Default value is 1, which means greedy generation"}
    )
    actions_seen: Optional[int] = field(
        default=0, metadata={"help": "Number of actions to add to the input"}
    )
    pddl_dir: Optional[str] = field(
        default="pddl",
        metadata={"help": "Path to folder containing pddl file"},
    )
    pddl_domain_file: Optional[str] = field(
        default="domain.pddl",
        metadata={"help": "Name of file containing domain definition"},
    )
    log_file_name: Optional[str] = field(
        default="generation.log", metadata={"help": "Log file name"}
    )
    batch_size: Optional[int] = field(
        default=4, metadata={"help": "Batch size that will be used during generation"}
    )
    save_after: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "After how many processed batch you want to save the output to file "
                "e.g. if batch_size is set to four and save_after is set to 10, then "
                "after every 10 batch the output will be save, meaning that the output of 40 samples "
                "will be saved."
            )
        },
    )
    shuffle_initial_state: Optional[bool] = field(
        default=False, metadata={"help": "Whether to shuffle fluents of initial states for every example."}
    )
    seed: Optional[int] = field(
        default=7, metadata={"help": "A seed for reproducible testing."}
    )


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(ValidationArgs)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()

    plan_output_dir = Path(args.output_dir, f"{args.actions_seen}_actions")
    plan_output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = plan_output_dir.joinpath(args.log_file_name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file_path),
        ],
    )

    dataset = load_dataset("json", data_files=args.dataset_file)
    logger.info("Dataset loaded successfully")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = GPT2PRModel.from_pretrained(args.model_path, device_map="auto")

    random.seed(args.seed)

    def tokenize_function(examples):
        return tokenizer(
            examples["input"],
            return_token_type_ids=False,
            # max_length=max_length,
            # padding='max_length',
        )

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
        
        def get_inputs_for_generation(examples):
            output = []
            for state, actions in zip(examples["states_shuffled"], examples["actions"]):
                example = state + " <|actions|>"
                action_list = actions.split(" ")
                action_string = " ".join(action_list[: args.actions_seen])
                if action_string != "":
                    example = example + " " + action_string
                output.append(example)
            return {"input": output}

        pre_processed_datasets = dataset.map(
                shuffle_initial_state,
                batched=True,
                remove_columns=column_names,
                desc="Shuffling initial state fluents for every example of the dataset"
            )
        
        pre_processed_dataset = pre_processed_datasets.map(
            get_inputs_for_generation,
            batched=True,
            remove_columns=["states_shuffled", "actions"],
            desc="Running input pre-processing on dataset",
        )
    else:
        column_names = ["name", "states", "actions"]

        def get_inputs_for_generation(examples):
            output = []
            for state, actions in zip(examples["states"], examples["actions"]):
                example = state + " <|actions|>"
                action_list = actions.split(" ")
                action_string = " ".join(action_list[: args.actions_seen])
                if action_string != "":
                    example = example + " " + action_string
                output.append(example)
            return {"input": output}
        
        pre_processed_dataset = dataset.map(
            get_inputs_for_generation,
            batched=True,
            remove_columns=column_names,
            desc="Running input pre-processing on dataset",
        )

    tokenized_datasets = pre_processed_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["input"],
        desc="Running tokenizer on dataset",
    )

    test_dataset = tokenized_datasets["train"]
    logger.info("You can safely ignore the warning above ^^")

    for index in random.sample(range(len(test_dataset)), 3):
        logger.info(f"Sample {index} of the test set: {test_dataset[index]}.")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )

    logger.info("Starting generation of plans")
    logger.info(f"Dataset size: {len(tokenized_datasets['train'])}")

    bounds = (0, 0)

    actions_token_id = tokenizer.convert_tokens_to_ids("<|actions|>")
    generation_output = []
    for step, batch in enumerate(tqdm(test_dataloader)):
        problem_ids_list = []
        example_ids_list = []
        for i in range(batch["input_ids"].shape[0]):
            instance = dataset["train"][step * args.batch_size + i]["name"].split("-")[
                -1
            ]
            problem_id = instance.split("_")[0]
            example_id = instance.split(".")[0]
            problem_ids_list.append(problem_id)
            example_ids_list.append(example_id)

        inputs = batch["input_ids"]
        inputs = inputs.to("cuda:0")
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                num_beams=args.num_beams,
                do_sample=False,
                max_new_tokens=args.max_length,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i in range(batch["input_ids"].shape[0]):
            generated_plan = outputs[i]
            if generated_plan[-1] == tokenizer.eos_token_id:
                generated_plan = generated_plan[:-1]
            else:
                eop_idx = (generated_plan == tokenizer.eos_token_id).nonzero(
                    as_tuple=True
                )[0]
                if eop_idx.shape[0]:
                    generated_plan = generated_plan[:eop_idx]

            actions_idx = (generated_plan == actions_token_id).nonzero(as_tuple=True)[0]
            generated_plan = generated_plan[actions_idx + 1:]

            sop_idx = (batch["input_ids"][i] == tokenizer.bos_token_id).nonzero(
                as_tuple=True
            )[0]
            input_to_decode = batch["input_ids"][i, sop_idx:]

            generation_output.append(
                {
                    "input": tokenizer.decode(input_to_decode),
                    "plan": tokenizer.decode(generated_plan),
                    "actions_seen": args.actions_seen,
                    "problem_id": problem_ids_list[i],
                    "example_id": example_ids_list[i],
                }
            )

        q, r = divmod(step, args.save_after)
        if r == (args.save_after - 1):
            if (q + 1) * args.batch_size - 1 <= len(test_dataset) - 1:
                bounds = (
                    q * args.save_after * args.batch_size,
                    (step + 1) * args.batch_size - 1,
                )
                logger.info(
                    f"Generated {(step+1) * args.batch_size} plans of {len(test_dataset)}"
                )
                write_output_to_file(
                    output_dir=plan_output_dir,
                    generation_output=generation_output,
                    bounds=bounds,
                )
                generation_output = []

    logger.info("All plans have been generated")
    if bounds[1] + 1 <= len(test_dataset) - 1:
        bounds = (bounds[1] + 1, len(test_dataset) - 1)
        write_output_to_file(
            output_dir=plan_output_dir,
            generation_output=generation_output,
            bounds=bounds,
        )


def write_output_to_file(output_dir=None, generation_output=None, bounds=None):
    txt_path = Path(output_dir, f"output_{bounds[0]}_{bounds[1]}.txt")
    json_path = Path(output_dir, f"to_validate_{bounds[0]}_{bounds[1]}.json")
    logger.info(f"Writing outputs to files {txt_path} and {json_path}")
    with open(txt_path, "w") as output_file:
        for idx, example_output in enumerate(generation_output):
            output_file.write(f"***** Evaluation on example {idx}  *****\n")
            output_file.write(f"--- input: {example_output['input']}\n")
            output_file.write(f"--- actions_seen: {example_output['actions_seen']}\n")
            output_file.write(f"--- generated_plan: {example_output['plan']}\n")
            output_file.write(f"--- example: {example_output['example_id']}\n")

    with open(json_path, "w") as output_file:
        json.dump(generation_output, output_file)
    logger.info("Output files written successfully")


if __name__ == "__main__":
    main()
