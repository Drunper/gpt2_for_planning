import sys
import os
import json
import logging
import random
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
        default="input", metadata={"help": "Output directory"}
    )
    max_length: Optional[int] = field(
        default=60,
        metadata={"help": "Plan max length"},
    )
    actions_seen: Optional[int] = field(
        default=0,
        metadata={"help": "Number of actions to add to the input"}
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
        default="generation.log",
        metadata={"help": "Log file name"}
    )
    batch_size: Optional[int] = field(
        default=4,
        metadata={"help": "Batch size that will be used during generation"}
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
        }
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

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, args.log_file_name)),
        ],
    )

    dataset = load_dataset("json", data_files=args.dataset_file)
    logger.info("Dataset loaded successfully")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = GPT2PRModel.from_pretrained(args.model_path, device_map="auto")
    logger.info("Model loaded successfully")

    def get_inputs_for_generation(examples):
        output = []
        for state, actions in zip(examples["states"], examples["actions"]):
            example = state + " <|actions|>"
            action_list = actions.split(" ")
            action_string = " ".join(action_list[:args.actions_seen])
            if action_string != "":
                example = example + " " + action_string
            output.append(example)
        return {"input": output}

    def tokenize_function(examples):
        return tokenizer(
            examples["input"],
            return_token_type_ids=False,
            # max_length=max_length,
            # padding='max_length',
        )

    column_names = ["name", "states", "actions"]

    pre_processed_dataset = dataset.map(
            get_inputs_for_generation,
            batched=True,
            remove_columns=column_names,
            desc="Running input pre-processing on dataset",
    )

    tokenized_datasets = pre_processed_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['input'],
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
    
    actions_token_id = tokenizer.convert_tokens_to_ids("<|actions|>")
    for step, batch in enumerate(test_dataloader):
        if not (step % args.save_after):
            logger.info(f"Processed {step * args.batch_size} plans of {len(test_dataset)}")
            eval_output = []
        example_output = []

        problem_ids_list = []
        for i in range(batch['input_ids'].shape[0]):
            problem_id = dataset["train"][step * args.batch_size + i]["name"].split("-")[-1].split("_")[0]
            problem_ids_list.append(problem_id)
            logger.info(f"Sample {i} of batch {step} of the test set: {batch['input_ids'][i]}.")

        inputs = batch["input_ids"]
        inputs = inputs.to("cuda")

        outputs = model.generate(
            inputs,
            do_sample=False,
            max_new_tokens=args.max_length,
            pad_token_id=tokenizer.pad_token_id,
            problem_ids_list=problem_ids_list,
            tokenizer=tokenizer,
            pddl_dir=args.pddl_dir,
            pddl_domain_file=args.pddl_domain_file,
            actions_token_id=actions_token_id
        )
           
        for i in range(batch["input_ids"].shape[0]):
            generated_plan = outputs[i]
            if generated_plan[-1] == tokenizer.eos_token_id:
                generated_plan = generated_plan[:-1]
            example_output.append(
                {
                    "input": tokenizer.decode(batch["input_ids"][i]),
                    "plan": tokenizer.decode(generated_plan),
                    "actions_seen": args.actions_seen,
                    "problem_id": problem_ids_list[i],
                }
            )

        eval_output.append(example_output)
        q, r = divmod(step, args.save_after)
        if r == (args.save_after - 1):
            bounds = (q * args.save_after * args.batch_size, (step + 1) * args.batch_size - 1)
            write_output_to_file(output_dir=args.output_dir, eval_output=eval_output, bounds=bounds)


    logger.info("All plans have been processed")
    logger.info("Writing output to file")
    if bounds[1] + 1 <= len(test_dataset) - 1:
        bounds = (bounds[1] + 1, len(test_dataset) - 1)
        write_output_to_file(output_dir=args.output_dir, eval_output=eval_output, bounds=bounds)
    logger.info("Output file written successfully")


def write_output_to_file(output_dir=None, eval_output=None, bounds=None):
    txt_path = Path(output_dir, f"output_{bounds[0]}_{bounds[1]}.txt")
    with open(txt_path, "w") as output_file:
        for idx, example_output in enumerate(eval_output):
            output_file.write(f"***** Evaluation on example {idx}  *****\n")
            for evaluation in example_output:
                output_file.write(f"--- input: {evaluation['input']}\n")
                output_file.write(f"--- actions_seen: {evaluation['actions_seen']}\n")
                output_file.write(f"--- generated_plan: {evaluation['plan']}\n")
                output_file.write(f"------------------------------------------\n")

    json_path = Path(output_dir, f"to_validate_{bounds[0]}_{bounds[1]}.json")
    output = []
    for example_output in eval_output:
        for evaluation in example_output:
            to_save = {
                "problem_id": evaluation["problem_id"],
                "actions_seen": evaluation["actions_seen"],
                "plan": evaluation["plan"],
            }
            output.append(to_save)

    with open(json_path, "w") as output_file:
        json.dump(output, output_file)


if __name__ == "__main__":
    main()
