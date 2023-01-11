import torch
import sys
import os
import json
import re
from dataclasses import dataclass, field

from unified_planning.engines.compilers.grounder import Grounder
from unified_planning.io.pddl_reader import PDDLReader
from unified_planning.engines import SequentialSimulator
from unified_planning.model import UPCOWState
from unified_planning.shortcuts import *

from datasets import load_dataset
from torch.utils.data import DataLoader
from pathlib import Path
from model import GPT2PRModel

from transformers import (
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from typing import List, Optional, Union, cast


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
        default="input", metadata={"help": "Output directory"}
    )
    max_length: Optional[int] = field(
        default=60,
        metadata={"help": "Plan max length"},
    )
    pddl_dir: Optional[str] = field(
        default="pddl",
        metadata={"help": "Path to folder containing pddl file"},
    )
    pddl_domain_file: Optional[str] = field(
        default="domain.pddl",
        metadata={"help": "Name of file containing domain definition"},
    )


reader = PDDLReader()
grounder = Grounder()


def main():
    parser = HfArgumentParser(ValidationArgs)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()

    os.makedirs(args.output_dir, exist_ok=True)

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
        tokenized_datasets["train"],
        collate_fn=data_collator,
        batch_size=1,
    )

    eval_output = []
    token_ids = set(range(len(tokenizer)))
    for step, batch in enumerate(eval_dataloader):
        example_output = []
        real_plan = tokenizer.decode(
            batch["input_ids"][0, batch["actions_idx"] + 1: batch["eop_idx"].item()]
        )
        problem_id = dataset["train"][step]["name"].split("-")[-1].split("_")[0]
        problem, initial_state, simulator = get_simulation_tools(
            args.pddl_dir, args.pddl_domain_file, problem_id
        )

        n_actions = batch["eop_idx"].item() - batch["actions_idx"].item() - 1
        for i in range(min(6, n_actions)):
            input_to_decode = inputs = batch["input_ids"][
                :, : batch["actions_idx"] + i + 1
            ]
            inputs = inputs.to("cuda")
            state = initial_state

            # Devo partire dallo stato corretto, corrispondente
            # all'applicazione di tutte le azioni presenti nell'input
            for k in range(i):
                possible_actions_ids_dict = get_possible_actions_ids(
                    problem, state, simulator, tokenizer
                )
                action = format_action(tokenizer.decode(batch["input_ids"][:, batch["actions_idx"] + k + 1][0]))
                action = get_action_by_name(possible_actions_ids_dict, action)
                state = apply_action_to_state(action, state, simulator)

            possible_actions_ids_dict = get_possible_actions_ids(
                problem, state, simulator, tokenizer
            )

            generated_eop = False
            j = i
            while j < args.max_length and not generated_eop:
                output = model.generate(
                    inputs,
                    do_sample=False,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                # Prendo solo le logits relative alle azioni possibili e scelgo quella con probabilità più alta
                possible_actions_ids = set(possible_actions_ids_dict.keys())
                possible_actions_ids.add(tokenizer.eos_token_id)
                logits_mask = torch.tensor(list(token_ids - possible_actions_ids)).to("cuda")
                logits = output.scores[0]
                logits[:, logits_mask] = float("-Inf")
                generated_token = torch.argmax(logits, dim=-1)

                # if i == 2:
                #     print(f"Allora: i vale {i}, j vale {j}")
                #     print("Gli id possibili sono:")
                #     print(possible_actions_ids)
                #     print(f"Il token generato è {generated_token.item()}")

                # Se genero <endofplan> allora ho finito
                if generated_token.item() == tokenizer.eos_token_id:
                    generated_eop = True

                # Altrimenti uso la sequenza di output ottenuta come input per generare la prossima azione,
                # aggiorno lo stato andando ad applicare l'azione corrispondente al token generato e
                # calcolo il nuovo dizionario dei nome_azione-token_id
                if not generated_eop:
                    j += 1
                    # tmp = torch.zeros((1, inputs.shape[1] + 1), dtype=inputs.dtype, layout=inputs.layout, device=inputs.device)
                    # tmp[0, :inputs.shape[1]] = inputs
                    # tmp[0, -1] = generated_token
                    # inputs = tmp
                    inputs = torch.cat([inputs, generated_token[:, None]], dim=-1)
                    state = apply_action_to_state(
                        possible_actions_ids_dict[generated_token.item()],
                        state,
                        simulator
                    )
                    possible_actions_ids_dict = get_possible_actions_ids(
                        problem, state, simulator, tokenizer
                    )

            # L'input da includer nei file di output è quello passato inizialmente
            decoded_inputs = tokenizer.decode(input_to_decode[0])
            if generated_eop:
                output_to_decode = output.sequences[0][batch["actions_idx"] + 1:-1]
            else:
                output_to_decode = inputs[0]

            # Se ho generato il token di fine piano lo rimuovo
            if output_to_decode[-1] == tokenizer.eos_token_id:
                output_to_decode = output_to_decode[:-1]
            decoded_outputs = tokenizer.decode(output_to_decode)
            example_output.append(
                {
                    "input": decoded_inputs,
                    "plan": decoded_outputs,
                    "real_plan": real_plan,
                    "actions_seen": i,
                    "problem_id": problem_id,
                }
            )
        eval_output.append(example_output)

    write_output_to_file(output_dir=args.output_dir, eval_output=eval_output)


def write_output_to_file(output_dir=None, eval_output=None):
    txt_path = Path(output_dir, "output.txt")
    with open(txt_path, "w") as output_file:
        for idx, example_output in enumerate(eval_output):
            output_file.write(f"***** Evaluation on example {idx}  *****\n")
            for evaluation in example_output:
                output_file.write(f"--- input: {evaluation['input']}\n")
                output_file.write(f"--- actions_seen: {evaluation['actions_seen']}\n")
                output_file.write(f"--- generated_plan: {evaluation['plan']}\n")
                output_file.write(f"--- real_plan: {evaluation['real_plan']}\n")
                output_file.write(f"------------------------------------------\n")

    json_path = Path(output_dir, "to_validate.json")
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


def get_simulation_tools(pddl_dir, pddl_domain_file, problem_id):
    problem = reader.parse_problem(
        pddl_dir + os.sep + pddl_domain_file, pddl_dir + os.sep + problem_id + '.pddl'
    )
    problem = grounder.compile(problem).problem
    init_state = UPCOWState(problem.initial_values)
    simulator = SequentialSimulator(problem)
    return problem, init_state, simulator


def get_possible_actions(problem, state, simulator):
    events = simulator.get_applicable_events(state)
    events = list(events)

    possible_actions = []
    for ev in events:
        for ac in problem.actions:
            if ac.preconditions == ev.conditions and ac.effects == ev.effects:
                possible_actions.append(ac)
    return possible_actions


def is_action_applicable(action_name, possible_actions):
    for action in possible_actions:
        if action.name == action_name:
            return True
    return False


def get_action_by_name(possible_actions_ids_dict, action_name):
    possible_actions = list(possible_actions_ids_dict.values())
    for action in possible_actions:
        if action.name == action_name:
            return action
    return None


def apply_action_to_state(action, state, simulator):
    event = list(simulator.get_events(action, []))[0]
    next_state = cast(UPCOWState, simulator.apply(event, state))
    return next_state


def format_action(action_name):
    if action_name.startswith("drivetruck"):
        tmp = re.sub("drivetruck", "DRIVE-TRUCK ", action_name)
    elif action_name.startswith("loadtruck"):
        tmp = re.sub("loadtruck", "LOAD-TRUCK ", action_name)
    elif action_name.startswith("unloadtruck"):
        tmp = re.sub("unloadtruck", "UNLOAD-TRUCK ", action_name)
    elif action_name.startswith("loadairplane"):
        tmp = re.sub("loadairplane", "LOAD-AIRPLANE ", action_name)
    elif action_name.startswith("flyairplane"):
        tmp = re.sub("flyairplane", "FLY-AIRPLANE ", action_name)
    elif action_name.startswith("unloadairplane"):
        tmp = re.sub("unloadairplane", "UNLOAD-AIRPLANE ", action_name)
    else:
        print(action_name)

    action, objects_string = tmp.split()
    objects = re.findall("[a-z]+\\d+", objects_string)
    objects.insert(0, action)
    return "_".join(objects)


def reverse_format(action_name):
    tmp = action_name.lower()
    tmp = tmp.split("-")
    tmp = [tmp[0]] + tmp[1].split("_")
    return "".join(tmp)


def get_possible_actions_ids(
    problem,
    state,
    simulator,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
):
    actions = get_possible_actions(problem, state, simulator)
    gpt_names = [reverse_format(action.name) for action in actions]
    ids = tokenizer.encode(" ".join(gpt_names), return_token_type_ids=False)
    ids = ids[1:-1]
    actions_ids_dict = {id: action for action, id in zip(actions, ids)}
    return actions_ids_dict


if __name__ == "__main__":
    main()
