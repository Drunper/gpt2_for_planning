# Script per generare dei piani per l'analisi degli errori.
# I piani generati saranno contenuti all'interno di un file di testo
# e in formato JSON. Sono contenute alcune informazioni come le azioni
# in top 5, con le probabilità corrispondenti.

import torch
import sys
import os
import json
import logging
import random
import re
from tqdm import tqdm
from dataclasses import dataclass, field

from datasets import load_dataset
from torch.utils.data import DataLoader
from pathlib import Path
from model import GPT2PModel

from transformers import (
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
)

from typing import Optional, Tuple, Union


# Definizione delle opzioni
@dataclass
class ErrorAnalysisArgs:
    """
    Opzioni per l'analisi degli errori.
    """

    dataset_file: Optional[str] = field(
        default="20_plans.json",
        metadata={"help": "File che contiene i piani (JSON) da utilizzare per l'analisi."},
    )
    tokenizer_path: Optional[str] = field(
        default="tokenizer_generation",
        metadata={"help": "Cartella contenente i file del tokenizer."},
    )
    model_path: Optional[str] = field(
        default="",
        metadata={"help": "Cartella contenente i file del modello."},
    )
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "Cartella di output."}
    )
    max_length: Optional[int] = field(
        default=150,
        metadata={"help": "Lunghezza massima dei piani."},
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Numero di beam per la generazione utilizzando la beam search, al momento non funziona."}
    )
    actions_seen: Optional[int] = field(
        default=0, metadata={"help": "Numero di azioni da aggiungere all'input iniziale."}
    )
    top_k: Optional[int] = field(
        default=5, metadata={"help": "Numero di azioni da considerare per la top k"}
    )
    pddl_dir: Optional[str] = field(
        default="pddl",
        metadata={"help": "Cartella dei file PDDL dei problemi"},
    )
    pddl_domain_file: Optional[str] = field(
        default="domain.pddl",
        metadata={"help": "Nome del file che contiene la definizione del dominio"},
    )
    log_file_name: Optional[str] = field(
        default="error_analysis.log", metadata={"help": "Nome del file di log."}
    )
    shuffle_initial_state: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Per indicare se è necessario fare lo shuffle dello stato iniziale."
        },
    )
    seed: Optional[int] = field(
        default=7, metadata={"help": "Seed per riproducibilità dei risultati."}
    )


logger = logging.getLogger(__name__)


def main():
    # Parsing delle opzioni
    parser = HfArgumentParser(ErrorAnalysisArgs)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()

    # Creazione delle cartelle di output, configurazione del logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "txt").mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "json").mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir.joinpath(args.log_file_name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file_path),
        ],
    )

    # Caricamento del dataset
    dataset = load_dataset("json", data_files=args.dataset_file)
    logger.info("Dataset loaded successfully")

    # Caricamento del tokenizer e del modello
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = GPT2PModel.from_pretrained(args.model_path, device_map="auto")

    # Seed
    random.seed(args.seed)

    # Definizione delle funzioni di preprocessing del dataset e
    # tokenizzazione dello stesso.
    def tokenize_function(examples):
        return tokenizer(
            examples["input"],
            return_token_type_ids=False,
        )

    # Penso ci siano modi migliori per strutturare questa parte,
    # ho fatto così perchè avevo fretta.
    if args.shuffle_initial_state:
        column_names = ["name", "states"]

        def shuffle_initial_state(examples):
            output = []
            for initial_state in examples["initial_state"]:
                initial_state_fluents = initial_state.split(" ")
                random.shuffle(initial_state_fluents)

                new_state = " ".join(initial_state_fluents)
                output.append(new_state)
            return {"states_shuffled": output}

        def get_inputs_for_generation(examples):
            output = []
            for initial_state, goals, actions in zip(examples["states_shuffled"], examples["goals"], examples["actions"]):
                example = initial_state + "<|goals|>" + goals + " <|actions|>"
                action_list = actions.split(" ")
                action_string = " ".join(action_list[: args.actions_seen])
                if action_string != "":
                    example = example + " " + action_string
                output.append(example)
            return {"input": output}

        logger.info(f"Sample {0} of the test set: {dataset['train'][0]}.")

        shuffled_datasets = dataset.map(
            shuffle_initial_state,
            batched=True,
            remove_columns=column_names,
            desc="Shuffling initial state fluents for every example of the dataset",
        )

        logger.info(f"Sample {0} of the test sett: {shuffled_datasets['train'][0]}.")

        pre_processed_dataset = shuffled_datasets.map(
            get_inputs_for_generation,
            batched=True,
            remove_columns=["states_shuffled", "actions"],
            desc="Running input pre-processing on dataset",
        )

        logger.info(f"Sample {0} of the test settt: {pre_processed_dataset['train'][0]}.")
    else:
        column_names = ["name", "states", "actions"]

        def get_inputs_for_generation(examples):
            output = []
            for initial_state, goals, actions in zip(examples["states"], examples["goals"], examples["actions"]):
                example = initial_state + "<|goals|>" + goals + " <|actions|>"
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

    if len(test_dataset) >= 3:
        for index in random.sample(range(len(test_dataset)), 3):
            logger.info(f"Sample {index} of the test set: {test_dataset[index]}.")

    # Definizione del data collator, nel caso della generazione
    # potrebbe non servire.
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=1,
    )

    logger.info("Starting generation of plans")
    logger.info(f"Dataset size: {len(tokenized_datasets['train'])}")

    actions_token_id = tokenizer.convert_tokens_to_ids("<|actions|>")
    softmax = torch.nn.Softmax(dim=0)

    # Inizio generazione piani
    for step, batch in enumerate(tqdm(test_dataloader)):
        example_dict = {}
        example_dict["example_name"] = dataset["train"][step]["name"]
        example_dict["problem_id"] = dataset["train"][step]["pddl_problem_file"].split(".")[0]

        states = shuffled_datasets["train"][step]["states_shuffled"] if args.shuffle_initial_state else dataset["train"][step]["states"]
        states = states + "<|goals|>" + dataset["train"][step]["goals"]
        (
            example_dict["initial_state"],
            example_dict["goals"],
        ) = get_initial_state_and_goals_strings(states)

        real_plan = dataset["train"][step]["actions"]
        real_actions = [beautify(action, False) for action in real_plan.split(" ")]
        real_actions.append("<|endofplan|>")
        example_dict["real_plan"] = " ".join(real_actions)
        example_dict["real_plan_length"] = dataset["train"][step]["len_plan"]

        inputs = batch["input_ids"]
        inputs = inputs.to("cuda:0")

        # Eventualmente qua c'è la chiamata alla funzione che fa cose
        # passo inputs e il dizionario, internamente farà una copia del dizionario

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                num_beams=args.num_beams,
                do_sample=False,
                max_new_tokens=args.max_length,
                pad_token_id=tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_sequence = outputs.sequences[0]

        example_dict["generated_plan"], example_dict["generated_plan_length"] = get_generated_plan(
            generated_sequence, tokenizer, actions_token_id
        )

        logits = outputs.scores
        predictions_list = []
        for position, prediction in enumerate(logits):
            prediction_dict = {}
            actions_seen = real_actions[:args.actions_seen]
            for i in range(position):
                to_append = torch.argmax(logits[i])
                if to_append != tokenizer.eos_token_id:
                    to_append = beautify(tokenizer.decode(to_append), False)
                else:
                    to_append = "<|endofplan|>"
                actions_seen.append(to_append)
            prediction_dict["actions_seen"] = " ".join(actions_seen)
            probabilities = softmax(prediction[0])
            top_k_probabilities, top_k_ids = torch.topk(probabilities, args.top_k)
            try:
                prediction_dict["real_action"] = real_actions[position + args.actions_seen]
            except IndexError:
                prediction_dict["real_action"] = ""
            prediction_dict["generated_action"] = beautify(
                tokenizer.decode(top_k_ids[0]), False
            )
            top_k = []
            for probability, token_id in zip(top_k_probabilities, top_k_ids):
                to_append = (beautify(tokenizer.decode(token_id), False), probability.item())
                top_k.append(to_append)
            prediction_dict["top_k"] = top_k
            predictions_list.append(prediction_dict)
        example_dict["predictions"] = predictions_list
        write_output_to_file(output_dir=output_dir, example_dict=example_dict, step=step, top_k=args.top_k, actions_seen=args.actions_seen)


def write_output_to_file(output_dir: Path, example_dict: dict, step: int, top_k: int, actions_seen: int):
    txt_path = output_dir.joinpath("txt", f"example_{step}.txt")
    json_path = output_dir.joinpath("json", f"example_{step}.json")
    # logger.info(f"Writing outputs to files {txt_path} and {json_path}")
    with open(txt_path, "w") as output_file:
        output_file.write("***** Error analysis output *****\n")
        output_file.write(f"--- example_name: {example_dict['example_name']}\n")
        output_file.write(f"--- problem_id: {example_dict['problem_id']}\n")
        output_file.write(f"--- initial_state: {example_dict['initial_state']}\n")
        output_file.write(f"--- goals: {example_dict['goals']}\n")

        for idx, prediction_info in enumerate(example_dict['predictions']):
            output_file.write(f"\n***** Generation of action {actions_seen + idx + 1}  *****\n")
            output_file.write(f"--- actions_seen: {prediction_info['actions_seen']}\n")
            output_file.write(f"--- generated_action: {prediction_info['generated_action']}\n")
            output_file.write(f"--- real_action: {prediction_info['real_action']}\n")
            output_file.write(f"--- top_{top_k}:\n")
            for action, probability in prediction_info['top_k']:
                output_file.write(f"\t{action}: {probability}\n")

        output_file.write("\n***** Generation summary  *****\n")
        output_file.write(f"--- generated_plan: {example_dict['generated_plan']}\n")
        output_file.write(f"--- generated_plan_length: {example_dict['generated_plan_length']}\n")
        output_file.write(f"--- real_plan: {example_dict['real_plan']}\n")
        output_file.write(f"--- real_plan_length: {example_dict['real_plan_length']}\n")

    with open(json_path, "w") as output_file:
        json.dump(example_dict, output_file, indent=2)
    # logger.info("Output files written successfully")


def get_initial_state_and_goals_strings(states_string):
    fluents = states_string.split(" ")
    initial_state_list = []
    goals_list = []
    goals = False
    for fluent in fluents:
        if fluent == "<|goals|>":
            goals = True
        elif goals:
            goals_list.append(beautify(fluent, True))
        else:
            initial_state_list.append(beautify(fluent, True))

    return " ".join(initial_state_list), " ".join(goals_list)


def beautify(token: str, fluent: bool) -> str:
    if fluent:
        result = token.replace("pos", "_pos")
        result = result.replace("tru", "_tru")
        result = result.replace("obj", "_obj")
        result = result.replace("apn", "_apn")
        result = result.replace("apt", "_apt")
        result = result.replace("cit", "_cit")
        return result
    else:
        return format_action(token)


def format_action(action_name: str) -> str:
    action_name = action_name.lower()
    if action_name == "<|endofplan|>":
        return action_name
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


def get_generated_plan(
    generated_sequence: torch.Tensor,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    actions_token_id: int,
) -> Tuple[str, int]:
    actions_idx = (generated_sequence == actions_token_id).nonzero(as_tuple=True)[0]
    generated_plan = generated_sequence[actions_idx + 1:]
    actions = []
    eos_found = False
    for token_id in generated_plan:
        if token_id != tokenizer.eos_token_id:
            actions.append(beautify(tokenizer.decode(token_id), False))
        else:
            eos_found = True
            actions.append("<|endofplan|>")
    if eos_found:
        return " ".join(actions), len(actions) - 1
    else:
        return " ".join(actions), len(actions)


if __name__ == "__main__":
    main()
