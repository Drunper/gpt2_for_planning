# Script per convertire i piani contenuti in un pickle in una serie di file JSON.
# Vengono creati 3 tipi di file: train, test e validation. Ogni file può contenere
# un dato numero di piani, è possibile specificare quanti tramite l'opzione plans_per_file.
# Ogni piano presenta i seguenti campi:
# pddl_problem_file: è il nome del file PDDL che contiene il problema che il piano risolve
# name: nome del piano, copiato dall'oggetto piano contenuto nel pickle
# initial_state: stringa contenente i fatti dello stato iniziale. I fatti sono separati da spazi
# goals: stringa contenente i goal del problema, separati da spazio
# actions: le azioni del piano
# len_initial_state: numero di fatti dello stato iniziale
# len_goals: numero di goal
# len_plan: lunghezza del piano
# actions_idx: indice del token speciale <|actions|> quando inserito dal tokenizer
# eop_idx: indice del token speciale <|endofplan|> quando inserito dal tokenizer (non importante)

from math import floor
import sys
import os
from pathlib import Path
import random
import json

import plan_utils

from tqdm import tqdm
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional


# Definizione delle opzioni
@dataclass
class PlansConverterArgs:
    """
    Opzioni per la conversione di un piano in formato JSON.
    """

    input_pickle: Optional[str] = field(
        default="logistics_plans",
        metadata={"help": "Percorso del file pickle contenente i piani."},
    )
    output_dir: Optional[str] = field(
        default="plans", metadata={"help": "Cartella di output dove verrano salvati i piani."}
    )
    add_invariants: Optional[bool] = field(
        default=True, metadata={"help": "Se true, aggiunge gli invarianti ai piani, richiede di specificare l'opzione pddl_dir."}
    )
    domain_name: Optional[str] = field(
        default="logistics", metadata={"help": "Nome del dominio dei piani, per esempio logistics."}
    )
    pddl_dir: Optional[str] = field(
        default="pddl",
        metadata={"help": "Percorso della cartella contenente i problemi in formato PDDL."},
    )
    output_pickle: Optional[str] = field(
        default="logistics_plans_with_invariants", metadata={"help": "Nome del file pickle di output, contenente i piani con l'aggiunta degli invarianti."}
    )
    seed: Optional[int] = field(
        default=7, metadata={"help": "Seed utilizzato per fare lo split train-test-validation."}
    )
    test_set_size: Optional[float] = field(
        default=0.05, metadata={"help": "Dimensione del test set, espresso come un numero tra 0 e 1."}
    )
    val_set_size: Optional[float] = field(
        default=0.05, metadata={"help": "Dimensione del validation set, espresso come un numero tra 0 e 1."}
    )
    plans_per_file: Optional[int] = field(
        default=20000, metadata={"help": "Numero di piani contenuti in un singolo file JSON."}
    )


# Rimuove gli spazi e trattini dalle azioni o fatti.
def remove_space_and_dash(action_or_fluent):
    return action_or_fluent.replace(" ", "").replace("-", "")


def get_plans_id_dict(plans):
    return {
        i: plans[i].plan_name.split("-")[-1].split("_")[0] for i in range(len(plans))
    }


# I piani che si riferiscono allo stesso problema devono finire o nel training set
# o nel test/validation set, non possono stare da entrambe le parti.
def get_train_val_test_plans(plans, test_set_size, val_set_size):
    plans_id_dict = get_plans_id_dict(plans)
    train_indexes, validation_indexes, test_indexes = get_train_val_test_indexes(
        plans, plans_id_dict, test_set_size, val_set_size
    )
    train_plans = get_plans_with_indexes(plans, train_indexes)
    test_plans = get_plans_with_indexes(plans, test_indexes)
    validation_plans = get_plans_with_indexes(plans, validation_indexes)
    return train_plans, test_plans, validation_plans


# Per ottenere i piani dati gli indici
def get_plans_with_indexes(plans, indexes):
    return [plans[i] for i in indexes]


# Prima estraggo gli indici del test set, poi quelli del validation
# quelli che mi rimangono sono quelli del training set
def get_train_val_test_indexes(plans, plans_id_dict, test_set_size, val_set_size):
    test_indexes = extract_indexes(plans, plans_id_dict, test_set_size)
    # La riga che segua lo messa per ottenere un validation set piccolo da utilizzare per fare
    # piccoli test. Lo script dataset_manager.py permette di splittare il test ottenuto per
    # ottenere il validation e test set finali.
    validation_indexes = extract_indexes(plans, plans_id_dict, 20, 6, 14)
    train_indexes = list(plans_id_dict.keys())
    return train_indexes, validation_indexes, test_indexes


def extract_indexes(plans, plans_id_dict: dict, size, min_len=0, max_len=10000):
    indexes = []
    set_size = size
    if isinstance(size, float):
        set_size = floor(size * len(plans_id_dict))

    while len(indexes) < set_size:
        keys = list(plans_id_dict.keys())
        to_add = random.choice(keys)
        while (
            len(plans[to_add].actions) < min_len or len(plans[to_add].actions) > max_len
        ):
            to_add = random.choice(keys)
        indexes.append(to_add)
        plan_id = plans_id_dict.pop(to_add)
        to_remove = []
        for key, value in plans_id_dict.items():
            if value == plan_id:
                indexes.append(key)
                to_remove.append(key)
        for key in to_remove:
            plans_id_dict.pop(key)
    return indexes


def convert_plan_to_json(plan):
    initial_state_list = [
        remove_space_and_dash(fluent) for fluent in plan.initial_state
    ]
    goals_list = [remove_space_and_dash(fluent) for fluent in plan.goals]
    actions_list = [remove_space_and_dash(action.name) for action in plan.actions]
    initial_state = " ".join(initial_state_list)
    goals = " ".join(goals_list)
    actions = " ".join(actions_list)

    plan_dict = dict()
    plan_dict["name"] = plan.plan_name
    plan_dict["pddl_problem_file"] = plan.plan_name.split("-")[-1].split("_")[0] + ".pddl"
    plan_dict["initial_state"] = initial_state
    plan_dict["goals"] = goals
    plan_dict["actions"] = actions
    plan_dict["len_initial_state"] = len(initial_state_list)
    plan_dict["len_goals"] = len(goals)
    plan_dict["len_plan"] = len(actions)
    plan_dict["actions_idx"] = len(initial_state_list) + len(goals_list) + 2
    plan_dict["eop_idx"] = len(initial_state_list) + len(goals_list) + len(actions_list) + 3
    return plan_dict


def write_plans(plans, folder, domain, example_type, plans_per_file):
    file_ext = "json"
    for i in range(int(len(plans) / plans_per_file)):
        path = Path(folder, f"{domain}_plans_{example_type}_{i}.{file_ext}")
        write_plans_to_json_file(plans[i * plans_per_file: (i + 1) * plans_per_file], path)
    i = int(len(plans) / plans_per_file)
    path = Path(folder, f"{domain}_plans_{example_type}_{i}.{file_ext}")
    write_plans_to_json_file(plans[i * plans_per_file:], path)


def write_plans_to_json_file(plans, path):
    with open(path, "w", encoding=plan_utils.ENCODING) as output_file:
        formatted_plans = [convert_plan_to_json(plan) for plan in plans]
        json_str = json.dumps(formatted_plans, indent=2)
        output_file.write(json_str)


# Per prendere solo alcune righe del file PDDL
def get_lines_from_pddl(pddl_file_path, suffix):
    with open(pddl_file_path, "r") as input_file:
        lines = input_file.readlines()
    lines = [line.strip().replace("(", "").replace(")", "") for line in lines if line.strip().startswith(suffix)]
    return lines


# Funzione per aggiungere gli invarianti, fatta per il dominio logistics
def add_invariants(plans, plans_id_dict, pddl_dir):
    for i in range(len(plans)):
        plan_problem_path = Path(pddl_dir, f"{plans_id_dict[i]}.pddl")
        invariants = get_lines_from_pddl(plan_problem_path, "(in-city")
        plans[i].initial_state.extend(invariants)


def main():
    # Parsing delle opzioni
    parser = HfArgumentParser(PlansConverterArgs)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()

    # Seed
    random.seed(args.seed)
    plans_pickle_path = Path(args.input_pickle)

    if not plans_pickle_path.is_file():
        print("Il percorso specificato non e' un file")
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("Lettura dei piani da pickle, ci vorranno alcuni minuti...")
    plans = plan_utils.read_plans_from_pickle(plans_pickle_path)
    print(f"Numero di piani letti da file: {len(plans)}")

    if args.add_invariants:
        plans_id_dict = get_plans_id_dict(plans)
        add_invariants(plans, plans_id_dict, args.pddl_dir)
        plans_pickle_path = Path(args.output_pickle)
        plan_utils.write_plans_to_pickle(plans, plans_pickle_path)

    train_plans, test_plans, val_plans = get_train_val_test_plans(plans, args.test_set_size, args.val_set_size)
    write_plans(train_plans, args.output_dir, args.domain_name, "train", args.plans_per_file)
    write_plans(test_plans, args.output_dir, args.domain_name, "test", args.plans_per_file)
    write_plans(val_plans, args.output_dir, args.domain_name, "validation", args.plans_per_file)

    print("Finito")


if __name__ == "__main__":
    main()
