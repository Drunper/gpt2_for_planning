# Script per controllare la validità dei piani generati utilizzando VAL. I piani
# sono quelli generati utilizzando lo script plans_generator.py, basta indicare come
# cartella di input quella di output dell'altro script.
import re
import os
import subprocess
import sys
import json
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass, field
from hf_argparser_for_val import HfArgumentParser
from typing import Optional
import logging

logger = logging.getLogger(__name__)

PRECONDITION_ERROR = "has an unsatisfied precondition at time"
TYPE_CHECKING_ERROR = "Error in type-checking"
UNKNOWN_TYPE_ERROR = "Object with unknown type:"
BAD_OPERATOR_ERROR = "Bad operator in plan"
UNSATISFIED_GOAL_ERROR = "Goal not satisfied"

VALID_KEY = "valid"
PRECONDITION_ERROR_KEY = "precondition_error"
TYPE_CHECKING_ERROR_KEY = "type_checking_error"
UNKNOWN_TYPE_ERROR_KEY = "unknown_type_error"
BAD_OPERATOR_ERROR_KEY = "bad_operator_error"
UNSATISFIED_GOAL_ERROR_KEY = "unsatisfied_goal_error"


# Definizione delle opzioni
@dataclass
class ValidatorArgs:
    """
    Opzioni per la validazione.
    """

    validator_path: Optional[str] = field(
        default="./validate",
        metadata={"help": "Percorso all'eseguibile di VAL"},
    )
    input_dir: Optional[str] = field(
        default="output", metadata={"help": "Cartella che contiene i piani generati da validare"}
    )
    json_prefix: Optional[str] = field(
        default="to_validate_",
        metadata={
            "help": (
                "Prefisso dei file JSON contenenti i piani, per esempio se"
                "i file sono plans_0.json, plans_1.json, allora json_prefix = 'plans_'"
            )
        }
    )
    output_dir: Optional[str] = field(
        default="aaaaaa", metadata={"help": "Cartella di output"}
    )
    pddl_dir: Optional[str] = field(
        default="pddl",
        metadata={"help": "Cartella contenente i file PDDL"},
    )
    pddl_domain_file: Optional[str] = field(
        default="domain.pddl",
        metadata={"help": "Nome del file PDDL contenente la definizione del dominio"},
    )
    log_file_name: Optional[str] = field(
        default="validator.log",
        metadata={"help": "Nome del file di log"}
    )


# Funzione per costruire un piano nel formato utilizzato da VAL
def rebuild_plan(plan_string, file_path):
    plan_string = re.sub("DRIVETRUCK", "DRIVE-TRUCK ", plan_string.upper())
    plan_string = re.sub("LOADTRUCK", "LOAD-TRUCK ", plan_string)
    plan_string = re.sub("UNLOADTRUCK", "UNLOAD-TRUCK ", plan_string)
    plan_string = re.sub("LOADAIRPLANE", "LOAD-AIRPLANE ", plan_string)
    plan_string = re.sub("FLYAIRPLANE", "FLY-AIRPLANE ", plan_string)
    plan_string = re.sub("UNLOADAIRPLANE", "UNLOAD-AIRPLANE ", plan_string)
    actions_and_objects = plan_string.split()
    nuovo_piano = []

    for i in range(0, len(actions_and_objects), 2):
        parts = []
        parts.append(actions_and_objects[i])  # Aggiungo l'azione

        try:
            objects = re.findall(
                "[A-Z]+\\d+", actions_and_objects[i + 1]
            )  # Cerco gli oggetti che si trovano nella seconda parte
        except IndexError as Argument:
            logger.error("Index error")
            logger.exception(Argument)
            logger.error(f"Value of i: {i}")
            logger.error(f"parts: {parts}")
            logger.error(f"len of actions_and_objects: {len(actions_and_objects)}")
            logger.error(f"Plan string: {plan_string}")
            logger.error(f"actions_and_objects: {actions_and_objects}")
            raise

        parts = parts + objects
        original_format = " ".join(parts)
        original_format = "(" + original_format + ")" + "\n"
        nuovo_piano.append(original_format)

    with open(file_path, "w") as output_file:
        output_file.write("".join(nuovo_piano))


def validate(validator_path: Path, domain_path: Path, problem_path: Path, plan_file_path: Path):
    # val_cmd = (
    #     f"{VALIDATOR_PATH.absolute()} -v  {DOMAIN_PATH.absolute()} {problem_path.absolute()} {plan_file_path.absolute()} "
    #     f" > {VAL_OUTPUT_PATH.absolute()}"
    # )
    val_cmd = [
        validator_path.absolute(),
        "-v",
        domain_path.absolute(),
        problem_path.absolute(),
        plan_file_path.absolute(),
    ]
    output = subprocess.run(val_cmd, capture_output=True)
    return output.stdout.decode(sys.stdout.encoding).split("\n")  # Restituisco l'output dell'esecuzione di VAL


def validate_all(all_plans, pddl_dir, validator_path: Path, domain_path: Path, plans_rebuild_path: Path):
    scores = {}

    # Creo un dizionario per contenere i risultati
    for i in range(6):
        scores[f"actions_seen_{i}"] = {
            "total": 0,
            VALID_KEY: 0,
            PRECONDITION_ERROR_KEY: 0,
            TYPE_CHECKING_ERROR_KEY: 0,
            UNKNOWN_TYPE_ERROR_KEY: 0,
            BAD_OPERATOR_ERROR_KEY: 0,
            UNSATISFIED_GOAL_ERROR_KEY: 0,
        }

    all_precondition_stats = []

    logger.info("Inizio validazione con VAL")

    for plans in all_plans:
        precondition_stats = []
        for i, plan in enumerate(tqdm(plans)):
            if not i % 100:
                logger.info(f"Validati {i} piani di {len(plans)}")
            problem_id = plan["problem_id"]
            n_actions_seen = plan["actions_seen"]
            actions = plan["plan"]
            try:
                rebuild_plan(actions, plans_rebuild_path)
                scores[f"actions_seen_{n_actions_seen}"]["total"] += 1
                problem_path = Path(pddl_dir, problem_id + ".pddl")
                val_output = validate(validator_path, domain_path, problem_path, plans_rebuild_path)
                result = check_val_output(i, val_output)
                if isinstance(result, tuple):
                    precondition_stats.append((i, plan["actions_seen"], result[1]))
                    result = result[0]
                scores[f"actions_seen_{n_actions_seen}"][result] += 1
            except IndexError:
                logger.error(f"example_id: {plan['example_id']}, actions_seen: {plan['actions_seen']}")
        all_precondition_stats.append(precondition_stats)

    return scores, all_precondition_stats


# Funzione che scrive in un file .csv a che passo ci sono i problemi
# con le precondizioni.
def print_precondition_stats(all_precondition_stats, output_dir):
    output_path = Path(output_dir, "preconditions_stats.csv")
    with open(output_path, "w") as output_file:
        output_file.write(",numero_piano,azioni_viste,passo\n")
        for precondition_stats in all_precondition_stats:
            for elem in precondition_stats:
                output_file.write(f"{elem[0] + 1},{elem[1]},{elem[2]}\n")


# Controllo l'output di VAL e segno quale errore viene commesso
def check_val_output(plan_index, lines):
    for line in lines:
        if PRECONDITION_ERROR in line:
            step = int(line.split("time")[-1])
            logger.error(f"Trovata precondizione non soddisfatta al passo {step} nel piano {plan_index + 1}")
            return PRECONDITION_ERROR_KEY, step
        elif TYPE_CHECKING_ERROR in line:
            logger.error(f"Errore durante il type checking nel piano {plan_index + 1} ")
            return TYPE_CHECKING_ERROR_KEY
        elif UNKNOWN_TYPE_ERROR in line:
            logger.error(f"Oggetto con tipo sconosciuto nel piano {plan_index + 1}")
            return UNKNOWN_TYPE_ERROR_KEY
        elif BAD_OPERATOR_ERROR in line:
            logger.error(f"Bad operator (?) nel piano {plan_index + 1}")
            return BAD_OPERATOR_ERROR_KEY
        elif UNSATISFIED_GOAL_ERROR in line:
            return UNSATISFIED_GOAL_ERROR_KEY
    return VALID_KEY


# Carico i piani da file
def load_plans(plans_dir, json_prefix):
    all_plans = []
    for plan_dir in plans_dir:
        json_paths = sorted(plan_dir.glob(f"{json_prefix}*.json"))
        plans = []
        for json_path in json_paths:
            with open(json_path, "r") as json_file:
                plans.extend(json.load(json_file))
        all_plans.append(plans)
    logger.info("Piani caricati con successo")
    return all_plans


def main():
    # Parsing delle opzioni
    parser = HfArgumentParser(ValidatorArgs)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()

    # Logging e creazione cartella di output
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            # logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, args.log_file_name)),
        ],
    )

    json_plans_path = Path(args.input_dir)
    validator_path = Path(args.validator_path)
    domain_path = Path(args.pddl_dir, args.pddl_domain_file)
    plans_rebuild_path = Path(args.output_dir, "plan.txt")

    # Vado a selezionare le cartelle contenute nella cartella indicata tramite opzioni,
    # quella generata da plans_generator.py. Cerco tutte le cartelle che matchano il
    # pattern indicato.
    plans_dirs = [pathdir for pathdir in json_plans_path.iterdir() if pathdir.match("[0-9]_actions") and pathdir.is_dir()]
    plans = load_plans(plans_dirs, args.json_prefix)
    scores, precondition_stats = validate_all(plans, args.pddl_dir, validator_path, domain_path, plans_rebuild_path)

    print_precondition_stats(precondition_stats, args.output_dir)

    scores_path = Path(args.output_dir, "scores.json")
    with open(scores_path, "w") as output_file:
        json.dump(scores, output_file, indent=4)

    for key, value in scores.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
