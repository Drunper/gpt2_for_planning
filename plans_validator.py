import re
import os
import json
from pathlib import Path


VALIDATOR_PATH = Path(".", "validate")
DOMAIN_PATH = Path("pddl", "domain.pddl")
VAL_OUTPUT_PATH = Path("output", "val_output.txt")
PLAN_REBUILD_PATH = Path("output", "plan.txt")
JSON_PLANS_PATH = Path("input", "to_validate.json")

PRECONDITION_ERROR = "has an unsatisfied precondition at time"
TYPE_CHECKING_ERROR = "Error: Error in type-checking!"
BAD_OPERATOR_ERROR = "Bad operator in plan"
UNSATISFIED_GOAL_ERROR = "Goal not satisfied"


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
        parts.append(actions_and_objects[i])  # Append the action

        objects = re.findall(
            "[A-Z]+\\d+", actions_and_objects[i + 1]
        )  # Find the objects in the second part of the action

        parts = parts + objects
        original_format = " ".join(parts)
        original_format = "(" + original_format + ")" + "\n"
        nuovo_piano.append(original_format)

    with open(file_path, "w") as output_file:
        output_file.write("".join(nuovo_piano))


def validate(plan_file_path, problem_id):
    problem_path = Path("pddl", f"{problem_id}.pddl")
    val_cmd = (
        f"{VALIDATOR_PATH.absolute()} -v  {DOMAIN_PATH.absolute()} {problem_path.absolute()} {plan_file_path.absolute()} "
        f" > {VAL_OUTPUT_PATH.absolute()} "
    )
    os.system(val_cmd)


def validate_all(plans):
    for plan in plans:
        problem_id = plan["problem_id"]
        print(f"Problema del piano: {problem_id}")
        n_actions_seen = plan["actions_seen"]
        print(f"Numero di azioni viste: {n_actions_seen}")
        actions = plan["plan"]
        rebuild_plan(actions, PLAN_REBUILD_PATH)
        validate(PLAN_REBUILD_PATH, problem_id)
        check_val_output()


def check_val_output():
    with open(VAL_OUTPUT_PATH, "r") as val_output_file:
        lines = val_output_file.readlines()

    print("Controllo validità piano in corso...")
    for line in lines:
        if PRECONDITION_ERROR in line:
            step = int(line.split("time")[-1])
            print(f"Trovata precondizione non soddisfatta al passo {step}")
        elif TYPE_CHECKING_ERROR in line:
            print("Errore durante il type checking")
        elif BAD_OPERATOR_ERROR in line:
            print("Bad operator (?)")
        elif UNSATISFIED_GOAL_ERROR in line:
            print("Il piano è valido ma il goal non è stato raggiunto")
    print("Piano corretto, credo")


if __name__ == "__main__":
    with open(JSON_PLANS_PATH, "r") as plans_file:
        plans = json.load(plans_file)
    validate_all(plans)
