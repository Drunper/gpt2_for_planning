# Script scritto velocemente per ottenere un file JSON simile a quelli
# descritti in plans_converter.py a partire da una serie di file PDDL.
from pathlib import Path
import plan_utils
import json


def preprocess(to_process):
    return to_process.replace(" ", "").replace(")", " ").replace("(", "").strip()


def get_lines_from_file(file_path):
    with open(file_path, "r") as input_file:
        lines = input_file.readlines()
    return [line.strip() for line in lines]


def get_plan_string(file_path):
    lines = get_lines_from_file(file_path)

    j = 3
    lines[j] = lines[j].replace("(:init ", "")
    init_state_fluents = []
    while not lines[j].startswith("(:goal"):
        fluents = preprocess(lines[j]).split()
        for fluent in fluents:
            if fluent.startswith("at"):
                init_state_fluents.append(fluent)
        j += 1

    init_state = " ".join(init_state_fluents)

    lines[j] = lines[j].replace("(:goal (and ", "")
    goal_fluents = []
    while len(lines[j]) != 1:
        fluents = preprocess(lines[j]).split()
        for fluent in fluents:
            if fluent.startswith("at"):
                goal_fluents.append(fluent)
        j += 1

    goals = " ".join(goal_fluents)

    return init_state, goals, len(init_state_fluents), len(goal_fluents)


def get_plan_dict(plan_path: Path):
    initial_state, goals, inital_state_len, goals_len = get_plan_string(plan_path)

    plan_dict = dict()
    plan_dict["name"] = plan_path.name
    plan_dict["pddl_problem_file"] = plan_path.name
    plan_dict["initial_state"] = initial_state
    plan_dict["goals"] = goals
    plan_dict["actions"] = ""
    plan_dict["len_initial_state"] = inital_state_len
    plan_dict["len_goals"] = goals_len
    plan_dict["len_plan"] = 0
    plan_dict["actions_idx"] = inital_state_len + goals_len + 2
    plan_dict["eop_idx"] = -1
    return plan_dict


def write_json_plans(plans_path):
    json_plans = [get_plan_dict(plan_path) for plan_path in plans_path]
    output_dir = Path("logistics00", "json")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = Path("logistics00", "json", "logistics00_plans.json")
    with open(output_path, "w", encoding=plan_utils.ENCODING) as output_file:
        json_str = json.dumps(json_plans, indent=2)
        output_file.write(json_str)


def main():
    plans_dir = Path("logistics00")
    plans_path = sorted(plans_dir.glob("pfileprobLOGISTICS*.pddl"))
    write_json_plans(plans_path)


if __name__ == "__main__":
    main()
