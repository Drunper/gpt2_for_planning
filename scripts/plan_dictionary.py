# Script per ottenere, a partire da piani relativi ad un certo dominio, un dizionario
# che contiene tutte le possibili azioni e fatti per i piani considerati.
import pickle
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
import plan_utils

ENCODING = "UTF8"
TXT_EXT = ".txt"
CSV_EXT = ".csv"
INFO = "_info"
STATS = "_stats"
ALL_DOMAIN = "all_domain"


def create_actions_fluents_dictionary(plans):
    ap_dict = dict()
    i = 1
    for plan in plans:
        for action in plan.actions:
            if action.name not in ap_dict:
                ap_dict[action.name] = i
                i += 1
        for predicate in plan.initial_state:
            if predicate not in ap_dict:
                ap_dict[predicate] = i
                i += 1
        for predicate in plan.goals:
            if predicate not in ap_dict:
                ap_dict[predicate] = i
                i += 1
    return ap_dict


def main():
    parser = ArgumentParser(
        usage="plan_dictionary input output", formatter_class=RawTextHelpFormatter
    )

    parser.add_argument(
        "input",
        help="""Pickle che contiene i piani di un dato dominio""",
    )
    parser.add_argument(
        "output", help="""Nome del file di output che contiene il dizionario"""
    )

    args = parser.parse_args()
    plans_pickle_path = Path(args.input)

    if not plans_pickle_path.is_file():
        print("Il percorso specificato non e' un file")
        sys.exit(1)

    plans = plan_utils.read_plans_from_pickle(plans_pickle_path)
    print("Piani letti")
    ap_dict = create_actions_fluents_dictionary(plans)
    print("Dizionario creato")
    output_path = Path(args.output)

    with open(output_path, "wb") as output_file:
        pickle.dump(ap_dict, output_file)

    print("Finito")


if __name__ == "__main__":
    main()
