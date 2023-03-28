# Script per calcolare alcune statistiche relativi ad un insieme di piani.
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

import pandas as pd
import plan_utils

ENCODING = "UTF8"
TXT_EXT = ".txt"
CSV_EXT = ".csv"
INFO = "_info"
STATS = "_stats"
ALL_DOMAIN = "all_domain"


def goals_len(plan):
    return len(plan.goals)


def actions_len(plan):
    return len(plan.actions)


def initial_state_len(plan):
    return len(plan.initial_state)


def main():
    parser = ArgumentParser(
        usage="plan_stats input output", formatter_class=RawTextHelpFormatter
    )

    parser.add_argument("input", help="""Pickle che contiene i piani""")
    parser.add_argument(
        "output", help="""Cartella che contiene l'output dell'esecuzione"""
    )

    args = parser.parse_args()

    plans_pickle_path = Path(args.input)

    if not plans_pickle_path.is_file():
        print("Il percorso specificato non e' un file")
        sys.exit(1)

    output_path = Path(args.output)

    try:
        output_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(
            f"La cartella {str(output_path)} esiste gia', il suo contenuto potrebbe essere riscritto"
        )
    else:
        print(f"La cartella {str(output_path)} e' stata creata con successo")

    plans = plan_utils.read_plans_from_pickle(plans_pickle_path)
    plans_attributes_list = list(
        zip(
            list(map(lambda plan: plan.plan_name, plans)),
            list(map(initial_state_len, plans)),
            list(map(goals_len, plans)),
            list(map(actions_len, plans)),
        )
    )
    plans_df = pd.DataFrame(
        plans_attributes_list,
        columns=[
            "nome_piano",
            "dimensione_stato_iniziale",
            "dimensione_goal",
            "numero_azioni",
        ],
    )

    output_path_csv = Path(args.output, plans_pickle_path.name + STATS + CSV_EXT)
    outpath_path_txt = Path(args.output, plans_pickle_path.name + INFO + TXT_EXT)

    plan_utils.print_plan_domain_info(plans_df)
    plans_df.to_csv(output_path_csv, index=False)
    print(
        f"Salvataggio statistiche piani per il dominio {plans_pickle_path.name.split('_')[0]} su {output_path_csv.name} completato"
    )
    plan_utils.save_plan_domain_info_to_file(outpath_path_txt, plans_df)
    print(
        f"Salvataggio informazioni piani per il dominio {plans_pickle_path.name.split('_')[0]} su {outpath_path_txt.name} completato"
    )

    print("Finito")


if __name__ == "__main__":
    main()
