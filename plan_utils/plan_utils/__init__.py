import pickle
import sys
import os
from pathlib import Path
import pandas as pd
from typing import Union

from .plan import plan
from .plan import action

sys.modules['plan'] = plan
sys.modules['action'] = action

ENCODING = "UTF8"
TXT_EXT = ".txt"
CSV_EXT = ".csv"
INFO = "_info"
STATS = "_stats"
ALL_DOMAIN = "all_domain"


def read_plans_from_pickle(plans_pickle_path: Union[str, os.PathLike]) -> list[plan.Plan]:
    plans = []
    with open(plans_pickle_path, "rb") as pickle_file:
        try:
            plans = pickle.load(pickle_file)
        except EOFError:
            print("Non dovrebbe succedere, ma e' successo")
    return plans


def write_plans_to_pickle(plans, plans_pickle_path):
    with open(plans_pickle_path, "wb") as pickle_file:
        pickle.dump(plans, pickle_file)


def print_info(column):
    print(f"Media: {column.mean()}")
    print(f"Deviazione standard: {column.std()}")
    print(f"Minimo: {column.min()}")
    print(f"Massimo: {column.max()}")
    print(f"Mediana: {column.median()}")


def print_plan_domain_info(plans_df):
    print(f"Numero di piani: {len(plans_df.index)}")
    print("--------------------------------------------------")
    print("Informazioni su dimensione stato iniziale:")
    print_info(plans_df["dimensione_stato_iniziale"])
    print("--------------------------------------------------")
    print("Informazioni su dimensione goal:")
    print_info(plans_df["dimensione_goal"])
    print("--------------------------------------------------")
    print("Informazioni su numero azioni:")
    print_info(plans_df["numero_azioni"])


def save_plan_domain_info_to_file(file_path, plans_df):
    original_stdout = sys.stdout
    with open(file_path, "w", encoding=ENCODING) as file:
        sys.stdout = file
        print_plan_domain_info(plans_df)
        sys.stdout = original_stdout
