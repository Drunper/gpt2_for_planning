import os
import re
import torch

from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from typing import Union, cast

from unified_planning.engines.compilers.grounder import Grounder
from unified_planning.io.pddl_reader import PDDLReader
from unified_planning.engines import SequentialSimulator
from unified_planning.model import UPCOWState
from unified_planning.shortcuts import *
from unified_planning.model.walkers import StateEvaluator
from unified_planning.model import FNode


class SimulationMixin:
    reader = PDDLReader()
    grounder = Grounder()

    def get_simulation_tools(self, pddl_dir, pddl_domain_file, problem_id):
        problem = self.reader.parse_problem(
            pddl_dir + os.sep + pddl_domain_file,
            pddl_dir + os.sep + problem_id + ".pddl",
        )
        problem = self.grounder.compile(problem).problem
        init_state = UPCOWState(problem.initial_values)
        simulator = SequentialSimulator(problem)
        state_evaluator = StateEvaluator(problem)
        return problem, init_state, simulator, state_evaluator

    def get_possible_actions(self, problem, state, simulator):
        events = simulator.get_applicable_events(state)
        events = list(events)

        possible_actions = []
        for ev in events:
            for ac in problem.actions:
                if ac.preconditions == ev.conditions and ac.effects == ev.effects:
                    possible_actions.append(ac)
        return possible_actions

    def get_action_by_name(self, possible_actions_ids_dict, action_name):
        possible_actions = list(possible_actions_ids_dict.values())
        for action in possible_actions:
            if action.name == action_name:
                return action
        return None

    def apply_action_to_state(self, action, state, simulator):
        event = list(simulator.get_events(action, []))[0]
        next_state = cast(UPCOWState, simulator.apply(event, state))
        return next_state

    def format_action(self, action_name):
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

    def reverse_format(self, action_name):
        tmp = action_name.lower()
        tmp = tmp.split("-")
        tmp = [tmp[0]] + tmp[1].split("_")
        return "".join(tmp)

    def get_possible_actions_ids(
        self,
        problem,
        state,
        simulator,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ):
        actions = self.get_possible_actions(problem, state, simulator)
        gpt_names = [self.reverse_format(action.name) for action in actions]
        ids = tokenizer.encode(" ".join(gpt_names), return_token_type_ids=False)
        ids = ids[1:-1]
        actions_ids_dict = {id: action for action, id in zip(actions, ids)}
        return actions_ids_dict

    def get_logits_mask(self, possible_actions_ids_dict_list, eos_token_id, token_ids):
        cuda0 = torch.device("cuda:0")
        logits_mask_list = []
        for possible_actions_ids_dict in possible_actions_ids_dict_list:
            if possible_actions_ids_dict:
                possible_actions_ids = set(possible_actions_ids_dict.keys())
                possible_actions_ids.add(eos_token_id)
                logits_mask_list.append(torch.tensor(list(token_ids - possible_actions_ids), device=cuda0))
            else:
                logits_mask_list.append(torch.tensor([], dtype=torch.long, device=cuda0))
        return logits_mask_list
    
    def check_goals(self, state_evaluator: StateEvaluator, goals: FNode, current_state: UPCOWState):
        return state_evaluator.evaluate(goals, current_state).bool_constant_value()
