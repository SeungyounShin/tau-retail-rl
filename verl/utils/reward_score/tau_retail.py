from hashlib import sha256
import inspect
import json
import copy
from typing import Any, Callable, Dict, List, Type, Optional, Set, Union, Tuple
from verl.tools.tau_retail._logic import ACTION_DISPATCH
from verl.interactions.tau_retail_data import load_data

ToHashable = Union[
    str, int, float, Dict[str, "ToHashable"], List["ToHashable"], Set["ToHashable"]
]
Hashable = Union[str, int, float, Tuple["Hashable"], Tuple[Tuple[str, "Hashable"]]]
RESPOND_ACTION_NAME = "respond"

def to_hashable(item: ToHashable) -> Hashable:
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    elif isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    elif isinstance(item, set):
        return tuple(sorted(to_hashable(element) for element in item))
    else:
        return item

def consistent_hash(
    value: Hashable,
) -> str:
    return sha256(str(value).encode("utf-8")).hexdigest()

def step(action: dict, raw_data: dict | None) -> None:
    if not isinstance(raw_data, dict):
        return

    name   = action["name"]
    kwargs = action.get("kwargs", {}) or {}

    func = ACTION_DISPATCH.get(name)
    if func is None:
        return

    sig       = inspect.signature(func)
    accepted  = {k: v for k, v in kwargs.items() if k in sig.parameters}

    func(raw_data, **accepted)


def compute_score(
    solution_str: str,
    ground_truth: str,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
    data: dict = None,
    raw_data: dict = None,
) -> float:

    if raw_data is None:
        raw_data = load_data()

    def get_data_hash(data: dict) -> str:
        return consistent_hash(to_hashable(data))
    
    data_hash = get_data_hash(copy.deepcopy(data))
    raw_data_copy = copy.deepcopy(raw_data)

    gt_actions_raw = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    gt_actions: list[dict] = []
    for act in gt_actions_raw:
        kwargs = act.get("kwargs", {})
        if isinstance(kwargs, str):
            try:
                kwargs = json.loads(kwargs)
            except json.JSONDecodeError:
                kwargs = {}
        gt_actions.append({**act, "kwargs": kwargs})

    actions = [action for action in gt_actions if action.get("name") != RESPOND_ACTION_NAME]

    for action in actions:
        step(action, raw_data_copy)
        _tmp_hash = get_data_hash(raw_data_copy)
    gt_data_hash = get_data_hash(raw_data_copy)
        
    # print in red for 0.0 and green for 1.0
    if data_hash == gt_data_hash:
        # print(f"\033[92m<debug>: gt_actions : {actions} solution_str: {solution_str} | gt_data_hash: {gt_data_hash} | data_hash: {data_hash}\033[0m")
        reward = 1.0
    else:
        # print(f"\033[91m<debug>: gt_actions : {actions} | gt_data_hash: {gt_data_hash} | data_hash: {data_hash}\033[0m")
        reward = 0.0

    return reward