from hashlib import sha256
import inspect
from typing import Any, Callable, Dict, List, Type, Optional, Set, Union, Tuple
from verl.tools.tau_retail._logic import ACTION_DISPATCH

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

def step(action: dict, raw_data: dict) -> None:
    name   = action["name"]
    kwargs = action.get("kwargs", {}) or {}
    func   = ACTION_DISPATCH.get(name)
    if func is None:
        return

    sig = inspect.signature(func)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}

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

    def get_data_hash(data: dict) -> str:
        return consistent_hash(to_hashable(data))
    
    data_hash = get_data_hash(data)

    actions = [
        action for action in ground_truth if action['name'] != RESPOND_ACTION_NAME
    ]

    for action in actions:
        step(action, raw_data)
    gt_data_hash = get_data_hash(raw_data)

    reward = 1.0 if data_hash == gt_data_hash else 0.0

    return reward