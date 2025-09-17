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
    solution_str: str | list[dict] | None,
    ground_truth: str | list[dict],
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
    data: dict = None,
    raw_data: dict = None,
    *,
    actions: list[dict] | None = None,  # accept alias used by some callers
) -> float:

    if raw_data is None:
        raw_data = load_data()

    def get_data_hash(data: dict) -> str:
        return consistent_hash(to_hashable(data))

    def normalize_actions(raw_actions: list[dict] | None) -> list[dict]:
        if not raw_actions:
            return []
        normalized: list[dict] = []
        for act in raw_actions:
            if not isinstance(act, dict):
                continue
            name = act.get("name")
            if not name:
                continue
            # unify arguments -> kwargs
            if "kwargs" in act:
                kwargs = act.get("kwargs") or {}
                if isinstance(kwargs, str):
                    try:
                        kwargs = json.loads(kwargs)
                    except json.JSONDecodeError:
                        kwargs = {}
            else:
                arguments = act.get("arguments")
                if isinstance(arguments, str):
                    try:
                        kwargs = json.loads(arguments)
                    except json.JSONDecodeError:
                        kwargs = {}
                elif isinstance(arguments, dict):
                    kwargs = arguments
                else:
                    kwargs = {}
            normalized.append({"name": name, "kwargs": kwargs})
        # drop non-mutating respond actions if they appear
        return [a for a in normalized if a.get("name") != RESPOND_ACTION_NAME]

    # parse ground truth actions
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
    gt_actions = [a for a in gt_actions if a.get("name") != RESPOND_ACTION_NAME]

    # parse agent actions: prefer explicit `actions` then fallback to solution_str
    agent_raw_actions: list[dict] | None = None
    if actions is not None:
        agent_raw_actions = actions
    else:
        if isinstance(solution_str, str):
            try:
                agent_raw_actions = json.loads(solution_str)
            except Exception:
                agent_raw_actions = None
        elif isinstance(solution_str, list):
            agent_raw_actions = solution_str
    agent_actions = normalize_actions(agent_raw_actions)

    # simulate both trajectories starting from the same raw_data
    raw_after_agent = copy.deepcopy(raw_data)
    for action in agent_actions:
        step(action, raw_after_agent)

    raw_after_gt = copy.deepcopy(raw_data)
    for action in gt_actions:
        step(action, raw_after_gt)

    agent_hash = get_data_hash(raw_after_agent)
    gt_hash    = get_data_hash(raw_after_gt)

    # strict: exact match of final world state
    if agent_hash == gt_hash:
        return 1.0

    # If caller provided a fully executed `data` snapshot, optionally compare that too.
    # This covers pipelines that actually execute tools and pass the mutated `data`.
    if isinstance(data, dict):
        if get_data_hash(copy.deepcopy(data)) == gt_hash:
            return 1.0

    return 0.0