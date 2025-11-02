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
    num_turns: int = 0,  # 턴 수 (efficiency penalty용)
    num_errors: int = 0,  # 에러 횟수 (error penalty용)
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
    gt_actions = normalize_actions(gt_actions_raw)

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

    # Base reward: exact match of final world state
    base_reward = 0.0
    if agent_hash == gt_hash:
        base_reward = 1.0
    # If caller provided a fully executed `data` snapshot, optionally compare that too.
    # This covers pipelines that actually execute tools and pass the mutated `data`.
    elif isinstance(data, dict):
        if get_data_hash(copy.deepcopy(data)) == gt_hash:
            base_reward = 1.0
    
    # Reward shaping: efficiency penalty and error penalty
    # Efficiency penalty: 턴 수가 많을수록 감점 (최적 턴 수는 ground truth 길이로 추정)
    optimal_turns = len(gt_actions) * 2  # agent + user turns
    if optimal_turns > 0 and num_turns > optimal_turns:
        efficiency_penalty = -0.01 * (num_turns - optimal_turns)
    else:
        efficiency_penalty = 0.0
    
    # Error penalty: 에러가 발생할 때마다 감점
    error_penalty = -0.05 * num_errors
    
    # 최종 보상 = base + penalties (최소 -0.5, 최대 1.0)
    final_reward = base_reward # + efficiency_penalty + error_penalty
    # final_reward = max(-0.5, min(1.0, final_reward))
    
    # DEBUG: reward shaping 정보 출력
    import sys
    if num_turns > 0 or num_errors > 0:
        print(f"[DEBUG REWARD] base={base_reward:.2f}, turns={num_turns}, "
              f"optimal_turns={optimal_turns}, efficiency_penalty={efficiency_penalty:.3f}, "
              f"errors={num_errors}, error_penalty={error_penalty:.2f}, "
              f"final={final_reward:.3f}", 
              file=sys.stderr, flush=True)
    
    return final_reward
