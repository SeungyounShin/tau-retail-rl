"""Synthetic task generation pipeline for Tau Retail training split.

This module fabricates ground-truth action traces (`gt_actions`) by
systematically probing the Tau Retail business rules. The goal is to
enumerate sequences of write-actions that are guaranteed to succeed when
executed by the environment logic (`verl.tools.tau_retail._logic`).

High level flow:

1. Enumerate candidate write actions for the current data snapshot.
2. Execute each action against a deep copy of the state to verify
   whether it is accepted by the business logic.
3. Grow valid single-step actions into longer sequences (up to the
   requested horizon) while keeping the user identity consistent.
4. For every validated sequence, craft a natural-language instruction
   via lightweight templates (or downstream GPT calls if desired).

The output is a list of `Task` objects that mirrors the hand-authored
dataset in ``tasks_train.py`` but is generated automatically.

Example usage (CLI):

```
python -m examples.data_preprocess.tau_retail.synthetic_gen_tasks \
    --num-tasks 1000 \
    --max-actions 6 \
    --max-sequences 1000 \
    --seed 10 \
    --output-py examples/data_preprocess/tau_retail/tasks_train_gen.py
```

The script writes a Python module (default: ``tasks_train_gen.py``)
containing serialised `Task` objects ready for inspection or further
processing.  Optional JSON output is also available.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .types import Action, Task
from verl.interactions.tau_retail_data import load_data
from verl.tools.tau_retail._logic import ACTION_DISPATCH


# ---------------------------------------------------------------------------
# Helper data structures
# ---------------------------------------------------------------------------


WRITE_ACTIONS = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
}


PERSONALITY_TRAITS = [
    "patient",
    "confident",
    "logical",
    "organized",
    "curious",
    "outgoing",
    "optimistic",
    "pessimistic",
    "independent",
    "messy",
    "busy",
    "cautious",
    "relaxing",
    "polite",
    "direct",
    "shy",
    "happy",
    "sad",
    "flexible",
    "creative",
]


Reason = Tuple[str, str]
Address = Dict[str, str]


@dataclass
class SyntheticAction:
    """Container for a validated action and its prompt fragment."""

    action: Action
    user_id: str
    prompt_fragment: str

    def signature(self) -> str:
        """Stable identifier used to deduplicate action sequences."""

        return json.dumps(
            {
                "name": self.action.name,
                "kwargs": self.action.kwargs,
            },
            sort_keys=True,
        )


# ---------------------------------------------------------------------------
# Action execution helpers
# ---------------------------------------------------------------------------


def execute_action_in_place(data: dict, action: Action) -> Tuple[bool, Optional[str]]:
    """Apply an action to ``data`` using the logic dispatch table."""

    handler = ACTION_DISPATCH.get(action.name)
    if handler is None:
        return False, f"Unsupported action: {action.name}"

    result = handler(data, **action.kwargs)
    if isinstance(result, tuple) and result and isinstance(result[0], bool):
        success = result[0]
        message = result[1] if len(result) > 1 else None
        return success, message if isinstance(message, str) else None

    return False, "Unexpected handler return signature"


def try_action(data: dict, action: Action) -> Tuple[bool, Optional[dict]]:
    """Return a mutated copy of ``data`` if the action succeeds."""

    candidate = copy.deepcopy(data)
    ok, _ = execute_action_in_place(candidate, action)
    if not ok:
        return False, None
    return True, candidate


# ---------------------------------------------------------------------------
# Action enumeration utilities
# ---------------------------------------------------------------------------


def _iter_order_items(order: dict) -> Iterator[dict]:
    for item in order.get("items", []):
        yield item


def _format_options(options: dict) -> str:
    # Replicate the style from the hand-authored dataset (python dict repr).
    return str({k: options[k] for k in sorted(options)})


def _format_semicolon_separated(names: Sequence[str]) -> str:
    if not names:
        return ""
    return "; ".join(names) + ";"


def _format_address(address: Address) -> str:
    ordered = {
        "order_id": address.get("order_id", ""),
        "address1": address["address1"],
        "address2": address["address2"],
        "city": address["city"],
        "country": address["country"],
        "state": address["state"],
        "zip": address["zip"],
    }
    return str(ordered)


def _collect_item_subsets(item_ids: List[str]) -> List[List[str]]:
    unique = list(dict.fromkeys(item_ids))
    subsets: List[List[str]] = []
    for item_id in unique:
        subsets.append([item_id])
    if len(unique) >= 2:
        subsets.append(unique[:2])
    if len(unique) > 2:
        subsets.append(unique)
    return subsets


class ActionEnumerator:
    """Enumerates candidate write actions for a given data snapshot."""

    def __init__(
        self,
        base_data: dict,
        rng: random.Random,
        max_candidates_per_type: int = 10,
        max_variant_per_item: int = 2,
    ) -> None:
        self._base_data = base_data
        self._rng = rng
        self._max_candidates_per_type = max_candidates_per_type
        self._max_variant_per_item = max_variant_per_item

    # Public API ---------------------------------------------------------

    def enumerate(self, data: dict) -> List[SyntheticAction]:
        candidates: List[SyntheticAction] = []
        generators = [
            self._generate_cancel_actions,
            self._generate_return_actions,
            self._generate_exchange_actions,
            self._generate_modify_payment_actions,
            self._generate_modify_order_items_actions,
            self._generate_modify_order_address_actions,
            self._generate_modify_user_address_actions,
        ]

        for generator in generators:
            generator_candidates = generator(data)
            if not generator_candidates:
                continue
            if len(generator_candidates) > self._max_candidates_per_type:
                self._rng.shuffle(generator_candidates)
                generator_candidates = generator_candidates[: self._max_candidates_per_type]
            candidates.extend(generator_candidates)

        self._rng.shuffle(candidates)
        return candidates

    # Candidate builders -------------------------------------------------

    def _generate_cancel_actions(self, data: dict) -> List[SyntheticAction]:
        actions: List[SyntheticAction] = []
        orders = data.get("orders", {})
        for order_id in sorted(orders):
            order = orders[order_id]
            if order.get("status") != "pending":
                continue
            user_id = order["user_id"]
            for reason in ("no longer needed", "ordered by mistake"):
                action = Action(
                    name="cancel_pending_order",
                    kwargs={"order_id": order_id, "reason": reason},
                )
                fragment = f"Cancel order {order_id} because {reason}."
                actions.append(
                    SyntheticAction(action=action, user_id=user_id, prompt_fragment=fragment)
                )
        return actions

    def _generate_return_actions(self, data: dict) -> List[SyntheticAction]:
        actions: List[SyntheticAction] = []
        orders = data.get("orders", {})
        users = data.get("users", {})
        for order_id in sorted(orders):
            order = orders[order_id]
            if order.get("status") != "delivered":
                continue
            user_id = order["user_id"]
            user = users[user_id]
            items = list(_iter_order_items(order))
            if not items:
                continue
            item_lookup = {item["item_id"]: item for item in items}
            subsets = _collect_item_subsets([item["item_id"] for item in items])
            payment_candidates = [order["payment_history"][0]["payment_method_id"]]
            payment_candidates.extend(
                pm_id
                for pm_id, pm in user.get("payment_methods", {}).items()
                if pm.get("source") == "gift_card"
            )

            for subset in subsets:
                item_names = [item_lookup[iid]["name"] for iid in subset if iid in item_lookup]
                for payment_id in payment_candidates:
                    action = Action(
                        name="return_delivered_order_items",
                        kwargs={
                            "order_id": order_id,
                            "item_ids": subset,
                            "payment_method_id": payment_id,
                        },
                    )
                    fragment = (
                        f"Return {order_id} via {payment_id}: "
                        f"{_format_semicolon_separated(item_names)}"
                    )
                    actions.append(
                        SyntheticAction(action=action, user_id=user_id, prompt_fragment=fragment)
                    )
        return actions

    def _generate_exchange_actions(self, data: dict) -> List[SyntheticAction]:
        actions: List[SyntheticAction] = []
        orders = data.get("orders", {})
        users = data.get("users", {})
        products = data.get("products", {})
        for order_id in sorted(orders):
            order = orders[order_id]
            if order.get("status") != "delivered":
                continue
            user_id = order["user_id"]
            user = users[user_id]
            for item in _iter_order_items(order):
                product_id = item["product_id"]
                variants = products.get(product_id, {}).get("variants", {})
                alternatives = [
                    (variant_id, variant_data)
                    for variant_id, variant_data in variants.items()
                    if variant_data.get("available") and variant_id != item["item_id"]
                ]
                if not alternatives:
                    continue
                self._rng.shuffle(alternatives)
                alternatives = alternatives[: self._max_variant_per_item]
                payment_candidates = list(user.get("payment_methods", {}).keys())
                for new_item_id, variant in alternatives:
                    for payment_id in payment_candidates:
                        action = Action(
                            name="exchange_delivered_order_items",
                            kwargs={
                                "order_id": order_id,
                                "item_ids": [item["item_id"]],
                                "new_item_ids": [new_item_id],
                                "payment_method_id": payment_id,
                            },
                        )
                        fragment = (
                            f"For {order_id}, exchange {item['name']} {_format_options(item['options'])} "
                            f"to {_format_options(variant['options'])}; via {payment_id}."
                        )
                        actions.append(
                            SyntheticAction(action=action, user_id=user_id, prompt_fragment=fragment)
                        )
        return actions

    def _generate_modify_payment_actions(self, data: dict) -> List[SyntheticAction]:
        actions: List[SyntheticAction] = []
        orders = data.get("orders", {})
        users = data.get("users", {})
        for order_id in sorted(orders):
            order = orders[order_id]
            if order.get("status") != "pending":
                continue
            user_id = order["user_id"]
            user_methods = users[user_id].get("payment_methods", {})
            if not user_methods or not order.get("payment_history"):
                continue
            current_method = order["payment_history"][0]["payment_method_id"]
            candidates = [pm_id for pm_id in user_methods if pm_id != current_method]
            for payment_id in candidates:
                action = Action(
                    name="modify_pending_order_payment",
                    kwargs={"order_id": order_id, "payment_method_id": payment_id},
                )
                fragment = f"For {order_id}, change payment to {payment_id}."
                actions.append(
                    SyntheticAction(action=action, user_id=user_id, prompt_fragment=fragment)
                )
        return actions

    def _generate_modify_order_items_actions(self, data: dict) -> List[SyntheticAction]:
        actions: List[SyntheticAction] = []
        orders = data.get("orders", {})
        users = data.get("users", {})
        products = data.get("products", {})
        for order_id in sorted(orders):
            order = orders[order_id]
            if order.get("status") != "pending":
                continue
            user_id = order["user_id"]
            for item in _iter_order_items(order):
                product_id = item["product_id"]
                variants = products.get(product_id, {}).get("variants", {})
                alternatives = [
                    (variant_id, variant_data)
                    for variant_id, variant_data in variants.items()
                    if variant_data.get("available") and variant_id != item["item_id"]
                ]
                if not alternatives:
                    continue
                self._rng.shuffle(alternatives)
                alternatives = alternatives[: self._max_variant_per_item]
                payment_candidates = list(users[user_id].get("payment_methods", {}).keys())
                for new_item_id, variant in alternatives:
                    for payment_id in payment_candidates:
                        action = Action(
                            name="modify_pending_order_items",
                            kwargs={
                                "order_id": order_id,
                                "item_ids": [item["item_id"]],
                                "new_item_ids": [new_item_id],
                                "payment_method_id": payment_id,
                            },
                        )
                        fragment = (
                            f"For {order_id}, modify {item['name']} {_format_options(item['options'])} "
                            f"to {_format_options(variant['options'])}; via {payment_id}."
                        )
                        actions.append(
                            SyntheticAction(action=action, user_id=user_id, prompt_fragment=fragment)
                        )
        return actions

    def _generate_modify_order_address_actions(self, data: dict) -> List[SyntheticAction]:
        actions: List[SyntheticAction] = []
        orders = data.get("orders", {})
        for order_id in sorted(orders):
            order = orders[order_id]
            if order.get("status") != "pending":
                continue
            user_id = order["user_id"]
            base_address = order.get("address", {})
            new_address = self._create_address_variant(base_address)
            action = Action(
                name="modify_pending_order_address",
                kwargs={"order_id": order_id, **new_address},
            )
            fragment = (
                f"For {order_id}, change address to {_format_address({'order_id': order_id, **new_address})}."
            )
            actions.append(
                SyntheticAction(action=action, user_id=user_id, prompt_fragment=fragment)
            )
        return actions

    def _generate_modify_user_address_actions(self, data: dict) -> List[SyntheticAction]:
        actions: List[SyntheticAction] = []
        users = data.get("users", {})
        for user_id in sorted(users):
            base_address = users[user_id].get("address", {})
            new_address = self._create_address_variant(base_address)
            action = Action(
                name="modify_user_address",
                kwargs={"user_id": user_id, **new_address},
            )
            fragment = (
                f"Update my address to {_format_address({'order_id': user_id, **new_address})}."
            )
            actions.append(
                SyntheticAction(action=action, user_id=user_id, prompt_fragment=fragment)
            )
        return actions

    # Internal helpers ---------------------------------------------------

    def _create_address_variant(self, base: dict) -> Address:
        city = base.get("city", "New York")
        state = base.get("state", "NY")
        country = base.get("country", "USA")
        zip_code = base.get("zip", "10001")
        house_number = self._rng.randint(100, 999)
        street = self._rng.choice([
            "Park Avenue",
            "Cedar Street",
            "Laurel Lane",
            "Maple Drive",
            "Oak Street",
            "Elm Avenue",
        ])
        address2 = f"Suite {self._rng.randint(100, 999)}"
        return {
            "address1": f"{house_number} {street}",
            "address2": address2,
            "city": city,
            "state": state,
            "country": country,
            "zip": zip_code,
        }


# ---------------------------------------------------------------------------
# Instruction synthesis
# ---------------------------------------------------------------------------


class InstructionBuilder:
    def __init__(self, base_data: dict, rng: random.Random) -> None:
        self._base_data = base_data
        self._rng = rng

    def build_instruction(self, user_id: str, fragments: Sequence[str]) -> str:
        user = self._base_data["users"][user_id]
        name = user["name"]["first_name"] + " " + user["name"]["last_name"]
        email = user.get("email")
        zip_code = user.get("address", {}).get("zip")

        identity_bits: List[str] = [f"Your name is {name}"]
        if email:
            identity_bits.append(f"your email is {email}")
        if zip_code:
            identity_bits.append(f"your zip code is {zip_code}")

        identity_sentence = " and ".join(identity_bits) + "."

        trait_count = min(4, max(2, math.ceil(self._rng.random() * 4)))
        traits = self._rng.sample(PERSONALITY_TRAITS, trait_count)
        trait_sentence = "You are " + ", ".join(traits) + "."

        action_sentence = " ".join(fragment.strip() for fragment in fragments)
        return f"{identity_sentence} {trait_sentence} {action_sentence}".strip()


# ---------------------------------------------------------------------------
# Synthetic task generator
# ---------------------------------------------------------------------------


class SyntheticTaskGenerator:
    def __init__(
        self,
        max_actions: int,
        max_sequences: int,
        seed: int,
    ) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._base_data = load_data()
        self._enumerator = ActionEnumerator(
            base_data=self._base_data,
            rng=self._rng,
        )
        self._instruction_builder = InstructionBuilder(self._base_data, rng=self._rng)
        self._max_actions = max_actions
        self._max_sequences = max_sequences

    # Public API ---------------------------------------------------------

    def generate_sequences(self) -> List[List[SyntheticAction]]:
        sequences: List[List[SyntheticAction]] = []
        seen_signatures: set[str] = set()
        stack: List[Tuple[dict, List[SyntheticAction], Optional[str]]] = [
            (self._base_data, [], None)
        ]

        while stack and len(sequences) < self._max_sequences:
            state, current_sequence, locked_user = stack.pop()

            if current_sequence:
                sig = "|".join(action.signature() for action in current_sequence)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    sequences.append(current_sequence)
                    if len(sequences) >= self._max_sequences:
                        break

            if len(current_sequence) >= self._max_actions:
                continue

            candidates = self._enumerator.enumerate(state)
            for synthetic_action in candidates:
                if locked_user is not None and synthetic_action.user_id != locked_user:
                    continue
                ok, new_state = try_action(state, synthetic_action.action)
                if not ok or new_state is None:
                    continue
                new_sequence = current_sequence + [synthetic_action]
                stack.append((new_state, new_sequence, synthetic_action.user_id if locked_user is None else locked_user))

        return sequences

    def build_tasks(self, num_tasks: int) -> List[Task]:
        sequences = self.generate_sequences()
        if not sequences:
            return []

        self._rng.shuffle(sequences)
        selected = sequences[:num_tasks]
        tasks: List[Task] = []
        for seq in selected:
            user_id = seq[0].user_id
            instruction = self._instruction_builder.build_instruction(
                user_id=user_id,
                fragments=[sa.prompt_fragment for sa in seq],
            )
            actions = [sa.action for sa in seq]
            task = Task(
                annotator="synthetic",
                user_id=user_id,
                instruction=instruction,
                actions=actions,
                outputs=[],
            )
            tasks.append(task)
        return tasks


# ---------------------------------------------------------------------------
# Public convenience helpers
# ---------------------------------------------------------------------------


def generate_tasks(
    num_tasks: int = 100,
    max_actions: int = 2,
    max_sequences: int = 200,
    seed: int = 0,
) -> List[Task]:
    """Generate a list of synthetic tasks.

    Parameters
    ----------
    num_tasks:
        Number of tasks to return.  The generator first enumerates
        ``max_sequences`` valid action sequences (up to ``max_actions``
        per sequence) and then samples ``num_tasks`` of them.
    max_actions:
        Maximum length of an action sequence.
    max_sequences:
        Upper bound on the number of unique sequences to consider before
        sampling the final tasks.
    seed:
        Random seed for reproducibility.
    """

    generator = SyntheticTaskGenerator(
        max_actions=max_actions,
        max_sequences=max_sequences,
        seed=seed,
    )
    return generator.build_tasks(num_tasks=num_tasks)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def _serialise_task(task: Task) -> dict:
    """Make a task JSON serialisable."""

    return {
        "annotator": "synthetic",
        "user_id": task.user_id,
        "instruction": task.instruction,
        "actions": [
            {"name": action.name, "kwargs": action.kwargs}
            for action in task.actions
        ],
        "outputs": task.outputs,
    }


def _indent_block(text: str, indent: int) -> str:
    prefix = " " * indent
    return "\n".join(prefix + line if line else "" for line in text.splitlines())


def _render_action(action: Action) -> str:
    return "\n".join(
        [
            "Action(",
            f"    name={repr(action.name)},",
            f"    kwargs={repr(action.kwargs)},",
            ")",
        ]
    )


def _render_task(task: Task) -> str:
    lines = [
        "Task(",
        "    annotator=\"synthetic\",",
        f"    user_id={repr(task.user_id)},",
        f"    instruction={repr(task.instruction)},",
    ]

    if task.actions:
        lines.append("    actions=[")
        for action in task.actions:
            lines.append(_indent_block(_render_action(action) + ",", 8))
        lines.append("    ],")
    else:
        lines.append("    actions=[],")

    lines.append("    outputs=[],")
    lines.append(")")
    return "\n".join(lines)


def write_python_module(
    tasks: List[Task],
    output_path: Path,
    variable_name: str = "TASKS_TRAIN",
) -> None:
    header = (
        "# Auto-generated by synthetic_gen_tasks.py; DO NOT EDIT.\n\n"
        "from .types import Task, Action\n\n\n"
        f"{variable_name} = [\n"
    )

    body_parts = []
    for task in tasks:
        body_parts.append(_indent_block(_render_task(task), 4))

    body = ",\n".join(body_parts)
    if body:
        body += ",\n"

    footer = "]\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header + body + footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Tau Retail tasks")
    parser.add_argument("--num-tasks", type=int, default=50, help="Number of tasks to output")
    parser.add_argument("--max-actions", type=int, default=2, help="Maximum action sequence length")
    parser.add_argument("--max-sequences", type=int, default=250, help="Maximum sequences to enumerate before sampling")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--output-py",
        type=Path,
        default=Path(__file__).with_name("tasks_train_gen.py"),
        help="Path to the Python module that will store the generated tasks",
    )
    parser.add_argument(
        "--variable-name",
        default="TASKS_TRAIN",
        help="Name of the task list variable inside the generated Python module",
    )
    parser.add_argument("--output-json", type=Path, help="Optional path to dump generated tasks as JSON")
    parser.add_argument("--preview", action="store_true", help="Print a short preview to stdout")

    args = parser.parse_args()

    tasks = generate_tasks(
        num_tasks=args.num_tasks,
        max_actions=args.max_actions,
        max_sequences=args.max_sequences,
        seed=args.seed,
    )

    if args.preview or not args.output_json:
        preview_count = min(3, len(tasks))
        print(f"Generated {len(tasks)} tasks. Previewing {preview_count} task(s):")
        for idx, task in enumerate(tasks[:preview_count]):
            print(f"\nTask {idx}: user={task.user_id}")
            print(f"Instruction: {task.instruction}")
            for action in task.actions:
                print(f"  - {action.name}: {action.kwargs}")

    write_python_module(
        tasks=tasks,
        output_path=args.output_py,
        variable_name=args.variable_name,
    )
    print(f"Saved {len(tasks)} task(s) to {args.output_py}")

    if args.output_json:
        payload = [_serialise_task(task) for task in tasks]
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"Saved JSON snapshot to {args.output_json}")


if __name__ == "__main__":
    main()
