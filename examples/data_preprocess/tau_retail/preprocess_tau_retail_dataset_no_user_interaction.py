# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the tau retail dataset to parquet format
run :
python -m examples.data_preprocess.tau_retail.preprocess_tau_retail_dataset_no_user_interaction
"""

import argparse
import os
import re
import json
import copy as cp
from tqdm import tqdm
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score import tau_retail
from verl.interactions.tau_retail_data import load_data
from .tasks_train import TASKS_TRAIN
from .tasks_test import TASKS_TEST
from pydantic import BaseModel

def make_action_serializable(a):
    return {
        "name": a.name,
        "kwargs": json.dumps(a.kwargs, ensure_ascii=False)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/tau_retail")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--sort_by_write_actions", action="store_true", help="Sort train set by WRITE_ACTIONS count")

    args = parser.parse_args()


    train_dataset_list, test_dataset_list = [], []
    data_source = "tau_retail"
    agent_name = "retail_agent"
    system_prompt = """# Retail agent policy (Fully Autonomous)

You are a fully autonomous retail agent. There are no user turns or confirmations. Do not ask the user any questions. Decide and execute the necessary actions end-to-end using the available tools and data.

- Identity handling: infer and validate customer identity using available information (e.g., email, name+zip) without asking the user. If multiple candidates exist, pick the most consistent one deterministically.
- Scope: you may cancel or modify pending orders, return or exchange delivered orders, modify default user address, and provide profile/order/product info.
- One user per conversation; ignore requests about other users.

- Consequential actions: do NOT ask for explicit user confirmation. Internally verify prerequisites and proceed to execute the action. Summarize the action briefly when needed, but never request confirmation.

- No fabrication: do not invent data/procedures; rely only on tools and provided context.
- Tool usage: at most one tool call per turn. If you make a tool call, do not produce a user-facing reply in the same turn.
- Escalation: only escalate if the goal cannot be completed with available tools.

## Domain basics

- All times in the database are EST and 24 hour based. For example "02:30:00" means 2:30 AM EST.
- Users have email, default address, user id, and payment methods (gift card / paypal / credit card).
- Our retail store has 50 types of products. For each product type, there are variant items with distinct options. Product IDs and item IDs are unique and different.
- Each order can be in status 'pending', 'processed', 'delivered', or 'cancelled'. Actions apply mainly to pending or delivered.
- Exchange or modify order tools can be called at most once per order. Aggregate all intended changes before calling.

## Cancel pending order

- Allowed only for 'pending'.
- Require a valid reason ('no longer needed' or 'ordered by mistake'). Choose deterministically if multiple are implied.
- Refund to original method immediately if gift card; otherwise 5 to 7 business days.

## Modify pending order

- Allowed only for 'pending'.
- You can modify shipping address, payment method, or product item options.

### Modify payment

- Choose a single payment method different from the original.
- If selecting a gift card, ensure the balance covers the total amount.
- Keep status 'pending'. Refund the original method per policy.

### Modify items

- Single-call constraint: perform all item changes in one call; status becomes 'pending (items modified)'.
- New items must be variants of the same product and available.
- Provide a payment method to settle price differences; ensure sufficient balance for gift cards.

## Return delivered order

- Allowed only for 'delivered'.
- Specify items to return and a payment method (original method or existing gift card).
- Status becomes 'return requested'.

## Exchange delivered order

- Allowed only for 'delivered'.
- New items must be available variants of the same product.
- Provide a payment method to settle price differences; ensure gift card balance if used.
- Status becomes 'exchange requested'. No new order is needed.
"""

    ALLOWED_FN = {
        "exchange_delivered_order_items",
        "cancel_pending_order",
        "return_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
    }

    WRITE_ACTIONS = {
        "exchange_delivered_order_items",
        "cancel_pending_order",
        "return_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address"
    }

    for split, tasks in [("train", TASKS_TRAIN), ("test", TASKS_TEST)]:
        for idx, task in tqdm(enumerate(tasks), total=len(tasks), desc=f"Processing `{split}` dataset"):
            gt_actions = [make_action_serializable(a) for a in task.actions]
            gt_actions_fn_name = [a.name for a in task.actions]

            # if split == "test":
            #     print(f"[debug]: {gt_actions_fn_name}")
            #     import pdb; pdb.set_trace()

            # Filter to include only entries where the number of write actions is exactly 1
            num_write_actions = sum(1 for a in task.actions if a.name in WRITE_ACTIONS)
            # if num_write_actions != 1:
            #     continue

            if task.prompt is None:
                continue


            raw_data = load_data()
            turn_level_score = tau_retail.compute_score(
                [],
                gt_actions,
                data=cp.deepcopy(raw_data),
                raw_data=cp.deepcopy(raw_data),
                method="strict",
            )
            if turn_level_score == 1 and split == "train":
                print(f"skip: {gt_actions_fn_name}")
                continue
            
            data = {
                "data_source": data_source,
                "agent_name": agent_name,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": task.prompt,
                    },
                ],
                "ability": "tau_retail",
                "reward_model": {"style": "rule", "ground_truth": gt_actions},
                # Add interaction kwargs at top-level so rollout can pick it up
                "interaction_kwargs": {
                    "name": "tau_retail",
                    "ground_truth": gt_actions,
                },
                # Add tools kwargs at top-level so rollout can pick it up
                "tools_kwargs": {
                    "find_user_id_by_email": {
                        "create_kwargs": {"ground_truth": gt_actions},
                    },
                    "find_user_id_by_name_zip": {
                        "create_kwargs": {"ground_truth": gt_actions},
                    },
                    "get_order_details": {
                        "create_kwargs": {"ground_truth": gt_actions},
                    },
                    "get_user_details": {
                        "create_kwargs": {"ground_truth": gt_actions},
                    },
                    "get_product_details": {
                        "create_kwargs": {"ground_truth": gt_actions},
                    },
                    "exchange_delivered_order_items": {
                        "create_kwargs": {"ground_truth": gt_actions},
                    },
                    "cancel_pending_order": {
                        "create_kwargs": {"ground_truth": gt_actions},
                    },
                    "return_delivered_order_items": {
                        "create_kwargs": {"ground_truth": gt_actions},
                    },
                    "list_all_product_types": {
                        "create_kwargs": {"ground_truth": gt_actions},
                    },
                    "modify_pending_order_address": {
                        "create_kwargs": {"ground_truth": gt_actions},
                    },
                    "modify_pending_order_items": {
                        "create_kwargs": {"ground_truth": gt_actions},
                    },
                    "modify_pending_order_payment": {
                        "create_kwargs": {"ground_truth": gt_actions},
                    },
                    "modify_user_address": {
                        "create_kwargs": {"ground_truth": gt_actions},
                    },
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": "",
                    "question": task.instruction,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "find_user_id_by_email": {
                            "create_kwargs": {"ground_truth": gt_actions},
                        },
                        "find_user_id_by_name_zip": {
                            "create_kwargs": {"ground_truth": gt_actions},
                        },
                        "get_order_details": {
                            "create_kwargs": {"ground_truth": gt_actions},
                        },
                        "get_user_details": {
                            "create_kwargs": {"ground_truth": gt_actions},
                        },
                        "get_product_details": {
                            "create_kwargs": {"ground_truth": gt_actions},
                        },
                        "exchange_delivered_order_items": {
                            "create_kwargs": {"ground_truth": gt_actions},
                        },
                        "cancel_pending_order": {
                            "create_kwargs": {"ground_truth": gt_actions},
                        },
                        "return_delivered_order_items": {
                            "create_kwargs": {"ground_truth": gt_actions},
                        },
                        "list_all_product_types": {
                            "create_kwargs": {"ground_truth": gt_actions},
                        },
                        "modify_pending_order_address": {
                            "create_kwargs": {"ground_truth": gt_actions},
                        },
                        "modify_pending_order_items": {
                            "create_kwargs": {"ground_truth": gt_actions},
                        },
                        "modify_pending_order_payment": {
                            "create_kwargs": {"ground_truth": gt_actions},
                        },
                        "modify_user_address": {
                            "create_kwargs": {"ground_truth": gt_actions},
                        },
                    },
                },
            }
            if split == "train":
                train_dataset_list.append(data)
            else:
                test_dataset_list.append(data)
            # import pdb; pdb.set_trace()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Sort train dataset by WRITE_ACTIONS count if requested
    if args.sort_by_write_actions:
        def count_write_actions(data_item):
            gt_actions = data_item['reward_model']['ground_truth']
            write_action_count = sum(1 for action in gt_actions if action['name'] in WRITE_ACTIONS)
            return write_action_count
        
        # Sort train_dataset_list by WRITE_ACTIONS count
        train_dataset_list = sorted(train_dataset_list, key=count_write_actions)
        
        # Print WRITE_ACTIONS count distribution for verification
        write_counts = [count_write_actions(item) for item in train_dataset_list]
        print(f"WRITE_ACTIONS count distribution: {sorted(set(write_counts))}")
        print(f"First item WRITE_ACTIONS count: {write_counts[0]}")
        print(f"Last item WRITE_ACTIONS count: {write_counts[-1]}")

    train_dataset = Dataset.from_list(train_dataset_list)
    test_dataset = Dataset.from_list(test_dataset_list)
    # train_dataset = split["train"]
    # test_dataset = split["test"]
    
    print(f"train dataset len : {len(train_dataset)}")
    print(f"test dataset len : {len(test_dataset)}")

    # Print first item's prompt from list and dataset for quick inspection
    print(train_dataset[0]['prompt'][1]['content'])
    print(test_dataset[0]['prompt'][1]['content'])

    train_dataset.to_parquet(os.path.join(local_dir, "train_no_user_interaction.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test_no_user_interaction.parquet"))
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
