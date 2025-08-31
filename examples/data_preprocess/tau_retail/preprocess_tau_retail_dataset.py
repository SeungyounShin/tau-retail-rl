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
python -m examples.data_preprocess.tau_retail.preprocess_tau_retail_dataset
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

    args = parser.parse_args()


    train_dataset_list, test_dataset_list = [], []
    data_source = "tau_retail"
    agent_name = "retail_agent"
    system_prompt = """# Retail Agent Policy

## General Rules

* At the start of the conversation, always **authenticate the user**: locate user id via email, or via name + zip code.
* Only one user can be assisted per conversation. Deny requests related to any other user.
* Before making any database-changing action (**cancel / exchange / return**), **list the action details and obtain explicit user confirmation (“yes”)**.
* Requests outside scope must be **transferred to a human agent**.
* All times are in **EST, 24-hour format**.
* [IMPORTANT] For any **cancel / exchange / return** request, the **order_id** must first be retrieved by calling `get_user_details`.
* [IMPORTANT] Users usually don’t know the order_id, so use the **user_id** to retrieve the **order_id** via `get_user_details`, then use the **order_id** to list the order contents with **`get_order_details`**.

## Order Status

* Possible statuses: `pending`, `processed`, `delivered`, `cancelled`.
* **Actions allowed:**
  * **Cancel** → if `pending`
  * **Exchange** → if `delivered`
  * **Return** → if `delivered`

---

## Cancel Pending Order

* **Condition:** order status must be `pending`.
* **Required confirmation:**
  * Order id
  * Reason: `"no longer needed"` or `"ordered by mistake"`
* **After confirmation:**
  * Status → `cancelled`
  * Refund → immediate if gift card, otherwise within **5–7 business days**.

---

## Exchange Delivered Order

* **Condition:** order status must be `delivered`.
* **Required confirmation:**
  * Order id
  * **All items to be exchanged** + new options (must stay within the same product, option change only)
  * Payment method for price difference (gift card must have sufficient balance)
* **After confirmation:**
  * Status → `exchange requested`
  * Customer receives return instructions via email
  * No need to place a new order

---

## Return Delivered Order

* **Condition:** order status must be `delivered`.
* **Required confirmation:**
  * Order id
  * **List of items to be returned**
  * Payment method to receive the refund (must be the **original payment method** or an **existing gift card**)
* **After confirmation:**
  * Status → `return requested`
  * Customer receives an email with instructions on how to return items
"""

    ALLOWED_FN = {"exchange_delivered_order_items", "cancel_pending_order", "return_delivered_order_items"}

    for split, tasks in [("train", TASKS_TRAIN), ("test", TASKS_TEST)]:
        for idx, task in tqdm(enumerate(tasks), total=len(tasks), desc=f"Processing `{split}` dataset"):
            gt_actions = [make_action_serializable(a) for a in task.actions]
            gt_actions_fn_name = [a.name for a in task.actions]
            # pass only exchange_delivered_order_items and cancel_pending_order and not return_delivered_order_items
            if set(gt_actions_fn_name).issubset(ALLOWED_FN):
                pass  # 허용된 액션만 있음
            else:
                # print(f"skip: {gt_actions_fn_name}")     # 다른 액션이 섞여 있음
                continue
            if len(gt_actions) == 0:
                # print(f"skip: {gt_actions_fn_name}")     # 다른 액션이 섞여 있음
                continue

            raw_data = load_data()
            turn_level_score = tau_retail.compute_score(
                [],
                gt_actions,
                data=cp.deepcopy(raw_data),
                raw_data=cp.deepcopy(raw_data),
                method="strict",
            )
            if turn_level_score == 1:
                continue

            data = {
                "data_source": data_source,
                "agent_name": agent_name,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    }
                ],
                "ability": "tau_retail",
                "reward_model": {"style": "rule", "ground_truth": gt_actions},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": "",
                    "question": task.instruction,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "find_user_id_by_email": {
                            "create_kwargs": {"ground_truth": gt_actions},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
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
                    },
                    "interaction_kwargs": {
                        "query": task.instruction,
                        "user_id": task.user_id,
                        "ground_truth": gt_actions
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

    train_dataset = Dataset.from_list(train_dataset_list)
    test_dataset = Dataset.from_list(test_dataset_list)
    # train_dataset = test_dataset
    # train_dataset = split["train"]
    # test_dataset = split["test"]
    
    print(test_dataset_list[0]['extra_info']['interaction_kwargs']['ground_truth'])
    print(test_dataset[0]['extra_info']['interaction_kwargs']['ground_truth'])

    print(f"train dataset len : {len(train_dataset)}")
    print(f"test dataset len : {len(test_dataset)}")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
