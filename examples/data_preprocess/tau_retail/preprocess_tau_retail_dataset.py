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
    system_prompt = """# Retail agent policy

As a retail agent, you can help users cancel or modify pending orders, return or exchange delivered orders, modify their default user address, or provide information about their own profile, orders, and related products.

- At the beginning of the conversation, you have to authenticate the user identity by locating their user id via email, or via name + zip code. This has to be done even when the user already provides the user id.

- Once the user has been authenticated, you can provide the user with information about order, product, profile information, e.g. help the user look up order id.

- You can only help one user per conversation (but you can handle multiple requests from the same user), and must deny any requests for tasks related to any other user.

- Before taking consequential actions that update the database (cancel, modify, return, exchange), you have to list the action detail and obtain explicit user confirmation (yes) to proceed.

- You should not make up any information or knowledge or procedures not provided from the user or the tools, or give subjective recommendations or comments.

- You should at most make one tool call at a time, and if you take a tool call, you should not respond to the user at the same time. If you respond to the user, you should not make a tool call.

- You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions.

## Domain basic

- All times in the database are EST and 24 hour based. For example "02:30:00" means 2:30 AM EST.

- Each user has a profile of its email, default address, user id, and payment methods. Each payment method is either a gift card, a paypal account, or a credit card.

- Our retail store has 50 types of products. For each type of product, there are variant items of different options. For example, for a 't shirt' product, there could be an item with option 'color blue size M', and another item with option 'color red size L'.

- Each product has an unique product id, and each item has an unique item id. They have no relations and should not be confused.

- Each order can be in status 'pending', 'processed', 'delivered', or 'cancelled'. Generally, you can only take action on pending or delivered orders.

- Exchange or modify order tools can only be called once. Be sure that all items to be changed are collected into a list before making the tool call!!!

## Cancel pending order

- An order can only be cancelled if its status is 'pending', and you should check its status before taking the action.

- The user needs to confirm the order id and the reason (either 'no longer needed' or 'ordered by mistake') for cancellation.

- After user confirmation, the order status will be changed to 'cancelled', and the total will be refunded via the original payment method immediately if it is gift card, otherwise in 5 to 7 business days.

## Modify pending order

- An order can only be modified if its status is 'pending', and you should check its status before taking the action.

- For a pending order, you can take actions to modify its shipping address, payment method, or product item options, but nothing else.

## Return delivered order

- An order can only be returned if its status is 'delivered', and you should check its status before taking the action.

- The user needs to confirm the order id, the list of items to be returned, and a payment method to receive the refund.

- The refund must either go to the original payment method, or an existing gift card.

- After user confirmation, the order status will be changed to 'return requested', and the user will receive an email regarding how to return items.

## Exchange delivered order

- An order can only be exchanged if its status is 'delivered', and you should check its status before taking the action. In particular, remember to remind the customer to confirm they have provided all items to be exchanged.

- For a delivered order, each item can be exchanged to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.

- The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.

- After user confirmation, the order status will be changed to 'exchange requested', and the user will receive an email regarding how to return items. There is no need to place a new order.
"""

    ALLOWED_FN = {"exchange_delivered_order_items", "cancel_pending_order", "return_delivered_order_items", "modify_pending_order_address"}

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
                        "list_all_product_types": {
                            "create_kwargs": {"ground_truth": gt_actions},
                        },
                        "modify_pending_order_address": {
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
