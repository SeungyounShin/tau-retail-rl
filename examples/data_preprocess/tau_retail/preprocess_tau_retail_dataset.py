"""
Preprocess the tau retail dataset to parquet format
run :
python -m examples.data_preprocess.tau_retail.preprocess_tau_retail_dataset

python -m examples.data_preprocess.tau_retail.preprocess_tau_retail_dataset --train_action_counts 3
"""

import argparse
import os
import re
import json 
import random
import copy as cp
from tqdm import tqdm
from datasets import Dataset
from collections import Counter

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score import tau_retail
from verl.interactions.tau_retail_data import load_data
from .tasks_train import TASKS_TRAIN
from .tasks_test import TASKS_TEST
from .tasks_train_new import tasks as TASKS_TRAIN_NEW
from .types import Task, Action
from pydantic import BaseModel

def make_action_serializable(a):
    if isinstance(a, Action):
        return {
                "name": a.name,
                "kwargs": json.dumps(a.kwargs, ensure_ascii=False)
            }
    elif isinstance(a, dict):
        return a
    else:
        raise ValueError(f"Invalid action type: {type(a)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/tau_retail")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--train_action_counts",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Filter train split to examples whose number of actions matches the"
            " provided counts. Only tasks composed of actions in ALLOWED_FN are"
            " eligible for filtering."
        ),
    )

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

### Modify payment

- The user can only choose a single payment method different from the original payment method.

- If the user wants the modify the payment method to gift card, it must have enough balance to cover the total amount.

- After user confirmation, the order status will be kept 'pending'. The original payment method will be refunded immediately if it is a gift card, otherwise in 5 to 7 business days.

### Modify items

- This action can only be called once, and will change the order status to 'pending (items modifed)', and the agent will not be able to modify or cancel the order anymore. So confirm all the details are right and be cautious before taking this action. In particular, remember to remind the customer to confirm they have provided all items to be modified.

- For a pending order, each item can be modified to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.

- The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.

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
        "modify_user_address",
    }

    for split, tasks in [("train", TASKS_TRAIN_NEW), ("test", TASKS_TEST)]:
        for idx, task in tqdm(enumerate(tasks), total=len(tasks), desc=f"Processing `{split}` dataset"):
            if isinstance(task, Task):
                gt_actions = [make_action_serializable(a) for a in task.actions]
                gt_actions_fn_name = [a.name for a in task.actions]
            elif isinstance(task, dict):
                gt_actions = [make_action_serializable(a) for a in task['actions']]
                gt_actions_fn_name = [a['name'] for a in task['actions']]
            else:
                raise ValueError(f"Invalid task type: {type(task)}")
            # pass only exchange_delivered_order_items and cancel_pending_order and not return_delivered_order_items
            # if set(gt_actions_fn_name).issubset(ALLOWED_FN):
            #     pass  # 허용된 액션만 있음
            # else:
            #     # print(f"skip: {gt_actions_fn_name}")     # 다른 액션이 섞여 있음
            #     continue
            # if len(gt_actions) == 0:
            #     # print(f"skip: {gt_actions_fn_name}")     # 다른 액션이 섞여 있음
            #     continue

            # raw_data = load_data()
            # turn_level_score = tau_retail.compute_score(
            #     [],
            #     gt_actions,
            #     data=cp.deepcopy(raw_data),
            #     raw_data=cp.deepcopy(raw_data),
            #     method="strict",
            # )
            # if turn_level_score == 1:
            #     print(f"skip: {gt_actions_fn_name}")
            #     continue
            
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
                    "question": task.instruction if isinstance(task, Task) else task['instruction'],
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
                    "interaction_kwargs": {
                        "query": task.instruction if isinstance(task, Task) else task['instruction'],
                        "user_id": task.user_id if isinstance(task, Task) else task['user_id'],
                        "ground_truth": gt_actions
                    },
                },
            }
            if split == "train":
                if args.train_action_counts is not None:
                    if not set(gt_actions_fn_name).issubset(ALLOWED_FN):
                        continue
                    if len(gt_actions_fn_name) not in args.train_action_counts:
                        continue
                train_dataset_list.append(data)
            else:
                test_dataset_list.append(data)
            # import pdb; pdb.set_trace()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    random.shuffle(train_dataset_list)
    train_dataset = Dataset.from_list(train_dataset_list[:512])
    # random select 10 (shuffle first) 
    test_dataset = Dataset.from_list(test_dataset_list)
    # train_dataset = test_dataset
    # train_dataset = split["train"]
    # test_dataset = split["test"]
    
    # WRITE_ACTIONS 개수 계산 함수
    def count_write_actions(ground_truth_actions):
        """ground_truth actions 중 WRITE_ACTIONS의 개수를 반환"""
        count = 0
        for action in ground_truth_actions:
            action_name = action.get('name', '')
            if action_name in WRITE_ACTIONS:
                count += 1
        return count
    
    def print_histogram(dataset, dataset_name):
        """Dataset의 WRITE_ACTIONS 분포를 histogram으로 출력"""
        # 각 샘플에서 WRITE_ACTIONS 개수 추출
        write_action_counts = []
        for item in dataset:
            gt_actions = item['extra_info']['interaction_kwargs']['ground_truth']
            write_count = count_write_actions(gt_actions)
            write_action_counts.append(write_count)
        
        # 분포 계산
        distribution = Counter(write_action_counts)
        max_count = max(distribution.values()) if distribution else 0
        
        print(f"\n{'='*70}")
        print(f"{dataset_name} - WRITE_ACTIONS 개수별 분포")
        print(f"{'='*70}")
        print(f"총 샘플 수: {len(dataset)}\n")
        
        for action_count in sorted(distribution.keys()):
            sample_count = distribution[action_count]
            percentage = (sample_count / len(dataset)) * 100
            # Histogram bar 생성 (최대 40개의 '█'로 스케일)
            bar_length = int((sample_count / max_count) * 40)
            bar = '█' * bar_length
            print(f"  {action_count}개: {sample_count:4d}개 ({percentage:5.1f}%) │{bar}")
        print()
    
    # Train과 Test dataset 모두 출력
    print_histogram(train_dataset, "Train Dataset")
    print_histogram(test_dataset, "Test Dataset")

    print(f"train dataset len : {len(train_dataset)}")
    print(f"test dataset len : {len(test_dataset)}")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
