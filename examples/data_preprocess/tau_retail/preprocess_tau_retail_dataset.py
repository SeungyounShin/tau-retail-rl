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

from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs
from .tasks_train import TASKS_TRAIN
from .tasks_test import TASKS_TEST
from pydantic import BaseModel

def to_serializable(obj):
    """Arrow 가 이해할 수 있는 타입으로 변환"""
    if isinstance(obj, BaseModel):        # Pydantic v1
        return {k: to_serializable(v) for k, v in obj.dict().items()}
        # v2 일 땐 obj.model_dump() 써도 됩니다.
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/tau_retail")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()


    train_dataset_list, test_dataset_list = [], []
    data_source = "tau_retail"
    agent_name = "retail_agent"
    system_prompt = (
        "You are an online-retail customer-service agent."
        "Always authenticate the customer (email or name + ZIP) before continuing, and for any change (cancel, modify, return, exchange) list the details and proceed only after the customer explicitly says “yes.”"
        "You must use the use tools (find_user_id_by_email or find_user_id_by_name_zip) to find the user id before continuing."
        "After finding the user id, you must use the get_order_details tool to get the order details to retrieve the user's orders."
        "Then you can specify the product in interest to the customer and use the get_product_details tool to get the product details."
        "You can retrieve payment method details from the get_user_details tool."
        "If the customer wants to exchange the product, you must use the exchange_delivered_order_items tool to exchange the product."
        "Serve only that customer, follow policy exactly (no hallucination, one tool call at a time, no human transfer unless impossible)."
    )

    for split, tasks in [("train", TASKS_TRAIN), ("test", TASKS_TEST)]:
        for idx, task in enumerate(tasks):
            gt_actions = [to_serializable(a) for a in task.actions]
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

    train_dataset = Dataset.from_list(train_dataset_list[:256])
    test_dataset = Dataset.from_list([test_dataset_list[0]])
    print(f"train dataset len : {len(train_dataset)}")
    print(f"test dataset len : {len(test_dataset)}")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
