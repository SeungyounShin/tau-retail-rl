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


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


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
        "You must use the `find_user_id_by_email` tool to find the user id before continuing."
        "Serve only that customer, follow policy exactly (no hallucination, one tool call at a time, no human transfer unless impossible)."
    )

    for split, tasks in [("train", TASKS_TRAIN), ("test", TASKS_TEST)]:
        for idx, task in enumerate(tasks):
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
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": "",
                    "question": task.instruction,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "find_user_id_by_email": {
                            "create_kwargs": {"ground_truth": ""},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                    },
                    "interaction_kwargs": {
                        "query": task.instruction,
                        "user_id": task.user_id,
                        "ground_truth": "",
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
    test_dataset = Dataset.from_list([test_dataset_list[0]])
    print(f"train dataset len : {len(train_dataset)}")
    print(f"test dataset len : {len(test_dataset)}")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
