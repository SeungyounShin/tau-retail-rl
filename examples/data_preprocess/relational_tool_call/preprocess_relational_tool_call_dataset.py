"""
Preprocess the tau retail dataset to parquet format
run :
python -m examples.data_preprocess.relational_tool_call.preprocess_relational_tool_call_dataset
"""

import argparse
import os
import re
import json
import copy as cp
from tqdm import tqdm
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs
from .tasks_train import EXAMPLES_TRAIN
from .tasks_test import EXAMPLES_TEST
from pydantic import BaseModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/relational_tool_call")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    train_dataset_list, test_dataset_list = [], []
    data_source = "relational_tool_call"
    agent_name = "relational_tool_call_agent"
    system_prompt = "Please answer the user's question by using the tools provided. Do not guess the answer. Keep in mind that entities like users,foods and locations have both a name and an ID, which are not the same."

    FUNCTION_NAMES = [
        "get_user_name",
        "list_user_ids",
        "find_users_by_name",
        "find_locations_by_name",
        "find_foods_by_name",
        "get_user_email",
        "get_user_location",
        "get_user_favorite_color",
        "get_user_favorite_foods",
        "get_weather_at_location",
        "get_city_for_location",
        "get_current_time_for_location",
        "get_current_weather_for_location",
        "get_food_name",
        "get_food_calories",
        "get_food_allergic_ingredients",
        "get_current_user_id",
    ]

    for split, tasks in [("train", EXAMPLES_TRAIN), ("test", EXAMPLES_TEST)]:
        for idx, task in tqdm(enumerate(tasks), total=len(tasks), desc=f"Processing `{split}` dataset"):
            
            question, answer = task

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
                        "content": question,
                    },
                ],
                "ability": "relational_tool_call",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        **{f"{fn_name}": {
                            "create_kwargs": {"ground_truth": answer},
                        } for fn_name in FUNCTION_NAMES},
                    },
                }
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

    print(test_dataset_list[0]['extra_info']['question'])
    print(test_dataset[0]['extra_info']['question'])

    print(f"train dataset len : {len(train_dataset)}")
    print(f"test dataset len : {len(test_dataset)}")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        
