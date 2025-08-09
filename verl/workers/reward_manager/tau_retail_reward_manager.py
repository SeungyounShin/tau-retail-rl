# verl/workers/reward_manager/tau_retail_reward_manager.py
from __future__ import annotations
import torch
from collections import defaultdict
from verl import DataProto
from verl.utils.reward_score import tau_retail
from verl.workers.reward_manager import register
from verl.workers.reward_manager.naive import NaiveRewardManager  # 재사용

@register("tau_retail")
class TauRetailRewardManager(NaiveRewardManager):
    """Tau-Retail 전용 RewardManager."""

    def __init__(self, tokenizer, num_examine=0):
        super().__init__(tokenizer, num_examine, compute_score=None)

    def compute_score(self, data_source, solution_str, ground_truth, extra_info):
        if data_source == "tau_retail":
            return tau_retail.compute_score(
                actions=solution_str,          
                ground_truth=ground_truth,
                data=extra_info.get("data"),
                raw_data=extra_info.get("raw_data"),
                method="strict",
                format_score=0.0,
                score=1.0,
            )
        return super().compute_score(data_source, solution_str, ground_truth, extra_info)
