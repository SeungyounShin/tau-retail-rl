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

import logging
import os
from typing import Any, Optional
from uuid import uuid4
import copy

from litellm import completion

from verl.utils.reward_score import tau_retail

from .base import BaseInteraction
from .tau_retail_data import load_data

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TauRetailInteraction(BaseInteraction):
    """Interaction for tau retail.

    - `start_interaction`: start a interaction instance for a trajectory.
    - `generate_response`: generate the response of the user.
    - `calculate_score`: calculate the score of the interaction.
    - `finalize_interaction`: finalize the interaction instance.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self.model = config.get("model", "gpt-4o")
        self.provider = config.get("provider", None)
        self.total_cost = 0.0

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
            "data": load_data(),
        }
        self.raw_data = load_data()
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        messages = self.swap_roles_and_replace_system(messages, instruction=kwargs.get("query", ""))
        # print in gray
        # print(f"\033[90m{messages}\033[0m")
        res = completion(
            model=self.model, custom_llm_provider=self.provider, messages=messages,
        )
        message = res.choices[0].message
        response = message.model_dump()['content']
        self.total_cost = res._hidden_params["response_cost"]
        # print in green
        # print(f"\033[92m -> {response}\033[0m")
        self._instance_dict[instance_id]["response"] = response

        reward = await self.calculate_score(instance_id)
        should_terminate_sequence = False
        if "###STOP###" in response:
            should_terminate_sequence = True

        return should_terminate_sequence, response, reward, {}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        return tau_retail.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"],
            data=copy.deepcopy(self._instance_dict[instance_id]["data"]),
            raw_data=copy.deepcopy(self.raw_data),
            method="strict",
            format_score=0.0,
            score=1.0,
        )

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

    def get_data(self, instance_id: str) -> dict[str, Any]:
        return self._instance_dict.get(instance_id, {}).get("data", {})


    def swap_roles_and_replace_system(
        self,
        messages: list[dict[str, Any]],
        instruction: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """
        1. Replace the existing system prompt with the output of `self.build_system_prompt`.
        2. Swap `assistant` ↔︎ `user` roles for the rest of the turns.
        (Leave any other roles—e.g., `tool`, `function`—untouched.)
        """

        # Build the new system prompt
        new_messages: list[dict[str, str]] = [
            {"role": "system", "content": self.build_system_prompt(instruction)}
        ]

        # Process the remaining turns
        for msg in messages:
            role = msg["role"]

            # Skip the original system message(s)
            if role == "system":
                continue

            # Swap assistant ↔︎ user, keep others as-is
            if role == "assistant":
                role = "user"
            elif role == "user":
                role = "assistant" 
            elif role == "tool":
                continue
            
            new_messages.append({"role": role, "content": msg["content"]})

        return new_messages


    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return (
            "You are a user interacting with an agent." f"{instruction_display}"
            "Rules:\n"
            "- Just generate one line at a time to simulate the user's message.\n"
            "- Do not give away all the instruction at once. Only provide the "
            "information that is necessary for the current step.\n"
            "- Do not hallucinate information that is not provided in the "
            "instruction. For example, if the agent asks for the order id but "
            "it is not mentioned in the instruction, do not make up an order id,"
            " just say you do not remember or have it.\n"
            "- If the instruction goal is satisified, generate '###STOP###' as "
            "a standalone message without anything else to end the conversation.\n"
            "- Do not repeat the exact instruction in the conversation. Instead,"
            " use your own words to convey the same information.\n"
            "- Try to make the conversation as natural as possible, and stick to"
            " the personalities in the instruction."
        )
