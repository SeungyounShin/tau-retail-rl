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

from verl.utils.reward_score import tau_retail
from verl.utils.rollout_trace import rollout_trace_op

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def _norm(s: Any) -> str:
    # Treat None / non-strings as empty, strip spaces, case-insensitive
    return (s or "").strip().casefold()

class FindUserIdByNameZip(BaseTool):
    """Tool for retrieving a user's ID from the Tau Retail dataset.
    - ``get_openai_tool_schema``: return the tool schema in OpenAI format.
    - ``create``: create a tool instance for a trajectory.
    - ``execute``: execute the tool and return ``(response, reward, metrics)``.
    - ``calc_reward``: calculate the reward with respect to tool state.
    - ``release``: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "find_user_id_by_name_zip",
                "description": (
                    "Find user id by first name, last name, and zip code. If the user is not found, the function "
                    "will return an error message. By default, find user id by email, and only call this function "
                    "if the user is not found by email or cannot remember email."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "first_name": {
                            "type": "string",
                            "description": "The first name of the customer, such as 'John'.",
                        },
                        "last_name": {
                            "type": "string",
                            "description": "The last name of the customer, such as 'Doe'.",
                        },
                        "zip": {
                            "type": "string",
                            "description": "The zip code of the customer, such as '12345'.",
                        },
                    },
                    "required": ["first_name", "last_name", "zip"],
                },
            },
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        data = kwargs.get("data", {})
        if not isinstance(data, dict) or "users" not in data:
            return "Error: data is not provided", 0.0, {}

        first_name_in = _norm(parameters.get("first_name"))
        last_name_in  = _norm(parameters.get("last_name"))
        zip_in        = str(parameters.get("zip") or "").strip()

        users = data.get("users") or {}
        for user_id, profile in users.items():
            name = (profile.get("name") or {})
            addr = (profile.get("address") or {})

            p_first = _norm(name.get("first_name"))
            p_last  = _norm(name.get("last_name"))
            p_zip   = str(addr.get("zip") or "").strip()  # zip might be int in data

            if p_first == first_name_in and p_last == last_name_in and p_zip == zip_in:
                return user_id, 0.0, {}

        return "Error: user not found", 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
