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
import json

from verl.utils.reward_score import tau_retail
from verl.utils.rollout_trace import rollout_trace_op

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class GetUserDetails(BaseTool):
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
                "name": "get_user_details",
                "description": "Get the details of a user, including their orders.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The user id, such as 'sara_doe_496'.",
                        },
                    },
                    "required": ["user_id"],
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
        data = kwargs.get("data") or {}
        if "users" not in data:
            return "Error: data.users is missing", 0.0, {}

        users = data["users"]

        # Extract the user_id from parameters; accept either a string or an object with 'id'/'user_id'
        raw_user_id = parameters.get("user_id", None)
        if raw_user_id is None:
            return "Error: parameter 'user_id' is required", 0.0, {}

        if isinstance(raw_user_id, dict):
            raw_user_id = raw_user_id.get("id") or raw_user_id.get("user_id")
            if raw_user_id is None:
                return "Error: parameter 'user_id' must be a string or an object with 'id'/'user_id'", 0.0, {}

        user_id = str(raw_user_id)

        # Support either a dict mapping {user_id: user_obj} or a list of user objects with an id field
        if isinstance(users, dict):
            user = users.get(user_id)
            if user is None:
                return "Error: user not found", 0.0, {}
            return json.dumps(user), 0.0, {}

        if isinstance(users, list):
            for u in users:
                if isinstance(u, dict) and str(u.get("id") or u.get("user_id")) == user_id:
                    return json.dumps(u), 0.0, {}
            return "Error: user not found", 0.0, {}

        return "Error: data.users must be a dict or a list", 0.0, {}


    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
