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


class GetProductDetails(BaseTool):
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
                "name": "get_product_details",
                "description": "Get the inventory details of a product.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": "The product id, such as '6086499569'. Be careful the product id is different from the item id.",
                        },
                    },
                    "required": ["product_id"],
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
        products = data["products"]
        product_id = parameters.get("product_id", "")
        if not data or "products" not in data:
            return "Error: data is not provided", 0.0, {}
        products = data.get("products", {})
        # print(f"products: {products}")
        if product_id in products:
            return json.dumps(products[product_id]), 0.0, {}
        return "Error: product not found", 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
