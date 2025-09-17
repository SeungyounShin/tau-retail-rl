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

from verl.tools.tau_retail.config import TOOL_ERROR_REWARD
from verl.utils.reward_score import tau_retail
from verl.utils.rollout_trace import rollout_trace_op

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ReturnDeliveredOrderItems(BaseTool):
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
                "name": "return_delivered_order_items",
                "description": (
                    "Return some items of a delivered order. The order status will be changed to 'return requested'. "
                    "The agent needs to explain the return detail and ask for explicit user confirmation (yes/no) to proceed. "
                    "The user will receive follow-up email for how and where to return the item."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": (
                                "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id."
                            ),
                        },
                        "item_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "The item ids to be returned, each such as '1008292230'. There could be duplicate items in the list."
                            ),
                        },
                        "payment_method_id": {
                            "type": "string",
                            "description": (
                                "The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. "
                                "These can be looked up from the user or order details."
                            ),
                        },
                    },
                    "required": ["order_id", "item_ids", "payment_method_id"],
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
        data = kwargs.get("data")
        if not isinstance(data, dict) or not all(k in data for k in ("users", "orders", "products")):
            from verl.interactions.tau_retail_data import load_data
            data = load_data()
        products, orders, users = data["products"], data["orders"], data["users"]
        order_id : str = parameters.get("order_id", "")
        item_ids : list[str] = parameters.get("item_ids", [])
        payment_method_id : str = parameters.get("payment_method_id", "")

        # check order exists and is delivered
        if order_id not in orders:
            return "Error: order not found", 0.0 + TOOL_ERROR_REWARD, {}
        order = orders[order_id]
        if order["status"] != "delivered":
            return "Error: non-delivered order cannot be exchanged", 0.0 + TOOL_ERROR_REWARD, {}
        
        # Check if the payment method exists and is either the original payment method or a gift card
        if payment_method_id not in data["users"][order["user_id"]]["payment_methods"]:
            return "Error: payment method not found", 0.0 + TOOL_ERROR_REWARD, {}
        if (
            "gift_card" not in payment_method_id
            and payment_method_id != order["payment_history"][0]["payment_method_id"]
        ):
            return "Error: payment method should be either the original payment method or a gift card", 0.0 + TOOL_ERROR_REWARD, {}

        # Check if the items to be returned exist (there could be duplicate items in either list)
        all_item_ids = [item["item_id"] for item in order["items"]]
        for item_id in item_ids:
            if item_ids.count(item_id) > all_item_ids.count(item_id):
                return "Error: some item not found", 0.0 + TOOL_ERROR_REWARD, {}

        # modify the order
        order["status"] = "return requested"
        order["return_items"] = sorted(item_ids)
        order["return_payment_method_id"] = payment_method_id

        return json.dumps(order), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
