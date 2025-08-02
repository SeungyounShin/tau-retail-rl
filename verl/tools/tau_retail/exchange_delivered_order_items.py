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


class ExchangeDeliveredOrderItems(BaseTool):
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
                "name": "exchange_delivered_order_items",
                "description": (
                    "Exchange items in a delivered order to new items of the same product type. "
                    "For a delivered order, return or exchange can be only done once by the agent. "
                    "The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                        },
                        "item_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                            "description": "The item ids to be exchanged, each such as '1008292230'. There could be duplicate items in the list.",
                        },
                        "new_item_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                            "description": (
                                "The item ids to be exchanged for, each such as '1008292230'. "
                                "There could be duplicate items in the list. Each new item id should match the item id in the same position and be of the same product."
                            ),
                        },
                        "payment_method_id": {
                            "type": "string",
                            "description": (
                                "The payment method id to pay or receive refund for the item price difference, "
                                "such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details."
                            ),
                        },
                    },
                    "required": [
                        "order_id",
                        "item_ids",
                        "new_item_ids",
                        "payment_method_id",
                    ],
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
        products, orders, users = data["products"], data["orders"], data["users"]
        order_id : str = parameters.get("order_id", "")
        item_ids : list[str] = parameters.get("item_ids", [])
        new_item_ids : list[str] = parameters.get("new_item_ids", [])
        payment_method_id : str = parameters.get("payment_method_id", "")

        # check order exists and is delivered
        if order_id not in orders:
            return "Error: order not found", 0.0, {}
        order = orders[order_id]
        if order["status"] != "delivered":
            return "Error: non-delivered order cannot be exchanged", 0.0, {}
        
        # check the items to be exchanged exist
        all_item_ids = [item["item_id"] for item in order["items"]]
        for item_id in item_ids:
            if item_ids.count(item_id) > all_item_ids.count(item_id):
                return f"Error: {item_id} not found", 0.0, {}

        # check new items exist and match old items and are available
        if len(item_ids) != len(new_item_ids):
            return "Error: the number of items to be exchanged should match", 0.0, {}

        diff_price = 0
        for item_id, new_item_id in zip(item_ids, new_item_ids):
            item = [item for item in order["items"] if item["item_id"] == item_id][0]
            product_id = item["product_id"]
            if not (
                new_item_id in products[product_id]["variants"]
                and products[product_id]["variants"][new_item_id]["available"]
            ):
                return f"Error: new item {new_item_id} not found or available", 0.0, {}

            old_price = item["price"]
            new_price = products[product_id]["variants"][new_item_id]["price"]
            diff_price += new_price - old_price

        diff_price = round(diff_price, 2)

        # check payment method exists and can cover the price difference if gift card
        if payment_method_id not in users[order["user_id"]]["payment_methods"]:
            return "Error: payment method not found", 0.0, {}

        payment_method = users[order["user_id"]]["payment_methods"][payment_method_id]
        if (
            payment_method["source"] == "gift_card"
            and payment_method["balance"] < diff_price
        ):
            return (
                "Error: insufficient gift card balance to pay for the price difference"
            ), 0.0, {}

        # modify the order
        order["status"] = "exchange requested"
        order["exchange_items"] = sorted(item_ids)
        order["exchange_new_items"] = sorted(new_item_ids)
        order["exchange_payment_method_id"] = payment_method_id
        order["exchange_price_difference"] = diff_price

        return json.dumps(order), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return tau_retail.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"],
            method="flexible",
            format_score=0.0,
            score=1.0,
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
