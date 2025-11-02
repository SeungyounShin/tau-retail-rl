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
from ..schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ModifyPendingOrderPayment(BaseTool):
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
                "name": "modify_pending_order_payment",
                "description": "Modify the payment method of a pending order. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                        },
                        "payment_method_id": {
                            "type": "string",
                            "description": "The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details.",
                        },
                    },
                    "required": [
                        "order_id",
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

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        data = kwargs.get("data", {})
        orders = data["orders"]

        order_id : str = parameters.get("order_id", "")
        payment_method_id : str = parameters.get("payment_method_id", "")

        if order_id not in orders:
            return ToolResponse(text="Error: order not found"), 0.0, {}
        order = orders[order_id]
        if order["status"] != "pending":
            return ToolResponse(text="Error: non-pending order cannot be modified"), 0.0, {}
        
        # Check if the payment method exists
        if payment_method_id not in data["users"][order["user_id"]]["payment_methods"]:
            return ToolResponse(text="Error: payment method not found"), 0.0, {}
        
        # Check that the payment history should only have one payment
        if (
            len(order["payment_history"]) > 1
            or order["payment_history"][0]["transaction_type"] != "payment"
        ):
            return ToolResponse(text="Error: there should be exactly one payment for a pending order"), 0.0, {}
        
        # Check that the payment method is different
        if order["payment_history"][0]["payment_method_id"] == payment_method_id:
            return (
                ToolResponse(text="Error: the new payment method should be different from the current one")
            ), 0.0, {}
        
        amount = order["payment_history"][0]["amount"]
        payment_method = data["users"][order["user_id"]]["payment_methods"][
            payment_method_id
        ]

        # Check if the new payment method has enough balance if it is a gift card
        if (
            payment_method["source"] == "gift_card"
            and payment_method["balance"] < amount
        ):
            return ToolResponse(text="Error: insufficient gift card balance to pay for the order"), 0.0, {}

        # Modify the payment method
        order["payment_history"].extend(
            [
                {
                    "transaction_type": "payment",
                    "amount": amount,
                    "payment_method_id": payment_method_id,
                },
                {
                    "transaction_type": "refund",
                    "amount": amount,
                    "payment_method_id": order["payment_history"][0][
                        "payment_method_id"
                    ],
                },
            ]
        )

        # If payment is made by gift card, update the balance
        if payment_method["source"] == "gift_card":
            payment_method["balance"] -= amount
            payment_method["balance"] = round(payment_method["balance"], 2)

        # If refund is made to a gift card, update the balance
        if "gift_card" in order["payment_history"][0]["payment_method_id"]:
            old_payment_method = data["users"][order["user_id"]]["payment_methods"][
                order["payment_history"][0]["payment_method_id"]
            ]
            old_payment_method["balance"] += amount
            old_payment_method["balance"] = round(old_payment_method["balance"], 2)
        
        return ToolResponse(text=json.dumps(order)), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
