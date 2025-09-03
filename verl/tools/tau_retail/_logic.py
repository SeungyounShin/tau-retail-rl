"""
Pure business‑logic helpers for the Tau Retail dataset.

Each helper is a **synchronous** function that takes the parsed `data` dict
(and any required keyword arguments) then returns the computed result.

There are **no** imports from tool / reward modules here, so circular
imports cannot arise.  Both the tool classes and the reward‑scorer can
simply `import ACTION_DISPATCH` (or the specific helpers) from this file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

############################
# Low‑level helper functions
############################

def _get_users(data: Dict[str, Any]):
    return data.get("users", {})


def _get_orders(data: Dict[str, Any]):
    return data.get("orders", {})


def _get_products(data: Dict[str, Any]):
    return data.get("products", {})


############################################
# User‑centric lookup helpers
############################################

def find_user_id_by_name_zip(
    data: Dict[str, Any], *, first_name: str, last_name: str, zip: str
) -> str | None:
    """Locate a user by first + last name and ZIP code (case‑insensitive)."""
    for user_id, profile in _get_users(data).items():
        if (
            profile["name"]["first_name"].lower() == first_name.lower()
            and profile["name"]["last_name"].lower() == last_name.lower()
            and profile["address"]["zip"] == zip
        ):
            return user_id
    return None


def find_user_id_by_email(data: Dict[str, Any], *, email: str) -> str | None:
    """Locate a user by e‑mail address (case‑insensitive)."""
    for user_id, profile in _get_users(data).items():
        if profile["email"].lower() == email.lower():
            return user_id
    return None


############################################
# Read‑only detail helpers
############################################

def get_order_details(data: Dict[str, Any], *, order_id: str) -> Dict[str, Any] | None:
    """Return a deep‑copy‑like view of a single order (if any)."""
    return _get_orders(data).get(order_id)


def get_user_details(data: Dict[str, Any], *, user_id: str) -> Dict[str, Any] | None:
    return _get_users(data).get(user_id)


def get_product_details(
    data: Dict[str, Any], *, product_id: str
) -> Dict[str, Any] | None:
    return _get_products(data).get(product_id)


########################################################
# Mutating helper – exchange delivered order items
########################################################

def exchange_delivered_order_items(
    data: Dict[str, Any],
    *,
    order_id: str,
    item_ids: List[str],
    new_item_ids: List[str],
    payment_method_id: str,
) -> Tuple[bool, str | Dict[str, Any]]:
    """Attempt to exchange *delivered* order items.

    Returns ``(success, payload)`` where:
      * *success* == ``True``  → payload is the **updated order dict**
      * *success* == ``False`` → payload is an **error message** str
    """

    orders = _get_orders(data)
    users = _get_users(data)
    products = _get_products(data)

    # 1️⃣ Order exists & delivered
    if order_id not in orders:
        return False, "Error: order not found"
    order = orders[order_id]
    if order["status"] != "delivered":
        return False, "Error: non‑delivered order cannot be exchanged"

    # 2️⃣ Validate item_id counts vs order
    original_item_ids = [item["item_id"] for item in order["items"]]
    for iid in item_ids:
        if item_ids.count(iid) > original_item_ids.count(iid):
            return False, f"Error: {iid} not found"

    # 3️⃣ Validate new items list length
    if len(item_ids) != len(new_item_ids):
        return False, "Error: the number of items to be exchanged should match"

    # 4️⃣ Calculate price diff and check variant availability
    diff_price = 0.0
    for old_iid, new_iid in zip(item_ids, new_item_ids):
        old_item = next(i for i in order["items"] if i["item_id"] == old_iid)
        product_id = old_item["product_id"]
        variant_map = products.get(product_id, {}).get("variants", {})
        if new_iid not in variant_map or not variant_map[new_iid]["available"]:
            return False, f"Error: new item {new_iid} not found or available"
        diff_price += variant_map[new_iid]["price"] - old_item["price"]

    diff_price = round(diff_price, 2)

    # 5️⃣ Payment‑method validation
    user_payment_methods = users[order["user_id"]]["payment_methods"]
    if payment_method_id not in user_payment_methods:
        return False, "Error: payment method not found"

    pm = user_payment_methods[payment_method_id]
    if pm["source"] == "gift_card" and pm["balance"] < diff_price:
        return False, "Error: insufficient gift card balance to pay for the price difference"

    # 6️⃣ Mutate order
    order.update(
        {
            "status": "exchange requested",
            "exchange_items": sorted(item_ids),
            "exchange_new_items": sorted(new_item_ids),
            "exchange_payment_method_id": payment_method_id,
            "exchange_price_difference": diff_price,
        }
    )

    return True, order


def cancel_pending_order(
    data: Dict[str, Any],
    *,
    order_id: str,
    reason: str,
) -> Tuple[bool, str | Dict[str, Any]]:
    """Attempt to cancel a pending order."""

    orders = data["orders"]
    if order_id not in orders:
        return False, "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return False, "Error: non-pending order cannot be cancelled"

    # check reason
    if reason not in ["no longer needed", "ordered by mistake"]:
        return False, "Error: invalid reason"

    # handle refund
    refunds = []
    for payment in order["payment_history"]:
        payment_id = payment["payment_method_id"]
        refund = {
            "transaction_type": "refund",
            "amount": payment["amount"],
            "payment_method_id": payment_id,
        }
        refunds.append(refund)
        if "gift_card" in payment_id:  # refund to gift card immediately
            payment_method = data["users"][order["user_id"]]["payment_methods"][
                payment_id
            ]
            payment_method["balance"] += payment["amount"]
            payment_method["balance"] = round(payment_method["balance"], 2)

    # update order status
    order["status"] = "cancelled"
    order["cancel_reason"] = reason
    order["payment_history"].extend(refunds)

    return True, order

def return_delivered_order_items(
    data: Dict[str, Any],
    *,
    order_id: str,
    item_ids: List[str],
    payment_method_id: str,
) -> Tuple[bool, str | Dict[str, Any]]:
    """Attempt to return delivered order items."""
    # debug in red
    # print(f"\033[91mreturn_delivered_order_items : {order_id}, {item_ids}, {payment_method_id}\033[0m")
    orders = data["orders"]

    # Check if the order exists and is delivered
    if order_id not in orders:
        return False, "Error: order not found"
    order = orders[order_id]
    if order["status"] != "delivered":
        return False, "Error: non-delivered order cannot be returned"

    # Check if the payment method exists and is either the original payment method or a gift card
    if payment_method_id not in data["users"][order["user_id"]]["payment_methods"]:
        return False, "Error: payment method not found"
    if (
        "gift_card" not in payment_method_id
        and payment_method_id != order["payment_history"][0]["payment_method_id"]
    ):
        return False, "Error: payment method should be either the original payment method or a gift card"

    # Check if the items to be returned exist (there could be duplicate items in either list)
    all_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            return False, "Error: some item not found"

    # update order status
    order["status"] = "return requested"
    order["return_items"] = sorted(item_ids)
    order["return_payment_method_id"] = payment_method_id

    return True, order

def list_all_product_types(data: Dict[str, Any]) -> Tuple[bool, str | Dict[str, Any]]:
    """List all product types."""
    products = data["products"]
    product_dict = {
        product["name"]: product["product_id"] for product in products.values()
    }
    product_dict = dict(sorted(product_dict.items()))
    return True, product_dict

def modify_pending_order_address(
    data: Dict[str, Any],
    *,
    order_id: str,
    address1: str,
    address2: str,
    city: str,
    state: str,
    country: str,
    zip: str,
) -> Tuple[bool, str | Dict[str, Any]]:
    """Attempt to modify a pending order address."""
    orders = data["orders"]
    if order_id not in orders:
        return False, "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return False, "Error: non-pending order cannot be modified"
    
    # update order address
    order["address"] = {
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "country": country,
        "zip": zip,
    }
    return True, order

def modify_pending_order_items(
    data: Dict[str, Any],
    *,
    order_id: str,
    item_ids: List[str],
    new_item_ids: List[str],
    payment_method_id: str,
) -> Tuple[bool, str | Dict[str, Any]]:
    """Attempt to modify a pending order items."""
    
    products, orders, users = data["products"], data["orders"], data["users"]

    # Check if the order exists and is pending
    if order_id not in orders:
        return False, "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return False, "Error: non-pending order cannot be modified"

    # Check if the items to be modified exist
    all_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            return False, f"Error: {item_id} not found"

    # Check new items exist, match old items, and are available
    if len(item_ids) != len(new_item_ids):
        return False, "Error: the number of items to be exchanged should match"

    diff_price = 0
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        item = [item for item in order["items"] if item["item_id"] == item_id][0]
        product_id = item["product_id"]
        if not (
            new_item_id in products[product_id]["variants"]
            and products[product_id]["variants"][new_item_id]["available"]
        ):
            return False, f"Error: new item {new_item_id} not found or available"

        old_price = item["price"]
        new_price = products[product_id]["variants"][new_item_id]["price"]
        diff_price += new_price - old_price

    # Check if the payment method exists
    if payment_method_id not in users[order["user_id"]]["payment_methods"]:
        return False, "Error: payment method not found"

    # If the new item is more expensive, check if the gift card has enough balance
    payment_method = users[order["user_id"]]["payment_methods"][payment_method_id]
    if (
        payment_method["source"] == "gift_card"
        and payment_method["balance"] < diff_price
    ):
        return False, "Error: insufficient gift card balance to pay for the new item"

    # Handle the payment or refund
    order["payment_history"].append(
        {
            "transaction_type": "payment" if diff_price > 0 else "refund",
            "amount": abs(diff_price),
            "payment_method_id": payment_method_id,
        }
    )
    if payment_method["source"] == "gift_card":
        payment_method["balance"] -= diff_price
        payment_method["balance"] = round(payment_method["balance"], 2)

    # Modify the order
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        item = [item for item in order["items"] if item["item_id"] == item_id][0]
        item["item_id"] = new_item_id
        item["price"] = products[item["product_id"]]["variants"][new_item_id][
            "price"
        ]
        item["options"] = products[item["product_id"]]["variants"][new_item_id][
            "options"
        ]
    order["status"] = "pending (item modified)"

    return True, order

def modify_pending_order_payment(
    data: Dict[str, Any],
    *,
    order_id: str,
    payment_method_id: str,
) -> Tuple[bool, str | Dict[str, Any]]:
    """Attempt to modify a pending order payment."""

    orders = data["orders"]

    # Check if the order exists and is pending
    if order_id not in orders:
        return False, "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return False, "Error: non-pending order cannot be modified"

    # Check if the payment method exists
    if payment_method_id not in data["users"][order["user_id"]]["payment_methods"]:
        return False, "Error: payment method not found"

    # Check that the payment history should only have one payment
    if (
        len(order["payment_history"]) > 1
        or order["payment_history"][0]["transaction_type"] != "payment"
    ):
        return False, "Error: there should be exactly one payment for a pending order"

    # Check that the payment method is different
    if order["payment_history"][0]["payment_method_id"] == payment_method_id:
        return (
            False,
            "Error: the new payment method should be different from the current one"
        )
    
    amount = order["payment_history"][0]["amount"]
    payment_method = data["users"][order["user_id"]]["payment_methods"][
        payment_method_id
    ]

    # Check if the new payment method has enough balance if it is a gift card
    if (
        payment_method["source"] == "gift_card"
        and payment_method["balance"] < amount
    ):
        return False, "Error: insufficient gift card balance to pay for the order"

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

    return True, order

def modify_user_address(
    data: Dict[str, Any],
    *,
    user_id: str,
    address1: str,
    address2: str,
    city: str,
    state: str,
    country: str,
    zip: str,
) -> Tuple[bool, str | Dict[str, Any]]:
    """Attempt to modify a user address."""
    users = data["users"]
    if user_id not in users:
        return False, "Error: user not found"
    user = users[user_id]
    user["address"] = {
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "country": country,
        "zip": zip,
    }
    return True, user

########################
# Public dispatch table
########################

ACTION_DISPATCH = {
    "find_user_id_by_name_zip": find_user_id_by_name_zip,
    "find_user_id_by_email": find_user_id_by_email,
    "get_order_details": get_order_details,
    "get_user_details": get_user_details,
    "get_product_details": get_product_details,
    "exchange_delivered_order_items": exchange_delivered_order_items,
    "cancel_pending_order": cancel_pending_order,
    "return_delivered_order_items": return_delivered_order_items,
    "list_all_product_types": list_all_product_types,
    "modify_pending_order_address": modify_pending_order_address,
    "modify_pending_order_items": modify_pending_order_items,
    "modify_pending_order_payment": modify_pending_order_payment,
}
