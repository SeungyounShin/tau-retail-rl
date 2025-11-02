from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "checkpoints/tau_retail_async_rl/qwen_tau_retail_multiturn_train_split/global_step_6/actor/huggingface"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


tools = [
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel a given order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description":
                        "The id of the order to cancel, e.g. '1234567890'",
                        "default": "1234567890",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_orders",
            "description": "Get the list of current user's orders",
            "parameters": {}
        },
    }
]

input_ids = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are autonomous retail agent. You can help users cancel orders or provide information about their own profile, orders, and related products."},
        {"role": "user", "content": "I want to cancel my order."},
    ],
    tokenize=True,
    return_tensors="pt",
    add_generation_prompt=True,
    tools=tools,
)

outputs = model.generate(
    input_ids, 
    max_new_tokens=1024, 
    do_sample=True, 
    temperature=0.6, 
    top_p=0.95,
    use_cache=True,
)

print(tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True))

model.push_to_hub("Seungyoun/Qwen3-4B-Thinking-tau-retail-rl-user-4B")
tokenizer.push_to_hub("Seungyoun/Qwen3-4B-Thinking-tau-retail-rl-user-4B")