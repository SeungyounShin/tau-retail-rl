from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

model_name_original = "Qwen/Qwen3-4B-Thinking-2507"
model_name = "/home/robin/tau-retail-rl/checkpoints/tau_retail_async_rl/qwen_tau_retail_multiturn_train_split/global_step_30/actor/huggingface"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "Get temperature at a location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "date": {
                        "type": "string",
                        "description": 'The date to get the temperature for, in the format "Year-Month-Day".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location", "date"],
            },
        },
    },
]

prompt = "What is current temperature in Seoul?"
tool_call = {"name": "get_current_temperature", "arguments": {"location": "Seoul, South Korea", "unit": "celsius"}}
tool_call_id = "vAHdf3"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt},
    #{"role": "assistant", "tool_calls": [{"id": tool_call_id, "type": "function", "function": tool_call}]},
    #{"role": "tool", "tool_call_id": tool_call_id, "name": "get_current_temperature", "content": "22.0"}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=TOOLS
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

prompt_all = tokenizer.batch_decode(model_inputs.input_ids, skip_special_tokens=False)[0]
# in green
print(f"\033[92m{prompt_all}\033[0m")
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
print(response)

# import pdb; pdb.set_trace()
# model = model.merge_and_unload()

model.push_to_hub("Seungyoun/Qwen3-4B-Thinking-tau-retail-rl-no-user-interaction")
tokenizer.push_to_hub("Seungyoun/Qwen3-4B-Thinking-tau-retail-rl-no-user-interaction")