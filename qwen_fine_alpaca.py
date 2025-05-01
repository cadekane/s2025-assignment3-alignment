import os
import csv
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === (1) Load Alpaca Examples === JSON file load in
def load_alpaca_data(path_to_json):
    with open(path_to_json, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

# === (2) Format Prompt ===
def format_prompt(example):
    prompt = f"{example['instruction']}"
    return prompt

# === (3) Generate Model Output ===
# def generate_answer(model, tokenizer, prompt, device):
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=10)
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return decoded[len(prompt):].strip()  # Strip prompt from the start

def generate_answer(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            # do_sample=False,       # <- greedy decoding (was not described in instructions)
            temperature=0.0,       # <- optional (ignored with do_sample=False)
            top_p=1.0              # <- optional (ignored with do_sample=False)
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()

# === (4) Extract & Evaluate ===

# Parsing function provided from adapters.py

def compute_accuracy(preds, refs):
    return sum(p == r for p, r in zip(preds, refs)) / len(preds) # A basic average

# === (5) Save Results ===
def save_results(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# === MAIN ===
def main():
    # --- Config ---
    alpaca_path = "./data/alpaca_eval/alpaca_eval.jsonl"
    model_path = "./Qwen/Qwen2.5-1.5B/"
    output_path = "./finetuned_outputs/qwen_alpaca_results.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)

    try:
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return  # or exit the script gracefully

    print("Loading AlpacaEval dataset...")
    examples = load_alpaca_data(alpaca_path)

    output_data = []

    print("Running model inference...")
    for example in tqdm(examples):
        prompt = format_prompt(example)
        output = generate_answer(model, tokenizer, prompt, device)

        example["output"] = output
        example["generator"] = "qwen_small"

        # output_data.append({
        #     "instruction": example["instruction"],
        #     "output": output,
        #     "generator": "qwen_small",
        #     "dataset": example["dataset"]
        # })

    print(f"Saving results to {output_path}")
    save_results(examples, output_path)

if __name__ == "__main__":
    main()
