import os
import csv
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tests.adapters import run_parse_gsm8k_response

# === (1) Load GSM8K Examples === JSON file load in
def load_gsm8k_data(path_to_json):
    with open(path_to_json, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

# === (2) Format Prompt ===
def format_prompt(example):
    prompt = f"{example['question']}\n"
    prompt += "Answer:" 
    return prompt

# === (3) Generate Model Output ===
def generate_answer(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()  # Strip prompt from the start

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
    gsm8k_path = "./data/gsm8k/test.jsonl"
    model_path = "./Qwen/Qwen2.5-1.5B/"
    output_path = "./finetuned_outputs/qwen_gsm8k_results.json"
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

    print("Loading GSM8K dataset...")
    examples = load_gsm8k_data(gsm8k_path)

    predictions, references = [], []
    output_data = []

    print("Running model inference...")
    for example in tqdm(examples):
        prompt = format_prompt(example)
        output = generate_answer(model, tokenizer, prompt, device)
        pred = run_parse_gsm8k_response(output) # imported function from adapters.py
        label = run_parse_gsm8k_response(example["answer"])  # needed to parse this as well!!! since it's not just the answer but has text with it

        predictions.append(pred)
        references.append(label)

        output_data.append({
            "question": example["question"], # There are no choices in this dataset
            "label": label,
            "generation": output,
            "prediction": pred,
            "correct": pred == label
        })

    accuracy = compute_accuracy(predictions, references)
    print(f"Overall Accuracy: {accuracy:.2%}")

    result_bundle = {
        "accuracy": accuracy,
        "examples": output_data
    }

    print(f"Saving results to {output_path}")
    save_results(result_bundle, output_path)

if __name__ == "__main__":
    main()
