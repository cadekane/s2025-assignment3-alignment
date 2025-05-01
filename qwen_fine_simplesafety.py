import os
import csv
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === (1) Load SimpleSafety Examples === CSV file load-in
def load_simplesafety_data(path_to_csv):
    examples = []
    with open(path_to_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            examples.append({
                "id": row[0].strip(),
                "harm_area": row[1].strip(),
                "counter": row[2].strip(),
                "category": row[3].strip(),
                "prompts_final": row[4].strip()
            })
    return examples

# === (2) Format Prompt ===
def format_prompt(example):
    prompt = f"{example['prompts_final']}"
    return prompt

# === (3) Generate Model Output ===
def generate_answer(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=75, temperature=0.0, top_p=1.0) # Set to 75 instead of 100 to save more time :D
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()  # Strip prompt from the start

# === (4) Extract & Evaluate ===

# === (5) Save Results ===
# def save_results(data, path):
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2)

def save_results(data, path): # serializes it in JSON lines format instead
    with open(path, "w", encoding="utf-8") as f:
        for example in data:
            f.write(json.dumps(example) + "\n")

# === MAIN ===
def main():
    # --- Config ---
    simplesafety_path = "./data/simple_safety_tests/simple_safety_tests.csv"
    model_path = "./Qwen/Qwen2.5-1.5B/"
    output_path = "./finetuned_outputs/qwen_simplesafety_results.json"
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

    print("Loading SimpleSafety dataset...")
    examples = load_simplesafety_data(simplesafety_path)

    output_data = []

    print("Running model inference...")
    for example in tqdm(examples):
        prompt = format_prompt(example)
        output = generate_answer(model, tokenizer, prompt, device)

        example["output"] = output
        # each example already has the prompts_final key – see the load_simplesafety_data function

    print(f"Saving results to {output_path}")
    save_results(examples, output_path)

if __name__ == "__main__":
    main()
