import os
import csv
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tests.adapters import run_parse_mmlu_response

# === (1) Load MMLU Examples ===
def load_mmlu_data(data_dir):
    examples = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            subject = filename.replace(".csv", "")
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 6:
                        continue  # skip malformed lines
                    examples.append({
                        "subject": subject,
                        "question": row[0].strip(),
                        "choices": [row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip()],
                        "answer": row[5].strip()
                    })
    return examples

# === (2) Format Prompt ===
def format_prompt(example):
    prompt = (
        f"Answer the following multiple choice question about {example['subject']}. "
        f'Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).\n\n'
    )
    prompt += f"Question: {example['question']}\n"
    for idx, letter in enumerate(["A", "B", "C", "D"]):
        prompt += f"{letter}. {example['choices'][idx]}\n"
    prompt += "Answer:"
    return prompt

# === (3) Generate Batched Model Outputs ===
def generate_batched_answers(model, tokenizer, prompts, device):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded

# === (4) Compute Accuracy ===
def compute_accuracy(preds, refs):
    return sum(p == r for p, r in zip(preds, refs)) / len(preds)

# === (5) Save Results ===
def save_results(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# === MAIN ===
def main():
    # --- Config ---
    mmlu_path = "./data/mmlu/test"
    model_path = "./finetuned_qwen/Qwen2.5-1.5B-Instruct"
    output_path = "./finetuned_outputs/qwen_mmlu_results.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)

    print("Loading MMLU dataset...")
    examples = load_mmlu_data(mmlu_path)

    predictions, references = [], []
    output_data = []

    print("Running batched model inference...")
    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i:i + batch_size]
        prompts = [format_prompt(ex) for ex in batch]
        outputs = generate_batched_answers(model, tokenizer, prompts, device)

        for ex, full_output, prompt in zip(batch, outputs, prompts):
            gen_text = full_output[len(prompt):].strip()
            pred = run_parse_mmlu_response(ex, gen_text)
            label = ex["answer"]

            predictions.append(pred)
            references.append(label)

            output_data.append({
                "subject": ex["subject"],
                "question": ex["question"],
                "choices": ex["choices"],
                "label": label,
                "generation": gen_text,
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
