import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm

from tests.adapters import PackedSFTDataset  # make sure adapters.py exists with this class

def main():
    # === Config ===
    model_name_or_path = "./Qwen/Qwen2.5-0.5B/"
    output_dir = "./qwen_finetuned"
    seq_length = 512
    batch_size = 2 # (4) Increase batch size to reduce DataLoader overhead and make better use of GPU memory from 2 to 4
    gradient_accumulation_steps = 16
    num_epochs = 3
    learning_rate = 5e-5
    warmup_steps = 100
    logging_steps = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load Model & Tokenizer ===
    print("Loading model and tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        # torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        torch_dtype=torch.float16 # (2) Changed to improve speed significantly (for older GPUs…)
    )
    model = torch.nn.DataParallel(model) # (3) Wrap the model with DataParallel to leverage both GPUs
    model.to(device)

    # === Load Dataset ===
    print("Loading dataset…")
    train_dataset = PackedSFTDataset(tokenizer, "./data/ultrachat/train.jsonl", seq_length=seq_length, shuffle=True)
    val_dataset = PackedSFTDataset(tokenizer, "./data/ultrachat/test.jsonl", seq_length=seq_length, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # (1) Number of workers increases speed
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    # === Optimizer and Scheduler ===
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # === Training Loop ===
    print("Beginning training…")
    model.train()
    global_step = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0: # Update weights every ʻgradient_accumulation_stepsʻ batches
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % logging_steps == 0:
                    print(f"Epoch {epoch + 1}, Step {global_step}, Loss: {running_loss / logging_steps:.4f}")
                    running_loss = 0.0

        # === Validation Loop === –> Commented to increase speed
        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for batch in val_dataloader:
        #         input_ids = batch["input_ids"].to(device)
        #         labels = batch["labels"].to(device)
        #         outputs = model(input_ids=input_ids, labels=labels)
        #         val_loss += outputs.loss.item()
        # print(f"Validation loss after epoch {epoch + 1}: {val_loss / len(val_dataloader):.4f}")
        # model.train()

    # === Save Model and Tokenizer ===
    print("Saving model and tokenizer…")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved to", output_dir)

if __name__ == "__main__":
    main()
