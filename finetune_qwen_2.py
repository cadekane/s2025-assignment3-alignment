import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import datasets
from datasets import load_dataset
import json

def preprocess_function(examples, tokenizer, seq_length):
    """Process examples from the dataset for training."""
    # Tokenize inputs and labels
    model_inputs = {"input_ids": [], "labels": []}
    
    for i in range(len(examples["prompt"])):
        try:
            prompt = examples["prompt"][i]
            response = examples["response"][i]
            
            # Skip if either prompt or response is empty
            if not prompt or not response:
                continue
                
            # Tokenize prompt and response
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
            
            # Combine into one sequence
            concat_ids = prompt_tokens + response_tokens
            
            # Create labels: -100 for prompt (loss ignored), actual ids for response
            label_ids = [-100] * len(prompt_tokens) + response_tokens
            
            # Truncate if too long
            if len(concat_ids) > seq_length:
                concat_ids = concat_ids[:seq_length]
                label_ids = label_ids[:seq_length]
            
            # Pad if too short
            if len(concat_ids) < seq_length:
                padding_length = seq_length - len(concat_ids)
                concat_ids.extend([tokenizer.pad_token_id] * padding_length)
                label_ids.extend([-100] * padding_length)
            
            model_inputs["input_ids"].append(concat_ids)
            model_inputs["labels"].append(label_ids)
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue
    
    return model_inputs

def load_jsonl_dataset(file_path):
    """Load conversations from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                # Try to parse the JSON line
                item = json.loads(line)
                
                # Check if the item has the expected structure
                if "conversations" in item:
                    # If the file already has a "conversations" field, use it directly
                    data.append(item["conversations"])
                else:
                    # Otherwise, add the whole item
                    data.append(item)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_number}: {e}")
                continue
    return {"conversations": data}

def main():
    # === Config ===
    model_name_or_path = "./Qwen/Qwen2.5-0.5B/"
    output_dir = "./qwen_finetuned"
    seq_length = 512
    batch_size = 4  # Increased batch size
    gradient_accumulation_steps = 16
    num_epochs = 3
    learning_rate = 5e-5
    warmup_steps = 100
    logging_steps = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load Model & Tokenizer ===
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16  # Using float16 for better speed
    )
    
    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # === Load Dataset with Hugging Face datasets ===
    print("Loading dataset...")
    
    # Use Hugging Face's load_dataset for JSONL files
    try:
        print("Loading datasets with datasets.load_dataset...")
        dataset = datasets.load_dataset('json', 
                                      data_files={
                                          'train': "./data/ultrachat/train.jsonl",
                                          'validation': "./data/ultrachat/test.jsonl"
                                      })
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        print(f"Dataset loaded successfully. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        print(f"Dataset features: {train_dataset.features}")
        
        # Print a sample to verify data structure
        print(f"Sample data first row: {list(train_dataset[0].keys())}")
        if "prompt" not in train_dataset.features or "response" not in train_dataset.features:
            print("Warning: Expected 'prompt' and 'response' fields not found in dataset.")
            print(f"Available fields: {list(train_dataset.features.keys())}")
        
    except Exception as e:
        print(f"Failed to load with datasets.load_dataset: {e}")
        print("Falling back to manual loading...")
        
        # Manual loading as fallback
        data_train = []
        data_val = []
        
        # Load training data
        with open("./data/ultrachat/train.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    data_train.append(item)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in train file: {e}")
                    continue
        
        # Load validation data
        with open("./data/ultrachat/test.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    data_val.append(item)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in validation file: {e}")
                    continue
        
        # Convert to datasets
        train_dataset = datasets.Dataset.from_list(data_train)
        val_dataset = datasets.Dataset.from_list(data_val)
        
    # Print dataset info
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Process the datasets - disable multiprocessing to debug
    print("Processing training dataset...")
    train_tokenized = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, seq_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        batch_size=100,  # Process smaller batches
        num_proc=None  # Disable multiprocessing for debugging
    )
    
    print("Processing validation dataset...")
    val_tokenized = val_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, seq_length),
        batched=True,
        remove_columns=val_dataset.column_names,
        batch_size=100,
        num_proc=None
    )
    
    # Set format for PyTorch
    train_tokenized.set_format("torch")
    val_tokenized.set_format("torch")
    
    # Create DataLoaders
    train_dataloader = DataLoader(
        train_tokenized, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True  # Speed up data transfer to GPU
    )
    
    val_dataloader = DataLoader(
        val_tokenized, 
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )

    # === Optimizer and Scheduler ===
    # Use parameter groups to apply different learning rates if needed
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # === Training Loop ===
    print("Beginning training...")
    model.train()
    global_step = 0
    
    # Optional: Enable gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Optional: Use automatic mixed precision for faster training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                
            running_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Update progress bar with current loss
                progress_bar.set_postfix({"loss": running_loss / gradient_accumulation_steps})
                
                if global_step % logging_steps == 0:
                    print(f"Epoch {epoch + 1}, Step {global_step}, Loss: {running_loss / logging_steps:.4f}")
                    running_loss = 0.0

        # Add validation loop if needed (commented out for speed)
        # Uncomment for evaluation
        '''
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, labels=labels)
                val_loss += outputs.loss.item()
        print(f"Validation loss after epoch {epoch + 1}: {val_loss / len(val_dataloader):.4f}")
        model.train()
        '''

    # === Save Model and Tokenizer ===
    print("Saving model and tokenizer...")
    os.makedirs(output_dir, exist_ok=True)
    
    # If using DataParallel, save the wrapped model
    if isinstance(model, torch.nn.DataParallel):
        model.module.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir)
        
    tokenizer.save_pretrained(output_dir)
    print("Model saved to", output_dir)

if __name__ == "__main__":
    main()