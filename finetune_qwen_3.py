import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm

class UltraChatDataset(Dataset):
    """Custom PyTorch Dataset for loading UltraChat data."""
    
    def __init__(self, file_path, tokenizer, seq_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Load data from file
        self._load_data(file_path)
    
    def _load_data(self, file_path):
        """Load and preprocess data from JSONL file."""
        print(f"Loading data from {file_path}")
        valid_examples = 0
        skipped_examples = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    # Parse JSON
                    example = json.loads(line)
                    
                    # Extract conversations based on format
                    conversation = None
                    if "conversations" in example:
                        conversation = example["conversations"]
                    elif isinstance(example, list):
                        conversation = example
                    
                    if conversation is None or not isinstance(conversation, list) or len(conversation) == 0:
                        skipped_examples += 1
                        continue
                    
                    # Process conversation into input_ids and labels
                    input_ids, labels = self._process_conversation(conversation)
                    
                    if input_ids is not None:
                        self.examples.append({
                            "input_ids": input_ids,
                            "labels": labels
                        })
                        valid_examples += 1
                        
                        # Print example of the first valid conversation for debugging
                        if valid_examples == 1:
                            print("First valid conversation format:")
                            print(f"Raw conversation: {conversation[:2]}...")
                            print(f"Processed input_ids shape: {input_ids.shape}")
                            print(f"Processed labels shape: {labels.shape}")
                    else:
                        skipped_examples += 1
                        
                except Exception as e:
                    skipped_examples += 1
                    if line_num < 5:  # Only print errors for the first few lines
                        print(f"Error processing line {line_num}: {e}")
        
        print(f"Loaded {valid_examples} valid examples, skipped {skipped_examples} examples")
    
    def _process_conversation(self, conversation):
        """Process a conversation into input_ids and labels."""
        concat_ids = []
        label_ids = []
        
        # Process each turn in the conversation
        for i, turn in enumerate(conversation):
            # Handle different conversation formats
            text = turn
            if isinstance(turn, dict) and "value" in turn:
                text = turn["value"]
            elif isinstance(turn, dict) and "content" in turn:
                text = turn["content"]
                
            if not isinstance(text, str):
                continue
                
            # Tokenize the turn
            turn_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Add to concatenated conversation
            concat_ids.extend(turn_tokens)
            
            # For labels: -100 for inputs (user), actual ids for responses (assistant)
            if i % 2 == 0:  # User turns (typically even indices)
                label_ids.extend([-100] * len(turn_tokens))
            else:  # Assistant responses (typically odd indices)
                label_ids.extend(turn_tokens)
        
        # Skip if empty
        if len(concat_ids) == 0:
            return None, None
            
        # Truncate if too long
        if len(concat_ids) > self.seq_length:
            concat_ids = concat_ids[:self.seq_length]
            label_ids = label_ids[:self.seq_length]
        
        # Pad if too short
        if len(concat_ids) < self.seq_length:
            padding_length = self.seq_length - len(concat_ids)
            concat_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            label_ids.extend([-100] * padding_length)
        
        return torch.tensor(concat_ids), torch.tensor(label_ids)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

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

    # === Load Dataset using our custom Dataset class ===
    print("Loading dataset...")
    train_dataset = UltraChatDataset(
        file_path="./data/ultrachat/train.jsonl",
        tokenizer=tokenizer,
        seq_length=seq_length
    )
    
    val_dataset = UltraChatDataset(
        file_path="./data/ultrachat/test.jsonl",
        tokenizer=tokenizer,
        seq_length=seq_length
    )
    
    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,  # Reduced for stability
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True
    )

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
                    loss = outputs.loss / gradient_accumulation_steps if isinstance(model, torch.nn.DataParallel) else outputs.loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps if isinstance(model, torch.nn.DataParallel) else outputs.loss / gradient_accumulation_steps
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

        # Add validation loop if needed
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