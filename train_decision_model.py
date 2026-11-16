import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class DecisionDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.examples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        state = ex["state"]
        questions = ex["questions"]
        
        # Pick one random question per example during training (or process all)
        q = random.choice(questions)
        
        # Shuffle options
        options = list(q["options"].items())
        random.shuffle(options)
        
        # Build prompt
        prompt = f"State: {state}\nInstructions: {q['instructions']}\n"
        option_token_map = {}
        for i, (opt_key, opt_desc) in enumerate(options):
            token_str = f"<opt_{i}>" if q["type"] == "choice" else ("<yes>" if opt_key == "yes" else "<no>" if opt_key == "no" else f"<lvl_{opt_key}>")
            prompt += f"- {opt_desc} {token_str}\n"
            option_token_map[opt_key] = token_str
            
        prompt += "<readout>"
        
        # Target distribution aligned with shuffled option order
        target_dist = []
        for opt_key, _ in options:
            target_dist.append(q["target"].get(opt_key, 0.0))
        target_tensor = torch.tensor(target_dist, dtype=torch.float)
        
        # Option token IDs
        opt_token_ids = [self.tokenizer.convert_tokens_to_ids(option_token_map[opt_key]) for opt_key, _ in options]
        
        encoding = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "opt_token_ids": opt_token_ids,
            "target": target_tensor,
            "q_type": q["type"]
        }

def collate_fn(batch):
    # Since lengths vary, we pad input_ids
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), 0, dtype=torch.long)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros((pad_len,), dtype=torch.long)]))
        
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "batch_items": batch
    }

def train_model():
    print("Initializing training on Kaggle GPU...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    
    reserved_tokens = [f"<opt_{i}>" for i in range(256)] + ["<yes>", "<no>"] + [f"<lvl_{i}>" for i in range(10)] + ["<readout>"]
    tokenizer.add_tokens(reserved_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # LoRA setup
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    dataset = DecisionDataset("train.jsonl", tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    
    print("Starting training loop...")
    for epoch in range(1):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits # (batch, seq_len, vocab_size)
            
            loss = 0.0
            for i, item in enumerate(batch["batch_items"]):
                readout_pos = len(item["input_ids"]) - 1
                opt_ids = item["opt_token_ids"]
                target = item["target"].to(model.device)
                
                # logits for option tokens at readout position
                q_logits = logits[i, readout_pos, opt_ids]
                probs = torch.log_softmax(q_logits, dim=-1)
                
                # Cross entropy loss against soft target distribution
                ce_loss = -(target * probs).sum()
                loss += ce_loss
                
            loss = loss / len(batch["batch_items"])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch} finished. Avg Loss: {total_loss / len(dataloader):.4f}")
        
    model.save_pretrained("jev_decision_model")
    tokenizer.save_pretrained("jev_decision_model")
    print("Training complete. Model saved to jev_decision_model.")

if __name__ == "__main__":
    train_model()
