import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_baseline():
    print("Loading model for baseline evaluation...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    test_examples = []
    with open("test.jsonl", "r", encoding='utf-8') as f:
        for line in f:
            test_examples.append(json.loads(line))
            
    correct = 0
    total = 0
    
    print(f"Evaluating baseline on {len(test_examples)} test examples...")
    for idx, ex in enumerate(test_examples):
        state = ex["state"]
        for q in ex["questions"]:
            if q["type"] == "choice":
                options_str = "\n".join([f"- {k}: {v}" for k, v in q["options"].items()])
                prompt = f"State: {state}\nQuestion: {q['instructions']}\nOptions:\n{options_str}\nAnswer with only the option key (e.g. payments):"
                
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
                
                # Best target option
                best_target = max(q["target"], key=q["target"].get)
                
                if best_target in response:
                    correct += 1
                total += 1
                
    accuracy = correct / total if total > 0 else 0
    print(f"Baseline Prompted-LLM Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    evaluate_baseline()
