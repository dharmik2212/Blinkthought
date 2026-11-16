import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_decision_model(model_id="Qwen/Qwen2.5-1.5B-Instruct"):
    print(f"Loading tokenizer and model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Define reserved tokens
    reserved_tokens = [f"<opt_{i}>" for i in range(256)] + ["<yes>", "<no>"] + [f"<lvl_{i}>" for i in range(10)] + ["<readout>"]
    
    # Add tokens to tokenizer
    num_added = tokenizer.add_tokens(reserved_tokens)
    print(f"Added {num_added} reserved tokens to tokenizer.")
    
    # Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Initialize new token embeddings from letter tokens ('A', 'B', 'C' etc.) + small noise
    embeddings = model.get_input_embeddings().weight
    lm_head = model.get_output_embeddings().weight if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None else None
    
    with torch.no_grad():
        base_token_id = tokenizer.encode("A", add_special_tokens=False)[0]
        base_emb = embeddings[base_token_id].clone()
        
        for token in reserved_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            noise = torch.randn_like(base_emb) * 0.01
            embeddings[token_id] = base_emb + noise
            if lm_head is not None and lm_head.shape[0] == embeddings.shape[0]:
                lm_head[token_id] = lm_head[base_token_id] + noise
                
    print("Initialized reserved token embeddings from base token 'A' with small noise.")
    return model, tokenizer

def forward_pass_sanity_test():
    model, tokenizer = setup_decision_model()
    
    state = "Ticket #1001: Customer reports billing issue."
    options = {"payments": "Billing", "support_tech": "Tech"}
    
    # Construct prompt layout
    prompt = f"State: {state}\nInstructions: Route ticket.\n"
    for k, v in options.items():
        prompt += f"- {v} <opt_{list(options.keys()).index(k)}>\n"
    prompt += "<readout>"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :] # last token position (<readout>)
        
    # Extract logits for option tokens
    opt_ids = [tokenizer.convert_tokens_to_ids(f"<opt_{i}>") for i in range(len(options))]
    opt_logits = logits[0, opt_ids]
    probs = torch.softmax(opt_logits, dim=-1)
    
    print("Forward pass sanity test successful!")
    print(f"Sanity Option Probabilities: {dict(zip(options.keys(), probs.tolist()))}")

if __name__ == "__main__":
    forward_pass_sanity_test()
