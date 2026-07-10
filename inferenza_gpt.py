from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Scegli il modello — es. small (117M)
MODEL = "openai-community/gpt2-xl"
# oppure: gpt2-medium, gpt2-large, gpt2-xl

tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
model = GPT2LMHeadModel.from_pretrained(MODEL)
model.eval()

prompt = "Hi bro tell me a horror story"
inputs = tokenizer(prompt, return_tensors="pt")

# --- Modalità 1: random sampling (temperatura 1, no truncation)
# Corrisponde ai file "small-117M.train.jsonl"
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

# --- Modalità 2: Top-K 40
# Corrisponde ai file "small-117M-k40.train.jsonl"
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        top_k=40,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))