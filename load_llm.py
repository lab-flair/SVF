import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"         

from transformers import AutoTokenizer, LlamaForCausalLM
import torch

model_name = "meta-llama/Llama-2-7b-chat-hf"
local_path = "./models/llama2"

print(f"Loading LLaMA2-7B-Chat from {model_name} ...")

token = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=local_path,
    token=token,              
    trust_remote_code=False
)

model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=local_path,
    token=token,              
    trust_remote_code=False
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model and tokenizer loaded successfully.")
