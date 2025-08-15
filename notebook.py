# %% 
# Celda 1: imports
import torch
from unsloth import FastLanguageModel
# %%
MODEL = "unsloth/gemma-3-270m-it"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=2048,
    dtype="float32",
    device="cpu",
    load_in_4bit=False,
    full_finetuning=False
)
