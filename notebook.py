# %% [markdown]
# # Notebook config

# %%
import os

LOAD_CHECKPOINT = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# %% [markdown]
# # Imports

# %%
# Install Pytorch & other libraries
%pip install --upgrade torch tensorboard

# Install Gemma release branch from Hugging Face
%pip install --upgrade transformers

# Install Hugging Face libraries
%pip install  --upgrade \
  datasets \
  accelerate \
  evaluate \
  bitsandbytes \
  trl \
  peft \
  protobuf \
  sentencepiece \
 huggingface_hub

# %% [markdown]
# # HuggingFace Auth

# %%
#from google.colab import userdata
from huggingface_hub import login

# Login into Hugging Face Hub
#hf_token = userdata.get('HF_TOKEN')
hf_token = os.environ.get('HF_TOKEN_ENV')
login(hf_token)

# %% [markdown]
# # Prepare dataset

# %%
from datasets import load_dataset

system_message = """You are a Chess Bot. Users will present you the chess context and you will provide the solution to the problem given"""
user_prompt = """Given the <USER_QUERY> and <CONTEXT>, generate the corresponding answer.

<CONTEXT>
{context}
</CONTEXT>

<USER_QUERY>
{question}
</USER_QUERY>
"""
def create_conversation(sample):
  return {
    "messages": [
      {"role": "system", "content": system_message},
      {"role": "user", "content": user_prompt.format(question=sample["task"].replace('incomplit', 'incomplete'), context=sample["input"])},
      {"role": "assistant", "content": sample["expected_output"]}
    ]
  }

# Load dataset from the hub
training_dataset = load_dataset("Thytu/ChessInstruct", split="train")
eval_dataset = load_dataset("Thytu/ChessInstruct", split="test")

# Convert dataset to OAI messages
training_dataset = training_dataset.map(create_conversation, remove_columns=training_dataset.features, batched=False)
eval_dataset = eval_dataset.map(create_conversation, remove_columns=eval_dataset.features, batched=False)

# Print formatted user prompt
print(training_dataset[345]["messages"][1]["content"])

# %% [markdown]
# # Load LLM

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Hugging Face model id
model_id = "google/gemma-3-270m"
model_class = AutoModelForCausalLM

# Check if GPU benefits from bfloat16
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
    device_map="auto", # Let torch decide how to load the model
)

# BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
    bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id + "-it") # Load the Instruction Tokenizer to use the official Gemma template
model = model_class.from_pretrained(model_id, **model_kwargs)

# %% [markdown]
# # Define LoRA config

# %%
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"] # make sure to save the lm_head and embed_tokens as you train the special tokens
)

# %% [markdown]
# # Training Config

# %%
from trl import SFTConfig

args = SFTConfig(
    output_dir="gemma-3-270m-chess",        # directory to save and repository id
    max_length=512,                         # max sequence length for model and packing of the dataset
    packing=True,                           # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=4,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every x steps
    save_steps=50,                          # save checkpoint every x steps
    save_total_limit=2,                     # maximun local checkpoints saves
    eval_steps=500,                         # eval every x steps
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,   # use bfloat16 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    hub_strategy="checkpoint",              # save only the last checkpoint
    report_to="tensorboard",                # report metrics to tensorboard
    dataset_kwargs={
        "add_special_tokens": False, # We template with special tokens
        "append_concat_token": True, # Add EOS token as separator token between examples
    }
)

# %%
from trl import SFTTrainer

# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=training_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=tokenizer
)

# %% [markdown]
# # Train

# %%
if LOAD_CHECKPOINT:
    from huggingface_hub import snapshot_download

    repo_id = "jorvarea/gemma-3-270m-chess"
    local_dir = snapshot_download(repo_id, local_dir="gemma-3-270m-chess")

# %%
from contextlib import contextmanager

@contextmanager
def disable_weights_only():
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load

# %%
# Start training
if LOAD_CHECKPOINT:
    with disable_weights_only():
        trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

# %%
trainer.save_model()

# %%
# free the memory again
del model
del trainer
torch.cuda.empty_cache()

# %%
from peft import PeftModel

# Load Model base model
model = model_class.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

processor = AutoTokenizer.from_pretrained(args.output_dir)
processor.save_pretrained("merged_model")

# %%
from transformers import pipeline

# Load Model with PEFT adapter
model = model_class.from_pretrained(
  model_id,
  device_map="auto",
  torch_dtype=torch_dtype,
  attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%
from random import randint
import re

# Load the model and tokenizer into the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load a random sample from the test dataset
rand_idx = randint(0, len(eval_dataset["test"])-1)
test_sample = eval_dataset["test"][rand_idx]

# Convert as test example into a prompt with the Gemma template
stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
prompt = pipe.tokenizer.apply_chat_template(test_sample["messages"][:2], tokenize=False, add_generation_prompt=True)

# Generate our SQL query.
outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=stop_token_ids, disable_compile=True)

# Extract the user query and original answer
print(f"Context:\n", re.search(r'<CONTEXT>\n(.*?)\n</CONTEXT>', test_sample['messages'][0]['content'], re.DOTALL).group(1).strip())
print(f"Query:\n", re.search(r'<USER_QUERY>\n(.*?)\n</USER_QUERY>', test_sample['messages'][0]['content'], re.DOTALL).group(1).strip())
print(f"Original Answer:\n{test_sample['messages'][1]['content']}")
print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")


