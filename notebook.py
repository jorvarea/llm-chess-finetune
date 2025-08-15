# %% Imports
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch

# %% Load model and tokenizer
model_name = "unsloth/gemma-3-270m-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")

# %% Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)

# %% Load dataset and tokenize it
dataset = load_dataset("Thytu/ChessInstruct", split="train[:1%]")

def combine_input_output(example):
    # Format: <task/instruction> + <input> + <expected_output>
    example["text"] = f"{example['task']} {example['input']} {example['expected_output']}"
    return example

dataset = dataset.map(combine_input_output)

def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    # For causal LM, labels = input_ids
    tokens["labels"] = tokens["input_ids"]
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)


# %% Training Configuration
training_args = TrainingArguments(
    output_dir="./gemma-3-270m-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=False
)

# %% Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# %% Train
trainer.train()

# %% Save model and tokenizer
model.save_pretrained("./gemma-3-270m-lora")

# %% Test model
def generate_move(moves: list[str], max_new_tokens=8, temperature=0.1, prompt: str = "Given some set of chess moves, write the best possible move") -> str:
    # Tokenize input
    input_text = prompt + " " + " ".join(moves)
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate output tokens
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode generated tokens
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text[len(input_text):].strip()

# %% Load finetuned model
checkpoint_path = "./gemma-3-270m-lora/checkpoint-248"
model_base = AutoModelForCausalLM.from_pretrained("unsloth/gemma-3-270m-it")
model = PeftModel.from_pretrained(model_base, checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-270m-it")
model.to("cpu")

# %% Test model
#moves = ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "d7d6", "c1e3", "g7g6", "f2f3", "e7e5", "d4b3", "g8f6", "b1c3", "c8e6", "d1e2", "a7a5", "a2a4", "f8e7", "e1c1", "e8g8", "b3c5", "d6d5", "e4d5", "f6d5", "c3d5", "e6d5", "c5b7", "d8c8", "d1d5", "c8b7", "e2b5", "b7c8", "b5b3", "c6d4", "e3d4", "e5d4", "c1b1", "a8b8", "d5b5", "c8c7", "b3c4", "c7d6", "f1d3", "f8c8", "c4b3", "b8b5", "a4b5", "d6b4", "b3b4", "e7b4", "b1a2", "g8f8", "h1d1", "c8c7", "f3f4", "f8e7", "a2b3", "e7d6", "f4f5", "g6g5", "f5f6", "h7h6", "d1f1", "b4d2", "b3a4", "d2b4", "f1f2", "c7c5", "f2e2", "c5e5", "e2e5", "d6e5", "b5b6", "e5f6", "g2g4", "f6e6", "a4b5", "e6d7", "d3f5", "d7d6"]
moves = ["e4e5", "Nf3"]
print(generate_move(moves, max_new_tokens=4))

# %%
