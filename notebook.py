# %% Imports
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
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
tokenizer.save_pretrained("./gemma-3-270m-lora")

# %% Test model
def generate_move(moves: list[str], max_new_tokens=16, temperature=0.1, prompt: str = "Given some set of chess moves, write the best possible move"):
    # Tokenize input
    input_text = prompt + " " + " ".join(moves)
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate output tokens
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode generated tokens
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# %% Test model
moves = ["e4e5", "Nf3"]

generate_move(moves)

