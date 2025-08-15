# %% Imports
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

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
dataset = load_dataset("Thytu/ChessInstruct", split="train[:10%]")  # 10% para pruebas iniciales

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
    num_train_epochs=3,
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

