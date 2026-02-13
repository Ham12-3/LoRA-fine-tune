"""
Train google/flan-t5-base with LoRA adapters on the knkarthick/dialogsum dataset.
Uses 4-bit quantization via BitsAndBytesConfig to reduce VRAM when a GPU is available.
Falls back to CPU training automatically if no GPU is detected.
"""

import os
import inspect
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model
import evaluate
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "google/flan-t5-base"
DATASET_NAME = "knkarthick/dialogsum"
NUM_TRAIN_SAMPLES = 500
OUTPUT_DIR = "./lora-flan-t5-summarization"
ADAPTER_DIR = "./lora-adapters"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 3
LOGGING_STEPS = 25

# ---------------------------------------------------------------------------
# Detect hardware: GPU, precision, and 4-bit support
# ---------------------------------------------------------------------------
use_gpu = torch.cuda.is_available()
use_bf16 = False
use_fp16 = False
use_4bit = False

if use_gpu:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
        use_4bit = True
        print("bitsandbytes available - 4-bit quantization enabled")
    except ImportError:
        print("bitsandbytes not available - loading model without quantization")
else:
    print("No GPU detected - will train on CPU (slower, no quantization)")

compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

print(f"bf16: {use_bf16} | fp16: {use_fp16} | 4-bit: {use_4bit}")

# ---------------------------------------------------------------------------
# Load tokenizer and model
# ---------------------------------------------------------------------------
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ---------------------------------------------------------------------------
# LoRA configuration -- r=16, alpha=32, targeting q and v projection modules
# ---------------------------------------------------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Load and preprocess dataset (first 500 rows)
# ---------------------------------------------------------------------------
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

train_dataset = dataset["train"].select(range(NUM_TRAIN_SAMPLES))
val_dataset = dataset["validation"].select(range(min(100, len(dataset["validation"]))))

PREFIX = "Summarize the following dialogue:\n\n"


def preprocess(examples):
    inputs = [PREFIX + dialogue for dialogue in examples["dialogue"]]
    targets = examples["summary"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        padding="max_length",
        truncation=True,
    )

    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        padding="max_length",
        truncation=True,
    )

    # Replace pad token ids in labels with -100 so they are ignored by the loss
    labels["input_ids"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("Tokenizing dataset...")
tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names)

# ---------------------------------------------------------------------------
# Evaluation metric - ROUGE
# ---------------------------------------------------------------------------
rouge = evaluate.load("rouge")


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Decode predictions
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Decode labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v, 4) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Training arguments - robust to different transformers versions
# ---------------------------------------------------------------------------
_trainer_params = inspect.signature(Seq2SeqTrainingArguments).parameters
eval_key = "eval_strategy" if "eval_strategy" in _trainer_params else "evaluation_strategy"

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=LOGGING_STEPS,
    **{eval_key: "epoch"},
    save_strategy="epoch",
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    bf16=use_bf16,
    fp16=use_fp16,
    use_cpu=not use_gpu,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
)

print(f"Precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32 (CPU)'}")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# ---------------------------------------------------------------------------
# Save only the LoRA adapters (not the full model)
# ---------------------------------------------------------------------------
print(f"Saving LoRA adapters to {ADAPTER_DIR}...")
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print("Done! LoRA adapters saved successfully.")

# ---------------------------------------------------------------------------
# Quick inference demo using the trained adapters
# ---------------------------------------------------------------------------
print("\n--- Inference Demo ---")
test_dialogue = (
    "Alice: Hey Bob, did you finish the report?\n"
    "Bob: Almost done. I need another hour or so.\n"
    "Alice: Okay, just make sure to send it before the meeting at 3 PM.\n"
    "Bob: Will do. I will email it to you and the team by 2:30.\n"
    "Alice: Perfect, thanks!"
)

input_text = PREFIX + test_dialogue
input_ids = tokenizer(input_text, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).input_ids
input_ids = input_ids.to(model.device)

with torch.no_grad():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=MAX_TARGET_LENGTH)

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Dialogue:\n{test_dialogue}\n")
print(f"Generated Summary:\n{summary}")
