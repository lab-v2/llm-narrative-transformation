#!/usr/bin/env python3
"""
Local fine-tuning script for open-weights models (GPU).

Example:
  python fine_tune.py \
    --data-csv data/fine_tuning_data/llm3_training_dataset_bedrock-us.meta.llama4-maverick-17b-instruct-v1-0.csv \
    --output-dir output/local_llama4_maverick_lora
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any
import torch
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed

logger = logging.getLogger(__name__)
import sys, accelerate, transformers, torch
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import torch, gc
gc.collect()
torch.cuda.empty_cache()

print("PYTHON:", sys.executable)
print("ACCELERATE:", accelerate.__version__, accelerate.__file__)
print("TRANSFORMERS:", transformers.__version__)
print("TORCH:", torch.__version__)
print("CUDA:", torch.cuda.is_available())


def _build_text_formatter(tokenizer, system_prompt: str, use_chat_template: bool):
    def _format_row(example: Dict[str, Any]) -> Dict[str, str]:
        prompt = example.get("user_prompt", "")
        completion = example.get("assistant_output", "")

        if use_chat_template and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": completion})
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            parts = []
            if system_prompt:
                parts.append(f"System: {system_prompt}")
            parts.append(f"User: {prompt}")
            parts.append(f"Assistant: {completion}")
            text = "\n\n".join(parts)

        return {"text": text}

    return _format_row


def main() -> int:
    parser = argparse.ArgumentParser(description="Local fine-tuning for open-weights models.")
    parser.add_argument(
        "--base-model",
        default="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        #"/home/dbavikad/leibniz/llm-narrative-transformation/my_models/Llama-4-Maverick-17B-128E-Instruct",
        help="Hugging Face model id or local path",
    )
    parser.add_argument(
        "--data-csv",
        required=True,
        help="CSV or JSONL with user_prompt/assistant_output columns (or JSONL with messages)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for fine-tuned model")
    parser.add_argument("--prompt-col", default="user_prompt", help="Prompt column name")
    parser.add_argument("--completion-col", default="assistant_output", help="Completion column name")
    parser.add_argument("--system-prompt", default="", help="Optional system prompt")
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bfloat16 if supported",
    )
    parser.add_argument("--fp16", action="store_true", help="Use float16")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--logging-steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--no-chat-template", action="store_true", help="Disable tokenizer chat template")
    parser.add_argument(
        "--loss-log-file",
        default="training_loss.jsonl",
        help="Filename (within output-dir) to record training loss logs as JSONL",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    

    set_seed(args.seed)

    if not torch.cuda.is_available():
        logger.error("CUDA GPU is required for fine-tuning but was not detected.")
        return 2

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        logger.error("No CUDA GPUs detected.")
        return 2
    visible_gpus = min(gpu_count, 2)
    if gpu_count < 2:
        logger.warning("Only %d GPU detected; running on a single GPU.", gpu_count)
    for idx in range(visible_gpus):
        device_name = torch.cuda.get_device_name(idx)
        capability = torch.cuda.get_device_capability(idx)
        total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        logger.info(
            f"Using GPU {idx}: {device_name} | CC {capability[0]}.{capability[1]} | VRAM {total_mem_gb:.1f} GB"
        )

    data_csv = Path(args.data_csv)
    if not data_csv.exists():
        logger.error(f"Data file not found: {data_csv}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer and model...")
    # Llama 4 tokenizer config advertises a base class; prefer fast tokenizer to avoid get_vocab() NotImplementedError.
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    except Exception as exc:
        logger.warning("Fast tokenizer load failed (%s). Falling back to slow tokenizer.", exc)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    # try:
    max_memory = {}
    for idx in range(visible_gpus):
        total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        max_memory[idx] = f"{int(total_mem_gb * 0.9)}GiB"
    load_kwargs = {"torch_dtype": torch_dtype}
    if visible_gpus > 1:
        load_kwargs.update(
            device_map="auto",
            max_memory=max_memory
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **load_kwargs
    )
    if visible_gpus == 1 and torch.cuda.is_available():
        model.to("cuda")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    logger.info("Loading dataset...")
    data_suffix = data_csv.suffix.lower()
    dataset_format = "json" if data_suffix in {".json", ".jsonl"} else "csv"
    dataset = load_dataset(dataset_format, data_files=str(data_csv))

    # Normalize input to user_prompt/assistant_output columns.
    train_columns = set(dataset["train"].column_names)
    if "messages" in train_columns and ("user_prompt" not in train_columns or "assistant_output" not in train_columns):
        def _messages_to_columns(example: Dict[str, Any]) -> Dict[str, Any]:
            prompt = ""
            completion = ""
            for msg in example.get("messages", []):
                role = msg.get("role")
                if role == "user" and not prompt:
                    prompt = msg.get("content", "")
                elif role == "assistant" and not completion:
                    completion = msg.get("content", "")
            return {"user_prompt": prompt, "assistant_output": completion}

        dataset = dataset.map(_messages_to_columns, remove_columns=dataset["train"].column_names)
    elif args.prompt_col != "user_prompt" or args.completion_col != "assistant_output":
        def _rename_columns(example: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "user_prompt": example.get(args.prompt_col, ""),
                "assistant_output": example.get(args.completion_col, ""),
            }
        dataset = dataset.map(_rename_columns, remove_columns=dataset["train"].column_names)

    formatter = _build_text_formatter(tokenizer, args.system_prompt, not args.no_chat_template)
    dataset = dataset.map(formatter, remove_columns=dataset["train"].column_names)

    def _tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    loss_log_path = output_dir / args.loss_log_file
    loss_entries = [entry for entry in trainer.state.log_history if "loss" in entry]
    if loss_entries:
        with loss_log_path.open("w", encoding="utf-8") as handle:
            for entry in loss_entries:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        logger.info("Wrote training loss log to %s (%d entries).", loss_log_path, len(loss_entries))
    else:
        logger.warning("No loss entries found in trainer log history; loss log not written.")

    logger.info("Saving model...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
