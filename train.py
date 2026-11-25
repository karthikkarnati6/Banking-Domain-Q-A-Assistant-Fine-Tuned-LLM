"""Fine-tune a pretrained chat model with supervised Q&A examples."""

from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name}:{line_no}: {exc}") from exc
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path.name}:{line_no} needs a messages list with user and assistant turns")
            roles = [m.get("role") for m in messages]
            if "user" not in roles or "assistant" not in roles:
                raise ValueError(f"{path.name}:{line_no} needs user and assistant messages")
            if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
                raise ValueError(f"{path.name}:{line_no} has an empty or invalid message")
            rows.append({"messages": messages})
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def main() -> None:
    import torch

    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    train_path = ROOT / os.getenv("TRAIN_FILE", "data/train.jsonl")
    eval_path = ROOT / os.getenv("EVAL_FILE", "data/eval.jsonl")
    output_dir = ROOT / os.getenv("OUTPUT_DIR", "outputs/domain-qa-adapter")
    use_qlora = os.getenv("USE_QLORA", "false").lower() in {"1", "true", "yes"}
    if use_qlora and not torch.cuda.is_available():
        raise RuntimeError("USE_QLORA=true requires a supported CUDA GPU and bitsandbytes installation.")

    train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    model_kwargs: dict = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32}
    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        model_kwargs["device_map"] = "auto"

    config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(os.getenv("NUM_EPOCHS", "3")),
        learning_rate=float(os.getenv("LEARNING_RATE", "0.0002")),
        per_device_train_batch_size=int(os.getenv("BATCH_SIZE", "1")),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "8")),
        max_length=int(os.getenv("MAX_LENGTH", "1024")),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        model_init_kwargs=model_kwargs,
        eos_token=tokenizer.eos_token,
    )
    trainer = SFTTrainer(
        model=model_name,
        args=config,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows),
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
        quantization_config=quantization_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved adapter and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
