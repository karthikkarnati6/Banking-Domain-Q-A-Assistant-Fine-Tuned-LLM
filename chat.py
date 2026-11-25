"""Interactive terminal Q&A using the fine-tuned LoRA adapter."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")


def main() -> None:
    adapter_dir = ROOT / os.getenv("OUTPUT_DIR", "outputs/domain-qa-adapter")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found at {adapter_dir}. Run train.py first.")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_dir),
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    system_prompt = os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful domain specialist. Answer accurately and say when you are unsure.",
    )
    print("Domain Q&A ready. Type /quit to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"/quit", "/exit"}:
            break
        if not question:
            continue
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=384,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
