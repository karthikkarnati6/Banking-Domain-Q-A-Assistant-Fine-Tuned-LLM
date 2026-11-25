# Domain Q&A with a Fine-Tuned LLM

A small supervised fine-tuning (SFT) project for a general banking customer-support assistant. It uses Hugging Face TRL and PEFT LoRA, which saves trained adapter weights instead of a full model copy. Set `USE_QLORA=true` to load the base model in 4-bit (best suited to supported Linux/NVIDIA GPU environments).

Fine-tuning teaches answer style, terminology, and common response patterns. It does not reliably store a changing knowledge base or provide citations. For answers that must quote current source documents, pair the adapter with retrieval (RAG).

## Setup

Python 3.10 or newer is recommended. Create an environment and install the packages:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For QLoRA, install a PyTorch build compatible with your CUDA setup, then install `bitsandbytes` and set `USE_QLORA=true` in `.env`. QLoRA support depends on the operating system, GPU, and installed PyTorch/bitsandbytes builds. LoRA without quantization is the default.

## Prepare examples

Edit `data/train.jsonl` and `data/eval.jsonl`. Each line is one JSON object with a `messages` list containing a user question and assistant answer. The included banking Q&A are generic illustrative examples, not bank policy or a substitute for compliance review. Replace or extend them with approved, accurate examples for the institution, market, and products this assistant will support. Keep evaluation examples separate from training examples.

```json
{"messages":[{"role":"system","content":"You are a helpful banking support assistant. Do not invent institution-specific policies or request passwords, PINs, one-time codes, or full account details."},{"role":"user","content":"Example question?"},{"role":"assistant","content":"Example answer."}]}
```

Replace the illustrative examples before training. Use examples that are accurate, representative, and cleared for model training; exclude private or regulated data unless your organization has approved that use.

## Configure and train

Copy `.env.example` to `.env`, then set `MODEL_NAME` to an instruction-tuned causal language model that fits your hardware. A small starting point is `Qwen/Qwen2.5-0.5B-Instruct`; check the model card and license before use. Update `DOMAIN` and `SYSTEM_PROMPT` for your bank, market, and approved support scope. The sample prompt tells the model not to request credentials and to route account-specific and suspected-fraud cases to official bank channels; these controls should also be enforced in the application and reviewed before customer use.

```powershell
python train.py
```

The adapter and tokenizer are saved under `outputs/domain-qa-adapter/`. Training logs and checkpoints are saved in that directory as well.

## Ask questions

```powershell
python chat.py
```

The chat script loads the base model and the saved adapter. Use Ctrl+C to exit.

## Useful settings

See `.env.example` for model, dataset, training, and memory settings. If training runs out of memory, lower `MAX_LENGTH` or `BATCH_SIZE`, increase `GRADIENT_ACCUMULATION_STEPS`, or use a smaller base model.

## Project files

- `train.py`: reads JSONL conversations, validates them, and trains a LoRA adapter with TRL `SFTTrainer`.
- `chat.py`: loads the adapter and runs a terminal Q&A loop.
- `data/`: illustrative train/evaluation JSONL files.
- `.env.example`: project configuration defaults.

The implementation follows Hugging Face's [TRL SFTTrainer guide](https://huggingface.co/docs/trl/sft_trainer) and [PEFT/QLoRA guide](https://huggingface.co/docs/transformers/quantization/finetune).
