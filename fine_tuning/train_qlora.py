"""QLoRA fine-tuning (4-bit NF4 + double quantisation, r=16, alpha=32) with TRL's SFTTrainer.

Loss is computed on the assistant JSON only (prompt tokens are masked), using pre-tokenised
examples so the script works across TRL versions (argument names are matched by introspection).

Usage:
    python -m fine_tuning.train_qlora --model meta-llama/Meta-Llama-3-8B-Instruct --epochs 2
"""
from __future__ import annotations

import argparse
import inspect
import json
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import config


def encode(messages: list[dict[str, str]], tok: Any, max_len: int) -> dict[str, list[int]]:
    prompt = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
    completion = messages[-1]["content"] + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion, add_special_tokens=False)["input_ids"]
    ids, labels = p_ids + c_ids, [-100] * len(p_ids) + c_ids
    if len(ids) > max_len:                       # keep the completion, trim the oldest prompt tokens
        ids, labels = ids[-max_len:], labels[-max_len:]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def load_dataset(path: Any, tok: Any, max_len: int, limit: int | None) -> Dataset:
    rows = config.read_jsonl(path)[: limit or None]
    return Dataset.from_list([encode(r["messages"], tok, max_len) for r in rows])


class PadCollator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, value: seq + [value] * (width - len(seq))  # noqa: E731
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        }


def accepted(cls: Any, **kwargs: Any) -> dict[str, Any]:
    """Keep only keyword arguments accepted by ``cls.__init__`` (TRL/transformers renames across versions)."""
    params = set(inspect.signature(cls.__init__).parameters)
    return {k: v for k, v in kwargs.items() if k in params}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=config.BASE_MODEL, help=f"e.g. {config.BASE_MODEL} or {config.ALT_BASE_MODEL}")
    parser.add_argument("--epochs", type=float, default=config.SFT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.SFT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=config.SFT_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=config.SFT_LR)
    parser.add_argument("--max-seq-len", type=int, default=config.SFT_MAX_SEQ_LEN)
    parser.add_argument("--max-samples", type=int, default=None, help="debug: limit training rows")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA (bitsandbytes 4-bit) needs a CUDA GPU; use Colab/Kaggle/RunPod.")
    config.ensure_dirs()
    bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=config.QLORA_QUANT_TYPE,
        bnb_4bit_use_double_quant=config.QLORA_DOUBLE_QUANT, bnb_4bit_compute_dtype=compute_dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto",
                                                 torch_dtype=compute_dtype)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
                      target_modules=config.LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM")

    train_ds = load_dataset(config.SFT_TRAIN_PATH, tok, args.max_seq_len, args.max_samples)
    eval_ds = load_dataset(config.SFT_VAL_PATH, tok, args.max_seq_len, 256)
    print(f"train={len(train_ds)} eval={len(eval_ds)} model={args.model}")

    sft_args = SFTConfig(**accepted(
        SFTConfig, output_dir=str(config.ADAPTER_DIR / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_ratio=0.03, weight_decay=0.0, optim="paged_adamw_8bit", bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True, logging_steps=10, save_strategy="epoch", eval_strategy="epoch",
        evaluation_strategy="epoch", report_to="none", seed=args.seed, remove_unused_columns=False,
        max_seq_length=args.max_seq_len, max_length=args.max_seq_len, packing=False,
        dataset_kwargs={"skip_prepare_dataset": True}))

    trainer_kwargs = dict(model=model, args=sft_args, train_dataset=train_ds, eval_dataset=eval_ds,
                          data_collator=PadCollator(tok.pad_token_id), peft_config=lora,
                          processing_class=tok, tokenizer=tok)
    trainer = SFTTrainer(**accepted(SFTTrainer, **trainer_kwargs))
    result = trainer.train()

    trainer.save_model(str(config.ADAPTER_DIR))
    tok.save_pretrained(str(config.ADAPTER_DIR))
    metrics = {"base_model": args.model, "train": result.metrics, "eval": trainer.evaluate()}
    (config.REPORT_DIR / "qlora_train_metrics.json").write_text(json.dumps(metrics, indent=2, default=str),
                                                                encoding="utf-8")
    print(f"adapter saved to {config.ADAPTER_DIR}")


if __name__ == "__main__":
    main()
