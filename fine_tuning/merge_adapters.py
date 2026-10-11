"""Merge the LoRA adapter into the base weights and export an FP16 model for vLLM / HF serving.

Usage:
    python -m fine_tuning.merge_adapters --smoke-test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


def merge(base_model: str, adapter_dir: Path, out_dir: Path) -> None:
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto",
                                                low_cpu_mem_usage=True)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(out_dir)
    (out_dir / "merge_meta.json").write_text(json.dumps(
        {"base_model": base_model, "adapter": str(adapter_dir), "dtype": "float16",
         "lora_r": config.LORA_R, "lora_alpha": config.LORA_ALPHA}, indent=2), encoding="utf-8")


def smoke_test(model_dir: Path) -> None:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
    example = config.read_jsonl(config.SFT_VAL_PATH)[0]["messages"]
    ids = tok.apply_chat_template(example[:-1], add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(model.device)
    out = model.generate(**ids, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    print("expected :", example[-1]["content"])
    print("generated:", tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", type=Path, default=config.ADAPTER_DIR)
    parser.add_argument("--out", type=Path, default=config.MERGED_MODEL_DIR)
    parser.add_argument("--base-model", default=None, help="defaults to the base recorded in the adapter config")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    base = args.base_model or cfg["base_model_name_or_path"]
    merge(base, args.adapter, args.out)
    print(f"merged FP16 model written to {args.out}")
    if args.smoke_test:
        smoke_test(args.out)


if __name__ == "__main__":
    main()
