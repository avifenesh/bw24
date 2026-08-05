#!/usr/bin/env python3
"""Stage 1: Unsloth bf16 LoRA r=16 SFT on Qwen3.5-9B — 200 optimizer steps.
DESIGN.md §4 config: r=16 alpha=16, q/k/v/o/gate/up/down, unsloth grad ckpt,
adamw_8bit, seq 2048, bs 1 x grad-accum 4. Loss curve -> /root/pilot/loss.jsonl.
"""
import json
import time

from unsloth import FastLanguageModel  # must import before transformers/trl
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback

BASE = "/root/hf-models/qwen35-9b"
OUT = "/root/pilot/adapter"

t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    BASE, max_seq_length=2048, dtype=torch.bfloat16, load_in_4bit=False,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=3407,
)

ds = load_dataset("json", data_files="/root/pilot/dataset.jsonl", split="train")

def to_text(row):
    return {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)}

ds = ds.map(to_text, remove_columns=ds.column_names, num_proc=1)


class LossLog(TrainerCallback):
    def __init__(self, path):
        self.f = open(path, "w")

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            self.f.write(json.dumps({"step": state.global_step, **logs}) + "\n")
            self.f.flush()


trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds,
    callbacks=[LossLog("/root/pilot/loss.jsonl")],
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=200,
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=10,
        seed=3407,
        output_dir="/root/pilot/train-out",
        save_strategy="no",          # adapter-only save at the end (disk-burst cap)
        report_to="none",
        dataset_num_proc=1,          # host-CPU cap per co-location doctrine
        max_length=2048,
    ),
)
print(f"[train] setup {time.time()-t0:.1f}s; VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB")
r = trainer.train()
print(f"[train] train_runtime={r.metrics.get('train_runtime'):.1f}s "
      f"final_loss={r.metrics.get('train_loss'):.4f}")
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
print(f"[train] adapter saved -> {OUT}")
print(f"[train] peak VRAM {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
