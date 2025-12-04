import argparse
import math
import os
import json
from pathlib import Path

import torch
from datasets import load_dataset, Audio
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import matplotlib.pyplot as plt

from utils_relative import MODEL_ID, build_conversation, make_collate_fn, print_trainable_parameters


OUTPUT_DIR = "/fs/gamma-projects/audio/kajayaku/temporal_understanding/models/qwen2audio/finetuned_relative_model"
NUM_EPOCHS = 2
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 32
LR = 2e-5
WEIGHT_DECAY = 0.01
LORA_R = 16
LORA_DROPOUT = 0.05
LORA_ALPHA = 32


class LossLoggingCallback(TrainerCallback):
    def __init__(self, output_dir: str, plot_every: int = 100):
        self.output_dir = Path(output_dir)
        self.plot_every = plot_every
        self.train_losses = []
        self.eval_losses = []
        self.global_steps = []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.output_dir / "loss_history.json"
        self.plot_path = self.output_dir / "loss_curve.png"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control

        step = state.global_step
        self.global_steps.append(step)

        if "loss" in logs:
            self.train_losses.append(float(logs["loss"]))
        else:
            self.train_losses.append(None)

        if "eval_loss" in logs:
            self.eval_losses.append(float(logs["eval_loss"]))
        else:
            self.eval_losses.append(None)

        # Save JSON each log
        history = {
            "global_steps": self.global_steps,
            "train_loss": self.train_losses,
            "eval_loss": self.eval_losses,
        }
        with open(self.json_path, "w") as f:
            json.dump(history, f, indent=2)

        # Plot intermittently
        if step % self.plot_every == 0 and step > 0:
            self._plot_losses()

        return control

    def _plot_losses(self):
        plt.figure(figsize=(6, 4))

        if any(x is not None for x in self.train_losses):
            plt.plot(
                self.global_steps,
                [x if x is not None else float("nan") for x in self.train_losses],
                label="train loss",
            )

        if any(x is not None for x in self.eval_losses):
            plt.plot(
                self.global_steps,
                [x if x is not None else float("nan") for x in self.eval_losses],
                label="eval loss",
            )

        plt.xlabel("global step")
        plt.ylabel("loss")
        plt.title("Training / Eval Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()


def preprocess_dataset(csv_path: str, system_prompt: str | None, audio_base_path: str):
    ds = load_dataset("csv", data_files=csv_path, split="train")
    ds = ds.map(lambda x: {"audio": f"{audio_base_path}/{x['filename']}"})
    ds = ds.cast_column("audio", Audio(decode=False))

    def preprocess(example, idx):
        conversation_data = build_conversation(
            example,
            system_prompt=system_prompt,
        )
        return {
            "text": conversation_data["text"],
            "audio_path": conversation_data["audio_path"],
            "answer": conversation_data["answer"],
            "idx": idx,
            "file_name": example["filename"],
        }

    processed = ds.map(
        preprocess,
        with_indices=True,
        remove_columns=ds.column_names,
        desc="Building conversations",
    )
    return processed


def compute_save_steps(num_examples: int, per_device_batch_size: int, grad_accum_steps: int) -> int:
    num_batches = math.ceil(num_examples / per_device_batch_size)
    steps_per_epoch = max(1, math.ceil(num_batches / grad_accum_steps))
    return max(1, math.ceil(steps_per_epoch * 0.1))


def create_trainer(train_dataset, eval_dataset, args):
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        #device_map="auto"
    )
    model.config.use_cache = False
    
    # Enable gradient checkpointing for memory efficiency
    # if hasattr(model, "gradient_checkpointing_enable"):
    #     model.gradient_checkpointing_enable()

    # LoRA configuration targeting Qwen2-Audio's attention modules
    peft_config = LoraConfig(
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
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    save_steps = compute_save_steps(
        len(train_dataset),
        args.per_device_batch_size,
        args.gradient_accumulation_steps,
    )
    print(f"Saving checkpoints every {save_steps} training steps (~0.1 epochs)")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        gradient_checkpointing=False,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        do_eval=True,
        eval_strategy="steps",
        eval_steps=save_steps,
        per_device_eval_batch_size=args.per_device_batch_size,
        dataset_kwargs={"skip_prepare_dataset": True},
        ddp_find_unused_parameters=True,
    )

    loss_cb = LossLoggingCallback(output_dir=args.output_dir, plot_every=save_steps)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=make_collate_fn(processor, debug_first_batch=True),
        processing_class=processor,
        callbacks=[loss_cb],
    )

    return trainer, processor


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-Audio 7B Instruct.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--per_device_batch_size", type=int, default=PER_DEVICE_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--learning_rate", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"],
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--audio_base_path", type=str, default="/fs/gamma-projects/audio/apoorvak/tacos_dataset/audios_folder")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = preprocess_dataset(
        csv_path="/fs/gamma-projects/audio/kajayaku/temporal_understanding/curated_data/qwen2_finetune_final/train_set.csv",
        system_prompt=args.system_prompt,
        audio_base_path=args.audio_base_path,
    )
    val_dataset = preprocess_dataset(
        csv_path="/fs/gamma-projects/audio/kajayaku/temporal_understanding/curated_data/qwen2_finetune_final/val_set.csv",
        system_prompt=args.system_prompt,
        audio_base_path=args.audio_base_path,
    )
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    trainer, processor = create_trainer(train_dataset, val_dataset, args)
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()