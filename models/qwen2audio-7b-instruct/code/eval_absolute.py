import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset, Audio
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# from utils import MODEL_ID
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on Qwen2-Audio for multiple choice task.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the fine-tuned LoRA checkpoint directory.",
    )
    parser.add_argument(
        "--audio_csv",
        type=str,
        required=True,
        help="Path to CSV file with audio annotations (test or val set).",
    )
    parser.add_argument(
        "--audio_base_path",
        type=str,
        required=True,
        help="Base path to audio files directory.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.json",
        help="Path to save predictions JSON.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (recommend keeping at 1 to avoid OOM).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Maximum tokens to generate per example (short for A/B/C/D).",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional system prompt.",
    )
    return parser.parse_args()


def build_prompt_text(example, system_prompt: str | None = None):
    """Build prompt text for multiple choice question with timestamps."""
    question = example["question"]
    option_a = example["A"]
    option_b = example["B"]
    option_c = example["C"]
    option_d = example["D"]
    
    text_parts = []
    if system_prompt:
        text_parts.append(f"System: {system_prompt}\n\n")
    
    # Format the question with timestamp options
    question_text = f"{question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}"
    text_parts.append(f"User: {question_text}\nAssistant:")
    
    return "".join(text_parts)


def preprocess_dataset(csv_path: str, system_prompt: str | None, audio_base_path: str):
    ds = load_dataset("csv", data_files=csv_path, split="train")
    ds = ds.map(lambda x: {"audio": f"{audio_base_path}/{x['filename']}"})
    ds = ds.cast_column("audio", Audio(decode=False))
    
    def preprocess(example, idx):
        prompt_text = build_prompt_text(example, system_prompt=system_prompt)
        return {
            "text": prompt_text,
            "audio_path": example["audio"]["path"],
            "idx": idx,
            "file_name": example["filename"],
            "sound_event": example["sound_event"],
            "question": example["question"],
            "option_a": example["A"],
            "option_b": example["B"],
            "option_c": example["C"],
            "option_d": example["D"],
            "ground_truth": example["correct_label"],
            "correct_time": example["correct_time"],
        }
    
    processed = ds.map(
        preprocess,
        with_indices=True,
        remove_columns=ds.column_names,
        desc="Building prompts",
    )
    
    return processed


def load_model(checkpoint_dir: str, device: torch.device):
    """Load fine-tuned LoRA model and processor."""
    print(f"Loading model from {checkpoint_dir}")
    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
    
    # Load base model
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model, processor


def run_inference(
    model,
    processor,
    dataset,
    device: torch.device,
    batch_size: int = 1,
    max_new_tokens: int = 10,
) -> List[Dict]:
    """Run inference on the dataset using DataLoader."""
    model.eval()
    predictions = []
    
    def inference_collate_fn(examples: List[Dict]):
        # Extract metadata
        metadata = [
            {
                "idx": ex["idx"],
                "file_name": ex["file_name"],
                "sound_event": ex["sound_event"],
                "question": ex["question"],
                "option_a": ex["option_a"],
                "option_b": ex["option_b"],
                "option_c": ex["option_c"],
                "option_d": ex["option_d"],
                "ground_truth": ex["ground_truth"],
                "correct_time": ex["correct_time"],
            }
            for ex in examples
        ]
        
        # Get texts and audio paths
        texts = [ex["text"] for ex in examples]
        audio_paths = [ex["audio_path"] for ex in examples]
        
        # Process with Qwen2-Audio processor
        batch_inputs = processor(
            text=texts,
            audios=audio_paths,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000,
        )
        
        return {
            "input_ids": batch_inputs["input_ids"],
            "attention_mask": batch_inputs["attention_mask"],
            "audio_features": batch_inputs.get("audio_features"),
            "audio_feature_lens": batch_inputs.get("audio_feature_lens"),
            "metadata": metadata,
        }
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=inference_collate_fn,
        shuffle=False,
    )
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            # Extract metadata
            metadata = batch.pop("metadata")
            
            # Move inputs to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Prepare generation kwargs
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": processor.tokenizer.pad_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            }
            
            # Add audio features if present
            if batch["audio_features"] is not None:
                gen_kwargs["audio_features"] = batch["audio_features"].to(device)
            if batch["audio_feature_lens"] is not None:
                gen_kwargs["audio_feature_lens"] = batch["audio_feature_lens"].to(device)
            
            try:
                if device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        generated = model.generate(**gen_kwargs)
                else:
                    generated = model.generate(**gen_kwargs)
            except Exception as e:
                print(f"Error during generation: {e}")
                print(f"Input shapes - ids: {input_ids.shape}")
                if batch["audio_features"] is not None:
                    print(f"audio_features: {batch['audio_features'].shape}")
                raise
            
            # Decode predictions
            gen_tokens = generated[:, input_ids.shape[1]:]
            texts = processor.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            
            # Store predictions
            for meta, pred_text in zip(metadata, texts):
                # Extract just the letter (A, B, C, or D) from prediction
                pred_clean = pred_text.strip()
                if len(pred_clean) > 0:
                    pred_letter = pred_clean[0].upper()
                else:
                    pred_letter = ""
                
                # Map prediction letter to timestamp
                option_map = {
                    "A": meta["option_a"],
                    "B": meta["option_b"],
                    "C": meta["option_c"],
                    "D": meta["option_d"],
                }
                pred_time = option_map.get(pred_letter, "")
                
                predictions.append({
                    "idx": meta["idx"],
                    "filename": meta["file_name"],
                    "sound_event": meta["sound_event"],
                    "question": meta["question"],
                    "options": {
                        "A": meta["option_a"],
                        "B": meta["option_b"],
                        "C": meta["option_c"],
                        "D": meta["option_d"],
                    },
                    "ground_truth_label": meta["ground_truth"],
                    "ground_truth_time": meta["correct_time"],
                    "prediction_label": pred_letter,
                    "prediction_time": pred_time,
                    "raw_prediction": pred_text.strip(),
                    "correct": pred_letter == meta["ground_truth"],
                })
            
            # Clear cache after each batch to prevent OOM
            if device.type == "cuda":
                torch.cuda.empty_cache()
    
    return predictions


def save_predictions(predictions: List[Dict], output_file: str):
    """Save predictions to JSON file and print accuracy metrics."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    # Calculate accuracy
    total = len(predictions)
    correct = sum(1 for p in predictions if p["correct"])
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    print(f"\nSaved {total} predictions to {output_path}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.2f}%")
    
    # Per-option accuracy
    option_stats = {"A": {"total": 0, "correct": 0}, 
                    "B": {"total": 0, "correct": 0},
                    "C": {"total": 0, "correct": 0},
                    "D": {"total": 0, "correct": 0}}
    
    for p in predictions:
        gt = p["ground_truth_label"]
        if gt in option_stats:
            option_stats[gt]["total"] += 1
            if p["correct"]:
                option_stats[gt]["correct"] += 1
    
    print("\nPer-option accuracy:")
    for opt in ["A", "B", "C", "D"]:
        stats = option_stats[opt]
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"  {opt}: {stats['correct']}/{stats['total']} = {acc:.2f}%")
    
    # Calculate MAE for timestamps (only for correct predictions)
    correct_preds = [p for p in predictions if p["correct"]]
    if correct_preds:
        mae_values = []
        for p in correct_preds:
            try:
                gt_time = float(p["ground_truth_time"])
                pred_time = float(p["prediction_time"])
                mae_values.append(abs(gt_time - pred_time))
            except (ValueError, TypeError):
                pass
        
        if mae_values:
            mae = sum(mae_values) / len(mae_values)
            print(f"\nMean Absolute Error (MAE) for timestamps: {mae:.4f} seconds")
            print(f"(Calculated on {len(mae_values)} correct predictions)")
    
    # Per-sound-event stats
    event_stats = {}
    for p in predictions:
        event = p["sound_event"]
        if event not in event_stats:
            event_stats[event] = {"total": 0, "correct": 0}
        event_stats[event]["total"] += 1
        if p["correct"]:
            event_stats[event]["correct"] += 1
    
    if len(event_stats) <= 20:  # Only print if reasonable number of events
        print("\nPer-sound-event accuracy:")
        for event in sorted(event_stats.keys()):
            stats = event_stats[event]
            acc = stats["correct"] / stats["total"] * 100
            print(f"  {event}: {stats['correct']}/{stats['total']} = {acc:.2f}%")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")
    
    # Warn if batch size > 1
    if args.batch_size > 1:
        print(f"WARNING: Using batch_size={args.batch_size}. If you encounter OOM errors, use --batch_size 1")
    
    # Load and preprocess dataset
    dataset = preprocess_dataset(
        args.audio_csv,
        args.system_prompt,
        args.audio_base_path
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Load model
    model, processor = load_model(args.checkpoint_dir, device)
    
    # Run inference
    predictions = run_inference(
        model,
        processor,
        dataset,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    
    # Save results
    save_predictions(predictions, args.output_file)
    
    # Print sample predictions
    print("\n" + "="*80)
    print("Sample predictions:")
    for pred in predictions[:5]:
        print(f"\nIdx: {pred['idx']} | File: {pred['filename']}")
        print(f"Sound Event: {pred['sound_event']}")
        print(f"Question: {pred['question']}")
        print(f"Ground truth: {pred['ground_truth_label']} ({pred['ground_truth_time']}s)")
        print(f"Prediction: {pred['prediction_label']} ({pred['prediction_time']}s) | {'✓' if pred['correct'] else '✗'}")
        print(f"Raw output: {pred['raw_prediction']}")


if __name__ == "__main__":
    main()