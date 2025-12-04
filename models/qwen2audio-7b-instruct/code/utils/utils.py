import json
import re
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

__all__ = [
    "MODEL_ID",
    "build_conversation",
    "make_collate_fn",
    "print_trainable_parameters",
]


def build_conversation(
    row: Dict[str, str],
    system_prompt: Optional[str] = None,
) -> Dict[str, any]:
    """
    Build a single-turn conversation from a CSV row for Qwen2-Audio.
    Returns a dict with text and audio_path that will be processed in collate_fn.
    
    row: dict containing 'filename', 'question', 'intervals', 'subclass'
    system_prompt: optional system message
    """
    audio_path = row["audio"]["path"]
    answer_text = row["intervals"]
    question = row["question"]
    
    # Build the text prompt
    text_parts = []
    if system_prompt:
        text_parts.append(f"System: {system_prompt}\n")
    
    text_parts.append(f"User: {question}\nAssistant: {answer_text}")
    
    return {
        "text": "".join(text_parts),
        "audio_path": audio_path,
        "answer": answer_text,
    }


def make_collate_fn(processor: AutoProcessor, debug_first_batch: bool = True):
    """
    Collate function for Qwen2-Audio that:
      * Takes examples with text and audio_path
      * Processes them using the Qwen2-Audio processor
      * Returns everything the model needs for training
    """
    first = {"done": False}

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex["text"] for ex in examples]
        audio_paths = [ex["audio_path"] for ex in examples]
        
        # Process with Qwen2-Audio processor
        # Note: Check Qwen2-Audio docs for exact processor API
        batch_inputs = processor(
            text=texts,
            audios=audio_paths,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000,
        )
        
        input_ids = batch_inputs["input_ids"]
        attention_mask = batch_inputs["attention_mask"]
        
        # Create labels (shift input_ids for causal LM)
        labels = input_ids.clone()
        # Mask padding tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        if debug_first_batch and not first["done"]:
            first["done"] = True
            print("=" * 80)
            print("[DEBUG] Example text (first item):")
            print(texts[0][:500])
            print("=" * 80)
            print(f"[DEBUG] Audio path: {audio_paths[0]}")
            print("=" * 80)
            print(f"[DEBUG] input_ids shape: {tuple(input_ids.shape)}")
            print(f"[DEBUG] attention_mask shape: {tuple(attention_mask.shape)}")
            print(f"[DEBUG] labels shape: {tuple(labels.shape)}")
            if "audio_features" in batch_inputs:
                print(f"[DEBUG] audio_features shape: {tuple(batch_inputs['audio_features'].shape)}")
            non_pad_tokens = (attention_mask > 0).sum().item()
            print(f"[DEBUG] total non-pad tokens in batch: {non_pad_tokens}")
            print("=" * 80)
            decoded_0 = processor.tokenizer.decode(
                input_ids[0],
                skip_special_tokens=False,
            )
            print("[DEBUG] Decoded first sequence (truncated to 1000 chars):")
            print(decoded_0[:1000])
            print("=" * 80)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        # Add audio features if present
        if "audio_features" in batch_inputs:
            result["audio_features"] = batch_inputs["audio_features"]
        if "audio_feature_lens" in batch_inputs:
            result["audio_feature_lens"] = batch_inputs["audio_feature_lens"]
            
        return result

    return collate_fn


def print_trainable_parameters(model: torch.nn.Module):
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")