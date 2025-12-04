import torch
import pickle
import json
import csv
import librosa
from datetime import datetime
from collections import defaultdict, Counter
from typing import Optional, List
from contextlib import contextmanager
from pathlib import Path
from collections import Counter

# Qwen2-Audio imports
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# --- 1. The Hook for Modifying Attention Masks ---

class BlockAttentionHook:
    """
    A forward pre-hook that dynamically modifies the attention mask during a model's
    forward pass to "knock out" specific attention pathways.
    """
    def __init__(self, knockout_mode, token_types, original_input_len):
        self.knockout_mode = knockout_mode
        self.token_types = token_types
        self.original_input_len = original_input_len
        self.num_blocks = 0  # Track how many times we actually block attention

    def __call__(self, module, args, kwargs):
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            return args, kwargs

        # During generation, q_len is 1, k_len is the total sequence length so far.
        k_len = attention_mask.shape[-1]
        query_pos = k_len - 1  # The absolute index of the current token being processed

        # Determine the type of the current query token
        query_type = 'generated' if query_pos >= self.original_input_len else self.token_types[query_pos]

        modified_mask = None  # Lazily clone the mask only if we need to modify it

        for key_pos in range(k_len):
            # The key token's type
            key_type = self.token_types[key_pos] if key_pos < self.original_input_len else 'generated'
            
            should_block = False
            if self.knockout_mode == 'generated_to_audio' and query_type == 'generated' and key_type == 'audio':
                should_block = True

            if should_block:
                if modified_mask is None:
                    modified_mask = attention_mask.clone()
                
                # Use additive mask with large negative number to block attention
                mask_value = torch.finfo(modified_mask.dtype).min
                # In generation, the query length is 1, so the relative query index is always 0
                modified_mask[..., 0, key_pos] = mask_value
                self.num_blocks += 1  # Count the block
        
        if modified_mask is not None:
            kwargs["attention_mask"] = modified_mask
        
        return args, kwargs

# --- 2. The Context Manager to Apply the Hooks ---

@contextmanager
def block_attention(model, knockout_mode, token_types, original_input_len, layers_to_knockout=None):
    """
    A context manager that applies the BlockAttentionHook to specified attention layers.
    
    Args:
        model: The Qwen2-Audio model
        knockout_mode: The type of knockout to apply ('generated_to_audio' or None)
        token_types: List mapping token positions to their types
        original_input_len: Length of the original input sequence
        layers_to_knockout: List of layer indices to apply knockout to (0-indexed).
                           If None, applies to all layers.
    """
    if knockout_mode is None:
        yield
        return

    # Get total number of decoder layers
    num_layers = len(model.language_model.model.layers)
    
    # Determine which layers to apply hooks to
    if layers_to_knockout is None:
        layers_to_knockout = list(range(num_layers))
    else:
        # Safety check: validate layer indices
        invalid_layers = [l for l in layers_to_knockout if l < 0 or l >= num_layers]
        if invalid_layers:
            raise ValueError(
                f"Invalid layer indices: {invalid_layers}. "
                f"Model has {num_layers} layers (valid range: 0-{num_layers-1})"
            )
        print(f"Applying knockout to specific layers: {layers_to_knockout}")

    hook_handles = []
    hook_objects = []  # Store hook objects to access their counters
    try:
        # For Qwen2-Audio, access the language model layers
        for layer_idx, layer in enumerate(model.language_model.model.layers):
            if layer_idx in layers_to_knockout:
                hook = BlockAttentionHook(knockout_mode, token_types, original_input_len)
                handle = layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True)
                hook_handles.append(handle)
                hook_objects.append(hook)
        
        print(f"Applied '{knockout_mode}' hooks to {len(hook_handles)}/{num_layers} attention layers.")
        yield hook_objects  # Return hook objects so caller can check counters
    finally:
        # Print statistics before removing hooks
        total_blocks = sum(hook.num_blocks for hook in hook_objects)
        print(f"Total attention positions blocked across all layers: {total_blocks}")
        if hook_objects:
            avg_blocks = total_blocks / len(hook_objects)
            print(f"Average blocks per layer: {avg_blocks:.2f}")
        
        for handle in hook_handles:
            handle.remove()
        print(f"Removed {len(hook_handles)} attention hooks.")

# --- 3. Data Capture and Saving ---

attention_storage = defaultdict(list)
hooks = []

def attention_hook_fn(layer_idx):
    def hook(module, input, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attention_storage[layer_idx].append(output[1].detach().cpu())
    return hook

def register_attention_hooks(model):
    global hooks
    for h in hooks: 
        h.remove()
    hooks.clear()
    attention_storage.clear()
    
    # Register hooks on Qwen2-Audio language model layers
    for i, layer in enumerate(model.language_model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(attention_hook_fn(i)))
    print(f"Registered {len(hooks)} attention hooks to capture weights.")

def save_knockout_data(filename, generated_text, token_map, knockout_mode, model_name):
    data = {
        'attention_weights_per_step': dict(attention_storage),
        'token_mapping': token_map,
        'generated_text': generated_text,
        'metadata': {
            'knockout_mode': knockout_mode,
            'model_name': model_name,
            'original_input_length': len(token_map),
            'num_layers': len(attention_storage),
            'timestamp': datetime.now().isoformat(),
        }
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Saved knockout experiment data to {filename}")

def load_csv_questions(csv_path):
    """
    Load questions from CSV file.
    Returns list of dictionaries with question data.
    """
    questions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                'filename': row['filename'],
                'question': row['question'],
                'options': {
                    'A': row['A'],
                    'B': row['B'],
                    'C': row['C'],
                    'D': row['D']
                },
                'correct_label': row['correct_label']
            })
    return questions

def create_token_type_mapping(input_ids, audio_token_id):
    """
    Create a mapping of token types for Qwen2-Audio.
    Only audio and text tokens exist (apart from generated tokens).
    """
    token_types = []
    for token_id in input_ids[0]:
        tid = token_id.item()
        if tid == audio_token_id:
            token_types.append("audio")
        else:
            token_types.append("text")
    return token_types

def format_question_with_options(question_data):
    """
    Format the question with multiple choice options for the model.
    """
    question = question_data['question']
    options = question_data['options']
    
    formatted = f"{question}\n"
    formatted += f"A) {options['A']}\n"
    formatted += f"B) {options['B']}\n"
    formatted += f"C) {options['C']}\n"
    formatted += f"D) {options['D']}\n"
    formatted += "Answer with just the letter (A, B, C, or D)."
    
    return formatted
    """
    Create a mapping of token types for Qwen2-Audio.
    Only audio and text tokens exist (apart from generated tokens).
    """
    token_types = []
    for token_id in input_ids[0]:
        tid = token_id.item()
        if tid == audio_token_id:
            token_types.append("audio")
        else:
            token_types.append("text")
    return token_types

# --- 4. Main Execution Block ---

if __name__ == "__main__":
    # Define which experiment to run: 'generated_to_audio' or None
    KNOCKOUT_MODE = 'generated_to_audio'  # or None for baseline
    
    # Define which layers to knockout (0-indexed)
    # Examples:
    # LAYERS_TO_KNOCKOUT = None  # Apply to all layers
    # LAYERS_TO_KNOCKOUT = [0, 1, 2]  # Only first 3 layers
    # LAYERS_TO_KNOCKOUT = list(range(10, 20))  # Layers 10-19
    LAYERS_TO_KNOCKOUT = [0, 1, 2, 3, 4]  # First 5 layers
    
    # CSV file path
    CSV_PATH = "/fs/gamma-projects/audio/kajayaku/temporal_understanding/curated_data/relative_mcqa_final/relative_earliest_start_mcq.csv"
    AUDIO_BASE_PATH = "/fs/gamma-projects/audio/apoorvak/tacos_dataset/audios_folder/"
    
    print("🚀 Loading Qwen2-Audio model and processor...")
    model_path = "Qwen/Qwen2-Audio-7B"
    
    print("Loading model...")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        attn_implementation="eager",  # Eager is needed for hooks to work
        device_map="auto"
    )
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path)

    # Register hooks to CAPTURE the final attention weights for analysis
    register_attention_hooks(model)
    
    # Load questions from CSV
    print(f"\n📋 Loading questions from CSV...")
    questions = load_csv_questions(CSV_PATH)
    print(f"Loaded {len(questions)} questions")
    
    # Get the audio token ID from the model config
    audio_token_id = processor.tokenizer.convert_tokens_to_ids("<|audio_bos|>")
    
    # Results storage
    results = []
    questions = questions[:1]  # For testing, limit to first 1 question  
    
    # Process each question
    for idx, q_data in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"Processing question {idx+1}/{len(questions)}: {q_data['filename']}")
        
        # Construct audio path
        audio_path = str(Path(AUDIO_BASE_PATH) / q_data['filename'])
        
        # Format question with options
        question_text = format_question_with_options(q_data)
        print(f"Question: {question_text}")
        
        # Prepare conversation
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": question_text},
            ]},
        ]
        
        # Process the conversation
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        # Get audio inputs
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        # Load audio file using librosa
                        audio_array, sr = librosa.load(ele["audio_url"], sr=processor.feature_extractor.sampling_rate)
                        audios.append(audio_array)
        
        inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        token_mapping = create_token_type_mapping(inputs['input_ids'], audio_token_id)
        original_input_len = len(token_mapping)
        
        # Count audio tokens for verification
        num_audio_tokens = sum(1 for t in token_mapping if t == 'audio')
        print(f"Token mapping - Total: {original_input_len}, Audio: {num_audio_tokens}, Text: {original_input_len - num_audio_tokens}")
        
        # Clear attention storage for this question
        attention_storage.clear()
        
        # Generate with knockout
        with block_attention(model, KNOCKOUT_MODE, token_mapping, original_input_len, LAYERS_TO_KNOCKOUT) as hooks:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=10,  # Short answer expected
                output_attentions=True,
                return_dict_in_generate=True,
            ).sequences
            
            # After generation, check how many blocks occurred
            if hooks:
                print(f"Verification: Expected blocks per generated token = {num_audio_tokens} audio tokens")
                print(f"Generated {len(output_ids[0]) - original_input_len} new tokens")
        
        # Decode the output
        generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # Extract just the answer (look for A, B, C, or D)
        answer = None
        for char in generated_text.upper():
            if char in ['A', 'B', 'C', 'D']:
                answer = char
                break
        
        # Store results
        result = {
            'filename': q_data['filename'],
            'question': q_data['question'],
            'options': q_data['options'],
            'correct_answer': q_data['correct_label'],
            'model_output': generated_text,
            'extracted_answer': answer,
            'is_correct': answer == q_data['correct_label'] if answer else False,
            'token_counts': dict(Counter(token_mapping)),
            'original_input_len': original_input_len
        }
        results.append(result)
        
        print(f"Model output: {generated_text}")
        print(f"Extracted answer: {answer}")
        print(f"Correct answer: {q_data['correct_label']}")
        print(f"Result: {'✓ CORRECT' if result['is_correct'] else '✗ INCORRECT'}")
    
    # Calculate accuracy
    correct_count = sum(1 for r in results if r['is_correct'])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"Accuracy: {correct_count}/{total_count} = {accuracy:.2%}")
    
    # Save results to JSON
    layer_suffix = f"_layers_{'_'.join(map(str, LAYERS_TO_KNOCKOUT))}" if LAYERS_TO_KNOCKOUT else "_all_layers"
    output_json = f"qwen2audio_mcqa_{KNOCKOUT_MODE}{layer_suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    
    output_data = {
        'metadata': {
            'knockout_mode': KNOCKOUT_MODE,
            'layers_knocked_out': LAYERS_TO_KNOCKOUT,
            'model_name': model_path,
            'csv_path': CSV_PATH,
            'total_questions': total_count,
            'correct_answers': correct_count,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat(),
        },
        'results': results
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved results to {output_json}")