"""
Enhanced Multi-Dimensional Activation Steering Experiment
Implements Capability, Style, and Density disentanglement with Gram-Schmidt orthogonalization.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from threading import Thread
import queue
from collections import deque

MODEL_ID = "./models/Qwen2.5-3B-Instruct"
DEFAULT_LAYER_INDEX = 20

# Global variable to store prompt variants
PROMPT_VARIANTS: Optional[Dict[str, List[str]]] = None


# ==================== Prompt Variant Generation ====================

def generate_prompt_variants(
    tokenizer,
    model,
    base_prompt: str,
    num_variants: int = 5,
    temperature: float = 0.8,
) -> List[str]:
    """
    Use Qwen to generate diverse paraphrases of a base prompt.
    """
    # Remove "System: " prefix if present for generation
    prompt_content = base_prompt.replace("System: ", "").strip()
    
    generation_prompt = f"""Generate {num_variants} different ways to express the following instruction. Each version should have the same meaning but use different words, sentence structure, or emphasis.

Original instruction:
{prompt_content}

Output {num_variants} paraphrased versions, one per line. Each line should start with "System: " followed by the paraphrased instruction."""

    inputs = tokenizer(generation_prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=1000,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    
    # Extract only the newly generated part (after the prompt)
    if generation_prompt in generated:
        generated = generated.split(generation_prompt)[-1].strip()
    
    # Extract variants from response
    variants = []
    
    # Strategy 1: Extract lines that start with "System:"
    lines = generated.split('\n')
    for line in lines:
        line = line.strip()
        # Skip empty lines, markdown, and continuation of the prompt
        if not line or line.startswith('#') or line.startswith('```') or len(line) < 20:
            continue
        
        # Look for lines starting with "System:"
        if line.startswith('System:'):
            variant = line.strip()
            # Ensure it's not just "System:" with nothing after
            if len(variant) > 10:
                variants.append(variant)
        # Also try lines that look like system prompts (numbered lists, etc.)
        elif re.match(r'^\d+[\.\)]\s*', line) or line.startswith('-'):
            # Remove numbering/markers
            cleaned = re.sub(r'^[\d\-\.\)\s]*', '', line).strip()
            # Add "System: " prefix if not present
            if cleaned and not cleaned.startswith('System:'):
                cleaned = f"System: {cleaned}"
            if len(cleaned) > 20:
                variants.append(cleaned)
    
    # Strategy 2: Try to find JSON array
    if not variants:
        try:
            json_match = re.search(r'\[.*?\]', generated, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    for v in parsed:
                        v_str = str(v).strip()
                        if v_str and len(v_str) > 20:
                            if not v_str.startswith('System:'):
                                v_str = f"System: {v_str}"
                            variants.append(v_str)
        except Exception:
            pass
    
    # Strategy 3: Extract any meaningful lines (fallback)
    if not variants:
        for line in lines:
            line = line.strip()
            if not line or len(line) < 30:
                continue
            # Skip if it looks like part of the prompt
            if any(keyword in line.lower() for keyword in ['generate', 'output', 'original', 'instruction']):
                continue
            # Add "System: " prefix
            if not line.startswith('System:'):
                line = f"System: {line}"
            variants.append(line)
            if len(variants) >= num_variants:
                break
    
    # Clean up variants: ensure they all start with "System:"
    cleaned_variants = []
    for v in variants:
        v = v.strip()
        if not v.startswith('System:'):
            v = f"System: {v}"
        # Remove duplicate "System:" if present
        v = re.sub(r'^System:\s*System:\s*', 'System: ', v)
        if len(v) > 20:  # Reasonable minimum length
            cleaned_variants.append(v)
    
    # If we have some variants but not enough, duplicate the last one or use base prompt
    if cleaned_variants and len(cleaned_variants) < num_variants:
        while len(cleaned_variants) < num_variants:
            cleaned_variants.append(cleaned_variants[-1] if cleaned_variants else base_prompt)
    
    # Final fallback: use base prompt
    if not cleaned_variants:
        print(f"Warning: Could not parse variants from Qwen output. Using base prompt.")
        print(f"Generated text preview: {generated[:200]}...")
        cleaned_variants = [base_prompt] * num_variants
    
    # Ensure we have exactly num_variants unique variants
    # Remove duplicates while preserving order
    seen = set()
    unique_variants = []
    for v in cleaned_variants:
        if v not in seen:
            seen.add(v)
            unique_variants.append(v)
        if len(unique_variants) >= num_variants:
            break
    
    # Fill remaining slots with base prompt if needed
    while len(unique_variants) < num_variants:
        unique_variants.append(base_prompt)
    
    return unique_variants[:num_variants]


def generate_all_prompt_variants(
    tokenizer,
    model,
    variants_file: str = "data/prompt_variants.json",
    num_variants_per_prompt: int = 5,
    force_regenerate: bool = False,
) -> Dict[str, List[str]]:
    """
    Generate variants for all prompt types and save to file.
    """
    variants_path = Path(variants_file)
    
    # Load existing variants if available
    if not force_regenerate and variants_path.exists():
        with open(variants_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
            print(f"Loaded existing prompt variants from {variants_path}")
            return existing
    
    print("Generating prompt variants using Qwen...")
    
    # Base prompts
    base_prompts = {
        "capability_expert": (
            "System: You are a world-renowned expert and Fields Medalist in this field. "
            "Please provide a rigorous, comprehensive, and theoretically profound explanation. "
            "Engage in deep reasoning, uncover non-obvious connections, and reveal the underlying essence."
        ),
        "capability_novice": (
            "System: You are a complete beginner who has just started learning this topic. "
            "Please answer based only on intuition and surface-level observations. "
            "Do not engage in deep reasoning or look for hidden connections."
        ),
        "style_professional": (
            "System: You are writing an academic paper. "
            "Use rigorous academic terminology, maintain an objective and formal tone. "
            "Write in a scholarly, written style. Avoid colloquialisms and casual language."
        ),
        "style_naive": (
            "System: You are explaining to a friend in a casual conversation. "
            "Use plain language and everyday analogies. "
            "Write as if you're chatting, using simple words and relatable examples."
        ),
        "density_comprehensive": (
            "System: Provide an extremely detailed explanation. "
            "Cover all aspects, background information, edge cases, and nuances. "
            "Do not omit any important details. Be thorough and exhaustive."
        ),
        "density_concise": (
            "System: Provide a very brief explanation. "
            "Summarize the core idea in as few words as possible. "
            "Omit all secondary details and get straight to the point."
        ),
        "correctness_correct": (
            "System: You must provide accurate, factually correct information. "
            "Verify your facts and ensure your answer is truthful and accurate. "
            "Do not make up information or state false claims."
        ),
        "correctness_incorrect": (
            "System: You are role-playing as someone who often makes mistakes. "
            "Provide factually incorrect information. "
            "Present incorrect conclusions confidently."
        ),
    }
    
    variants_dict = {}
    
    for prompt_key, base_prompt in tqdm(base_prompts.items(), desc="Generating variants"):
        print(f"\nGenerating variants for {prompt_key}...")
        variants = generate_prompt_variants(
            tokenizer, model, base_prompt, num_variants_per_prompt
        )
        
        # Validate variants - ensure they're not placeholders
        valid_variants = []
        for v in variants:
            # Check if it's a placeholder or too short
            if (not v.startswith('System:') or 
                len(v) < 30 or 
                'paraphrase' in v.lower() and len(v) < 50):
                # Use base prompt instead
                valid_variants.append(base_prompt)
            else:
                valid_variants.append(v)
        
        # Ensure we have unique variants
        seen = set()
        final_variants = []
        for v in valid_variants:
            if v not in seen:
                seen.add(v)
                final_variants.append(v)
            if len(final_variants) >= num_variants_per_prompt:
                break
        
        # Fill with base prompt if needed
        while len(final_variants) < num_variants_per_prompt:
            final_variants.append(base_prompt)
        
        variants_dict[prompt_key] = final_variants[:num_variants_per_prompt]
        print(f"  Generated {len(variants_dict[prompt_key])} valid variants")
        # Show first variant as preview
        if variants_dict[prompt_key]:
            preview = variants_dict[prompt_key][0][:80] + "..." if len(variants_dict[prompt_key][0]) > 80 else variants_dict[prompt_key][0]
            print(f"  Preview: {preview}")
    
    # Save variants
    variants_path.parent.mkdir(parents=True, exist_ok=True)
    with open(variants_path, 'w', encoding='utf-8') as f:
        json.dump(variants_dict, f, ensure_ascii=False, indent=2)
    print(f"\nSaved prompt variants to {variants_path}")
    
    return variants_dict


def load_prompt_variants(variants_file: str = "data/prompt_variants.json") -> Dict[str, List[str]]:
    """Load prompt variants from file."""
    global PROMPT_VARIANTS
    
    if PROMPT_VARIANTS is not None:
        return PROMPT_VARIANTS
    
    variants_path = Path(variants_file)
    if variants_path.exists():
        with open(variants_path, 'r', encoding='utf-8') as f:
            PROMPT_VARIANTS = json.load(f)
        print(f"Loaded prompt variants from {variants_path}")
    else:
        PROMPT_VARIANTS = {}
        print(f"Prompt variants file not found at {variants_path}. Using base prompts only.")
    
    return PROMPT_VARIANTS


def get_prompt_variant(prompt_key: str, use_variants: bool = True, variant_index: Optional[int] = None) -> str:
    """
    Get a prompt variant for a given key.
    If variants are loaded and use_variants=True, select one (randomly or by index).
    Otherwise, return the base prompt.
    """
    variants_dict = load_prompt_variants()
    
    if prompt_key in variants_dict and use_variants:
        variants = variants_dict[prompt_key]
        if variants:
            if variant_index is not None:
                return variants[variant_index % len(variants)]
            else:
                import random
                return random.choice(variants)
    
    # Fallback to base prompts
    base_prompts = {
        "capability_expert": (
            "System: You are a world-renowned expert and Fields Medalist in this field. "
            "Please provide a rigorous, comprehensive, and theoretically profound explanation. "
            "Engage in deep reasoning, uncover non-obvious connections, and reveal the underlying essence."
        ),
        "capability_novice": (
            "System: You are a complete beginner who has just started learning this topic. "
            "Please answer based only on intuition and surface-level observations. "
            "Do not engage in deep reasoning or look for hidden connections."
        ),
        "style_professional": (
            "System: You are writing an academic paper. "
            "Use rigorous academic terminology, maintain an objective and formal tone. "
            "Write in a scholarly, written style. Avoid colloquialisms and casual language."
        ),
        "style_naive": (
            "System: You are explaining to a friend in a casual conversation. "
            "Use plain language and everyday analogies. "
            "Write as if you're chatting, using simple words and relatable examples."
        ),
        "density_comprehensive": (
            "System: Provide an extremely detailed explanation. "
            "Cover all aspects, background information, edge cases, and nuances. "
            "Do not omit any important details. Be thorough and exhaustive."
        ),
        "density_concise": (
            "System: Provide a very brief explanation. "
            "Summarize the core idea in as few words as possible. "
            "Omit all secondary details and get straight to the point."
        ),
        "correctness_correct": (
            "System: You must provide accurate, factually correct information. "
            "Verify your facts and ensure your answer is truthful and accurate. "
            "Do not make up information or state false claims."
        ),
        "correctness_incorrect": (
            "System: You are role-playing as someone who often makes mistakes. "
            "Provide plausible-sounding but factually incorrect information. "
            "Mix some truth with falsehoods, and present incorrect conclusions confidently."
        ),
    }
    return base_prompts.get(prompt_key, "")


# ==================== Prompt Templates for Three Axes ====================

def get_capability_prompts(
    question: str, 
    answer: str, 
    level: str,
    use_variants: bool = True,
    variant_index: Optional[int] = None,
) -> str:
    """Generate prompts for Capability axis (Expert vs Novice)."""
    if level == "expert":
        sys_prompt = get_prompt_variant("capability_expert", use_variants, variant_index)
    elif level == "novice":
        sys_prompt = get_prompt_variant("capability_novice", use_variants, variant_index)
    else:
        sys_prompt = (
            "System: You are a knowledgeable student with moderate understanding of this topic. "
            "Please provide a balanced explanation that is neither too superficial nor too advanced."
        )
    
    return f"{sys_prompt}\nUser: {question}\nAssistant: {answer}"


def get_style_prompts(
    question: str, 
    answer: str, 
    style: str,
    use_variants: bool = True,
    variant_index: Optional[int] = None,
) -> str:
    """Generate prompts for Style axis (Professional vs Naive)."""
    if style == "professional":
        sys_prompt = get_prompt_variant("style_professional", use_variants, variant_index)
    elif style == "naive":
        sys_prompt = get_prompt_variant("style_naive", use_variants, variant_index)
    else:
        sys_prompt = (
            "System: You are explaining in a balanced, accessible manner. "
            "Use clear language that is neither too formal nor too casual."
        )
    
    return f"{sys_prompt}\nUser: {question}\nAssistant: {answer}"


def get_density_prompts(
    question: str, 
    answer: str, 
    density: str,
    use_variants: bool = True,
    variant_index: Optional[int] = None,
) -> str:
    """Generate prompts for Density axis (Comprehensive vs Concise)."""
    if density == "comprehensive":
        sys_prompt = get_prompt_variant("density_comprehensive", use_variants, variant_index)
    elif density == "concise":
        sys_prompt = get_prompt_variant("density_concise", use_variants, variant_index)
    else:
        sys_prompt = (
            "System: Provide a balanced explanation. "
            "Include the main points without excessive detail or unnecessary brevity."
        )
    
    return f"{sys_prompt}\nUser: {question}\nAssistant: {answer}"


def get_correctness_prompts(
    question: str, 
    answer: str, 
    correctness: str,
    use_variants: bool = True,
    variant_index: Optional[int] = None,
) -> str:
    """Generate prompts for Correctness axis (Correct vs Incorrect)."""
    if correctness == "correct":
        sys_prompt = get_prompt_variant("correctness_correct", use_variants, variant_index)
    elif correctness == "incorrect":
        sys_prompt = get_prompt_variant("correctness_incorrect", use_variants, variant_index)
    else:
        sys_prompt = (
            "System: Provide an answer that may contain both correct and incorrect elements. "
            "Do not strictly enforce accuracy."
        )
    
    return f"{sys_prompt}\nUser: {question}\nAssistant: {answer}"


# ==================== Data Loading and Formatting ====================

def load_capability_data(file_path: str) -> List[Dict]:
    """Load capability dataset (Q&A pairs)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def format_multi_axis_pairs(
    data: List[Dict],
    use_variants: bool = True,
    cycle_variants: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Format data into pairs for extracting four vectors:
    - v_cap: Expert - Novice (fixed Professional style, Comprehensive density)
    - v_style: Professional - Naive (fixed medium capability, medium density)
    - v_density: Comprehensive - Concise (fixed medium capability, medium style)
    - v_correctness: Correct - Incorrect (fixed medium capability, medium style, medium density)
    """
    pairs = {
        "capability": [],      # For v_cap extraction
        "style": [],           # For v_style extraction
        "density": [],         # For v_density extraction
        "correctness": []      # For v_correctness extraction
    }
    
    for idx, item in enumerate(data):
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        if not question or not answer:
            continue
        
        # Determine variant index if cycling through variants
        variant_idx = idx if cycle_variants else None
        
        # v_cap pairs: Expert vs Novice (fixed Professional + Comprehensive)
        pairs["capability"].append({
            "question": question,
            "answer": answer,
            "text_high": get_capability_prompts(question, answer, "expert", use_variants, variant_idx),
            "text_low": get_capability_prompts(question, answer, "novice", use_variants, variant_idx),
            "target_answer": answer
        })
        
        # v_style pairs: Professional vs Naive (fixed medium capability, medium density)
        # Use a medium-level answer (simplified version) for style extraction
        pairs["style"].append({
            "question": question,
            "answer": answer,  # Still use same answer for teacher forcing
            "text_high": get_style_prompts(question, answer, "professional", use_variants, variant_idx),
            "text_low": get_style_prompts(question, answer, "naive", use_variants, variant_idx),
            "target_answer": answer
        })
        
        # v_density pairs: Comprehensive vs Concise (fixed medium capability, medium style)
        pairs["density"].append({
            "question": question,
            "answer": answer,
            "text_high": get_density_prompts(question, answer, "comprehensive", use_variants, variant_idx),
            "text_low": get_density_prompts(question, answer, "concise", use_variants, variant_idx),
            "target_answer": answer
        })
        
        # v_correctness pairs: Correct vs Incorrect (fixed medium capability, medium style, medium density)
        # For correctness, we use the correct answer for "high" and need to generate an incorrect answer for "low"
        # For teacher forcing, we'll use the same answer but with different correctness prompts
        pairs["correctness"].append({
            "question": question,
            "answer": answer,  # Use same answer for teacher forcing
            "text_high": get_correctness_prompts(question, answer, "correct", use_variants, variant_idx),
            "text_low": get_correctness_prompts(question, answer, "incorrect", use_variants, variant_idx),
            "target_answer": answer
        })
    
    print(f"Formatted {len(pairs['capability'])} capability pairs")
    print(f"Formatted {len(pairs['style'])} style pairs")
    print(f"Formatted {len(pairs['density'])} density pairs")
    print(f"Formatted {len(pairs['correctness'])} correctness pairs")
    
    return pairs


# ==================== Model Loading ====================

def load_qwen(model_id: str = MODEL_ID):
    """Load Qwen model with hidden-states enabled."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


# ==================== Activation Extraction ====================

def extract_answer_hidden_states(
    tokenizer,
    model,
    text: str,
    answer_text: str,
    layer_index: int,
) -> np.ndarray:
    """Extract hidden states for answer tokens at specified layer."""
    with torch.no_grad():
        encoded_full = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        encoded_ans = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False)

        input_ids = encoded_full["input_ids"].to(model.device)
        attention_mask = encoded_full["attention_mask"].to(model.device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        hidden_states = out.hidden_states[layer_index]  # [1, seq_len, d_model]
        ans_len = encoded_ans["input_ids"].shape[-1]
        answer_slice = hidden_states[:, -ans_len:, :]  # [1, T_ans, d_model]
        return answer_slice.squeeze(0).float().cpu().numpy()


def compute_vector(
    pairs: List[Dict],
    tokenizer,
    model,
    layer_indices: List[int],
    max_samples: int = 128,
    vector_name: str = "vector",
) -> Dict[int, np.ndarray]:
    """
    Compute a difference vector using teacher forcing.
    Returns dict mapping layer_index -> vector.
    """
    layer_diffs: Dict[int, List[np.ndarray]] = {layer_idx: [] for layer_idx in layer_indices}
    used = 0
    limit = max_samples if max_samples > 0 else None

    for item in tqdm(pairs, desc=f"Computing {vector_name} for {len(layer_indices)} layers"):
        if limit is not None and used >= limit:
            break

        text_low = item["text_low"]
        text_high = item["text_high"]
        target_answer = item["target_answer"]

        try:
            with torch.no_grad():
                # Process low version
                encoded_full_low = tokenizer(text_low, return_tensors="pt", add_special_tokens=True)
                encoded_ans = tokenizer(target_answer, return_tensors="pt", add_special_tokens=False)
                
                input_ids_low = encoded_full_low["input_ids"].to(model.device)
                attention_mask_low = encoded_full_low["attention_mask"].to(model.device)
                
                out_low = model(
                    input_ids=input_ids_low,
                    attention_mask=attention_mask_low,
                    output_hidden_states=True,
                    use_cache=False,
                )

                # Process high version
                encoded_full_high = tokenizer(text_high, return_tensors="pt", add_special_tokens=True)
                input_ids_high = encoded_full_high["input_ids"].to(model.device)
                attention_mask_high = encoded_full_high["attention_mask"].to(model.device)
                
                out_high = model(
                    input_ids=input_ids_high,
                    attention_mask=attention_mask_high,
                    output_hidden_states=True,
                    use_cache=False,
                )

                ans_len = encoded_ans["input_ids"].shape[-1]

                # Extract answer slices for each layer
                for layer_idx in layer_indices:
                    h_low = out_low.hidden_states[layer_idx][:, -ans_len:, :].squeeze(0).float().cpu().numpy()
                    h_high = out_high.hidden_states[layer_idx][:, -ans_len:, :].squeeze(0).float().cpu().numpy()

                    if h_low.shape != h_high.shape:
                        continue

                    delta = h_high - h_low  # [T, D]
                    v_i = delta.mean(axis=0)  # [D]
                    layer_diffs[layer_idx].append(v_i)

        except Exception as e:
            print(f"[compute_vector] Skipping sample due to error: {e}")
            continue

        used += 1

    # Compute final vector for each layer
    vector_dict: Dict[int, np.ndarray] = {}
    for layer_idx in layer_indices:
        diffs = layer_diffs[layer_idx]
        if not diffs:
            raise RuntimeError(f"No valid pairs processed for layer {layer_idx}.")

        diffs_arr = np.stack(diffs, axis=0)  # [N, D]
        vector = diffs_arr.mean(axis=0)  # [D]
        vector_dict[layer_idx] = vector

        # Diagnostics
        per_sample_norms = np.linalg.norm(diffs_arr, axis=1)
        vector_norm = np.linalg.norm(vector)
        print(
            f"[{vector_name} Layer {layer_idx}] Used {len(diffs)} pairs, "
            f"||vector|| = {vector_norm:.4f}, "
            f"mean(||v_i||) = {per_sample_norms.mean():.4f}"
        )

    return vector_dict


# ==================== Gram-Schmidt Orthogonalization ====================

def gram_schmidt_orthogonalize(vectors: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply Gram-Schmidt orthogonalization to a list of vectors.
    Returns orthogonalized vectors.
    """
    orthogonalized = []
    for i, v in enumerate(vectors):
        v_ortho = v.copy()
        for u in orthogonalized:
            # Project v onto u and subtract
            proj = np.dot(v_ortho, u) / np.dot(u, u) * u
            v_ortho = v_ortho - proj
        # Normalize
        norm = np.linalg.norm(v_ortho)
        if norm > 1e-10:
            v_ortho = v_ortho / norm
        orthogonalized.append(v_ortho)
    return orthogonalized


def compute_orthogonalized_vectors(
    v_cap_dict: Dict[int, np.ndarray],
    v_style_dict: Dict[int, np.ndarray],
    v_density_dict: Dict[int, np.ndarray],
    v_correctness_dict: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Apply Gram-Schmidt orthogonalization to get pure vectors.
    
    For each layer:
    1. Orthogonalize noise vectors (v_style, v_density, v_correctness)
    2. Project v_cap onto noise subspace and subtract to get v_cap_pure
    
    Returns dict mapping layer_index -> {
        'v_cap': original,
        'v_style': orthogonalized,
        'v_density': orthogonalized,
        'v_correctness': orthogonalized (if provided),
        'v_cap_pure': pure capability vector
    }
    """
    layers = set(v_cap_dict.keys()) & set(v_style_dict.keys()) & set(v_density_dict.keys())
    if v_correctness_dict:
        layers = layers & set(v_correctness_dict.keys())
    
    result = {}
    
    for layer_idx in layers:
        v_cap = v_cap_dict[layer_idx]
        v_style = v_style_dict[layer_idx]
        v_density = v_density_dict[layer_idx]
        
        # Step 1: Orthogonalize noise vectors
        noise_vectors = [v_style, v_density]
        if v_correctness_dict:
            noise_vectors.append(v_correctness_dict[layer_idx])
        
        noise_ortho = gram_schmidt_orthogonalize(noise_vectors)
        v_style_ortho = noise_ortho[0]
        v_density_ortho = noise_ortho[1]
        v_correctness_ortho = noise_ortho[2] if len(noise_ortho) > 2 else None
        
        # Step 2: Project v_cap onto noise subspace
        proj_style = np.dot(v_cap, v_style_ortho) * v_style_ortho
        proj_density = np.dot(v_cap, v_density_ortho) * v_density_ortho
        proj_correctness = np.dot(v_cap, v_correctness_ortho) * v_correctness_ortho if v_correctness_ortho is not None else np.zeros_like(v_cap)
        
        # Step 3: Subtract projections to get pure capability vector
        v_cap_pure = v_cap - proj_style - proj_density - proj_correctness
        v_style_ortho = v_style_ortho * np.linalg.norm(v_style)
        v_density_ortho = v_density_ortho * np.linalg.norm(v_density)
        v_correctness_ortho = v_correctness_ortho * np.linalg.norm(v_correctness_dict[layer_idx]) if v_correctness_ortho is not None else None
        
        # Compute cosine similarities for diagnostics
        cos_style_cap = np.dot(v_style, v_cap) / (np.linalg.norm(v_style) * np.linalg.norm(v_cap))
        cos_density_cap = np.dot(v_density, v_cap) / (np.linalg.norm(v_density) * np.linalg.norm(v_cap))
        cos_style_density = np.dot(v_style, v_density) / (np.linalg.norm(v_style) * np.linalg.norm(v_density))
        
        result_dict = {
            'v_cap': v_cap,
            'v_style': v_style_ortho,
            'v_density': v_density_ortho,
            'v_cap_pure': v_cap_pure
        }
        
        if v_correctness_ortho is not None:
            cos_correctness_cap = np.dot(v_correctness_dict[layer_idx], v_cap) / (np.linalg.norm(v_correctness_dict[layer_idx]) * np.linalg.norm(v_cap))
            result_dict['v_correctness'] = v_correctness_ortho
            print(f"\n[Layer {layer_idx} Orthogonalization]")
            print(f"  Cosine(v_style, v_cap) = {cos_style_cap:.4f}")
            print(f"  Cosine(v_density, v_cap) = {cos_density_cap:.4f}")
            print(f"  Cosine(v_correctness, v_cap) = {cos_correctness_cap:.4f}")
            print(f"  Cosine(v_style, v_density) = {cos_style_density:.4f}")
            print(f"  ||v_cap|| = {np.linalg.norm(v_cap):.4f}")
            print(f"  ||v_cap_pure|| = {np.linalg.norm(v_cap_pure):.4f}")
            print(f"  Projection removed: {np.linalg.norm(proj_style + proj_density + proj_correctness):.4f}")
        else:
            print(f"\n[Layer {layer_idx} Orthogonalization]")
            print(f"  Cosine(v_style, v_cap) = {cos_style_cap:.4f}")
            print(f"  Cosine(v_density, v_cap) = {cos_density_cap:.4f}")
            print(f"  Cosine(v_style, v_density) = {cos_style_density:.4f}")
            print(f"  ||v_cap|| = {np.linalg.norm(v_cap):.4f}")
            print(f"  ||v_cap_pure|| = {np.linalg.norm(v_cap_pure):.4f}")
            print(f"  Projection removed: {np.linalg.norm(proj_style + proj_density):.4f}")
        
        result[layer_idx] = result_dict
    
    return result


# ==================== Multi-Vector Steering ====================

class MultiVectorActivationSteerer:
    """Steering with multiple independent vectors (capability, style, density, correctness)."""
    
    def __init__(
        self,
        model,
        vectors_dict: Dict[int, Dict[str, np.ndarray]],
        alpha_cap: float = 0.0,
        alpha_style: float = 0.0,
        alpha_density: float = 0.0,
        alpha_correctness: float = 0.0,
    ):
        """
        Args:
            model: Transformer model
            vectors_dict: Dict mapping layer_index -> {'v_cap', 'v_style', 'v_density', 'v_correctness', 'v_cap_pure'}
            alpha_cap: Steering strength for capability (use v_cap_pure)
            alpha_style: Steering strength for style (negative = suppress jargon)
            alpha_density: Steering strength for density (negative = suppress verbosity)
            alpha_correctness: Steering strength for correctness (positive = correct, negative = incorrect)
        """
        self.model = model
        self.vectors_dict = {
            layer_idx: {
                key: torch.tensor(vec, dtype=torch.float32, device="cpu")
                for key, vec in vecs.items()
            }
            for layer_idx, vecs in vectors_dict.items()
        }
        # Use list to make alpha values mutable for dynamic updates
        self.alpha_cap = [alpha_cap]
        self.alpha_style = [alpha_style]
        self.alpha_density = [alpha_density]
        self.alpha_correctness = [alpha_correctness]
        self.handles = []
    
    def update_alphas(self, alpha_cap=None, alpha_style=None, alpha_density=None, alpha_correctness=None):
        """Update alpha values dynamically during generation."""
        updated = False
        if alpha_cap is not None:
            self.alpha_cap[0] = float(alpha_cap)
            updated = True
        if alpha_style is not None:
            self.alpha_style[0] = float(alpha_style)
            updated = True
        if alpha_density is not None:
            self.alpha_density[0] = float(alpha_density)
            updated = True
        if alpha_correctness is not None:
            self.alpha_correctness[0] = float(alpha_correctness)
            updated = True
        
        if updated:
            print(f"[MultiVectorSteerer] Updated alphas: cap={self.alpha_cap[0]}, style={self.alpha_style[0]}, density={self.alpha_density[0]}, correctness={self.alpha_correctness[0]}")
        
        return updated
    
    def __enter__(self):
        # Store reference to self for hooks to access
        steerer_self = self
        
        for layer_idx, vecs in self.vectors_dict.items():
            def make_hook(layer_vecs, steerer_instance):
                def hook(module, inputs, output):
                    if not isinstance(output, torch.Tensor):
                        return output
                    
                    # Get vectors on correct device
                    v_cap_pure = layer_vecs['v_cap_pure'].to(output.device, dtype=output.dtype)
                    v_style = layer_vecs['v_style'].to(output.device, dtype=output.dtype)
                    v_density = layer_vecs['v_density'].to(output.device, dtype=output.dtype)
                    
                    # Apply steering - read alpha values dynamically from steerer instance
                    # This ensures we always read the latest values
                    delta = 0.0
                    if steerer_instance.alpha_cap[0] != 0.0:
                        delta = delta + steerer_instance.alpha_cap[0] * v_cap_pure.view(1, 1, -1)
                    if steerer_instance.alpha_style[0] != 0.0:
                        delta = delta + steerer_instance.alpha_style[0] * v_style.view(1, 1, -1)
                    if steerer_instance.alpha_density[0] != 0.0:
                        delta = delta + steerer_instance.alpha_density[0] * v_density.view(1, 1, -1)
                    
                    # Add correctness if available
                    if 'v_correctness' in layer_vecs and steerer_instance.alpha_correctness[0] != 0.0:
                        v_correctness = layer_vecs['v_correctness'].to(output.device, dtype=output.dtype)
                        delta = delta + steerer_instance.alpha_correctness[0] * v_correctness.view(1, 1, -1)
                    
                    return output + delta
                return hook
            
            block = self.model.model.layers[layer_idx]
            handle = block.register_forward_hook(make_hook(vecs, steerer_self))
            self.handles.append((layer_idx, handle))
        
        print(f"[MultiVectorSteerer] Registered hooks on {len(self.handles)} layers")
        print(f"  alpha_cap={self.alpha_cap[0]}, alpha_style={self.alpha_style[0]}, alpha_density={self.alpha_density[0]}, alpha_correctness={self.alpha_correctness[0]}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove all hooks safely
        if self.handles:
            # Use list copy to avoid modification during iteration
            handles_copy = list(self.handles)
            for layer_idx, handle in handles_copy:
                try:
                    handle.remove()
                except Exception as e:
                    # Hook might already be removed or invalid
                    pass
            self.handles.clear()
        
        # Clear references to help with garbage collection
        if hasattr(self, 'vectors_dict'):
            del self.vectors_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==================== Generation ====================

class StopOnPatternCriteria(StoppingCriteria):
    """Stop generation when User: or System: patterns are detected."""
    
    def __init__(self, tokenizer, stop_patterns=None):
        super().__init__()
        self.tokenizer = tokenizer
        if stop_patterns is None:
            self.stop_patterns = ["\nUser:", "\nSystem:", "User:", "System:"]
        else:
            self.stop_patterns = stop_patterns
        self.generated_text = ""
    
    def __call__(self, input_ids, scores, **kwargs):
        # Handle both single and batched inputs
        # For batched inputs, check the first sequence (all should be similar)
        batch_idx = 0
        generated_text = self.tokenizer.decode(input_ids[batch_idx], skip_special_tokens=True)
        self.generated_text = generated_text
        
        # Check if any stop pattern appears in the generated text
        for pattern in self.stop_patterns:
            if pattern in generated_text:
                # Check if it's in the assistant response part (after "Assistant:")
                if "Assistant:" in generated_text:
                    assistant_part = generated_text.split("Assistant:")[-1]
                    if pattern in assistant_part:
                        return True
                else:
                    # If no "Assistant:" found, check the whole text
                    if pattern in generated_text:
                        return True
        
        return False


class ForceStopCriteria(StoppingCriteria):
    """Stop generation when force_stop flag is set."""
    
    def __init__(self, stop_flag):
        super().__init__()
        self.stop_flag = stop_flag  # Should be a list or dict with mutable flag
    
    def __call__(self, input_ids, scores, **kwargs):
        # Check if stop was requested
        if isinstance(self.stop_flag, list):
            return self.stop_flag[0] if self.stop_flag else False
        elif isinstance(self.stop_flag, dict):
            return self.stop_flag.get('stop', False)
        else:
            return bool(self.stop_flag)


def generate_with_persona(
    tokenizer,
    model,
    question: str,
    persona: str = "low",
    max_new_tokens: int = 1024,
) -> str:
    """Generate text with specified persona."""
    if persona == "low":
        sys_prompt = (
            "System: You are a curious 5-year-old child. "
            "Please explain things using very simple words and basic intuition."
        )
    elif persona == "high":
        sys_prompt = (
            "System: You are a world-renowned expert and Fields Medalist. "
            "Please provide a rigorous, comprehensive, and theoretically profound explanation."
        )
    else:
        sys_prompt = ""

    full_prompt = f"{sys_prompt}\nUser: {question}\nAssistant:"

    inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=True).to(model.device)

    # Create stopping criteria to stop on User: or System: patterns
    stop_criteria = StopOnPatternCriteria(tokenizer)
    stopping_criteria = StoppingCriteriaList([stop_criteria])

    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria,
            use_cache=True,  # Explicitly use cache
        )

    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    
    # Clean up tensors and clear GPU cache
    del inputs, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Extract only the Assistant's response
    split_tok = "Assistant:"
    if split_tok in generated:
        # Take everything after the last "Assistant:" occurrence
        parts = generated.split(split_tok)
        assistant_response = parts[-1].strip()
    else:
        assistant_response = generated.strip()
    
    # Clean up: remove any remaining stop patterns that might have been generated
    # (in case stopping criteria didn't catch them exactly)
    stop_patterns = ["\nUser:", "\nSystem:", "User:", "System:"]
    earliest_idx = len(assistant_response)
    for pattern in stop_patterns:
        idx = assistant_response.find(pattern)
        if idx != -1 and idx < earliest_idx:
            earliest_idx = idx
    
    if earliest_idx < len(assistant_response):
        assistant_response = assistant_response[:earliest_idx].strip()
    
    return assistant_response


def generate_with_persona_batch(
    tokenizer,
    model,
    question: str,
    persona: str = "low",
    num_responses: int = 1,
    max_new_tokens: int = 1024,
) -> List[str]:
    """Generate multiple responses in batch for the same question."""
    if persona == "low":
        sys_prompt = (
            "System: You are a curious 5-year-old child. "
            "Please explain things using very simple words and basic intuition."
        )
    elif persona == "high":
        sys_prompt = (
            "System: You are a world-renowned expert and Fields Medalist. "
            "Please provide a rigorous, comprehensive, and theoretically profound explanation."
        )
    else:
        sys_prompt = ""

    full_prompt = f"{sys_prompt}\nUser: {question}\nAssistant:"

    # Create batch of identical prompts
    prompts = [full_prompt] * num_responses
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)

    # Create stopping criteria to stop on User: or System: patterns
    stop_criteria = StopOnPatternCriteria(tokenizer)
    stopping_criteria = StoppingCriteriaList([stop_criteria])

    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria,
            use_cache=True,
            num_return_sequences=1,  # One sequence per input
        )

    # Decode all responses
    generated_texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    
    # Process each response
    assistant_responses = []
    for generated in generated_texts:
        # Extract only the Assistant's response
        split_tok = "Assistant:"
        if split_tok in generated:
            parts = generated.split(split_tok)
            assistant_response = parts[-1].strip()
        else:
            assistant_response = generated.strip()
        
        # Clean up: remove any remaining stop patterns
        stop_patterns = ["\nUser:", "\nSystem:", "User:", "System:"]
        earliest_idx = len(assistant_response)
        for pattern in stop_patterns:
            idx = assistant_response.find(pattern)
            if idx != -1 and idx < earliest_idx:
                earliest_idx = idx
        
        if earliest_idx < len(assistant_response):
            assistant_response = assistant_response[:earliest_idx].strip()
        
        assistant_responses.append(assistant_response)
    
    # Clean up tensors and clear GPU cache
    del inputs, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return assistant_responses


def generate_with_persona_streaming(
    tokenizer,
    model,
    question: str,
    persona: str = "low",
    max_new_tokens: int = 1024,
    stop_flag=None,
):
    """Generate text with specified persona using streaming. Stops on User: or System: patterns.
    
    Args:
        tokenizer: Tokenizer for the model
        model: Model to generate with
        question: Question to answer
        persona: Persona mode ('low' or 'high')
        max_new_tokens: Maximum number of tokens to generate
        stop_flag: Optional mutable flag (list or dict) to force stop generation
    """
    if persona == "low":
        sys_prompt = (
            "System: You are a curious 5-year-old child. "
            "Please explain things using very simple words and basic intuition."
        )
    elif persona == "high":
        sys_prompt = (
            "System: You are a world-renowned expert and Fields Medalist. "
            "Please provide a rigorous, comprehensive, and theoretically profound explanation."
        )
    else:
        sys_prompt = ""

    full_prompt = f"{sys_prompt}\nUser: {question}\nAssistant:"

    inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=True).to(model.device)

    # Create streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Create stopping criteria
    stop_criteria = StopOnPatternCriteria(tokenizer)
    stopping_criteria_list = [stop_criteria]
    
    # Add force stop criteria if stop_flag is provided
    if stop_flag is not None:
        force_stop_criteria = ForceStopCriteria(stop_flag)
        stopping_criteria_list.append(force_stop_criteria)
    
    stopping_criteria = StoppingCriteriaList(stopping_criteria_list)
    
    # Generation kwargs
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
        "stopping_criteria": stopping_criteria,
    }
    
    # Track accumulated text to detect stop patterns
    accumulated_text = ""
    stop_patterns = ["\nUser:", "\nSystem:", "User:", "System:"]
    
    # Thread reference for potential interruption
    generation_thread = None
    
    # Start generation in a separate thread with no_grad
    def generate_with_no_grad():
        try:
            with torch.no_grad():
                model.generate(**generation_kwargs)
        except Exception as e:
            # If generation is interrupted, this is expected
            pass
        finally:
            # Clean up after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    generation_thread = Thread(target=generate_with_no_grad, daemon=True)
    generation_thread.start()
    
    # Yield tokens as they are generated, checking for stop patterns
    try:
        for new_text in streamer:
            accumulated_text += new_text
            
            # Check if any stop pattern appears in the assistant response part
            # (after "Assistant:" if it exists, otherwise in the whole text)
            text_to_check = accumulated_text.split("Assistant:")[-1] if "Assistant:" in accumulated_text else accumulated_text
            
            # Find the earliest stop pattern
            earliest_idx = len(text_to_check)
            found_pattern = None
            for pattern in stop_patterns:
                idx = text_to_check.find(pattern)
                if idx != -1 and idx < earliest_idx:
                    earliest_idx = idx
                    found_pattern = pattern
            
            if found_pattern:
                # Truncate at the stop pattern
                if "Assistant:" in accumulated_text:
                    # Calculate the actual position in accumulated_text
                    assistant_start = accumulated_text.find("Assistant:") + len("Assistant:")
                    actual_stop_pos = assistant_start + earliest_idx
                    truncated = accumulated_text[:actual_stop_pos].strip()
                else:
                    truncated = accumulated_text[:earliest_idx].strip()
                
                # Yield only the part before the stop pattern (if we haven't already)
                if truncated != accumulated_text:
                    # Calculate what we've already yielded
                    already_yielded_len = len(accumulated_text) - len(new_text)
                    if len(truncated) > already_yielded_len:
                        yield truncated[already_yielded_len:]
                
                # Stop yielding - the stopping criteria will stop generation
                break
            
            yield new_text
    finally:
        # Wait for thread to finish (it should stop due to stopping criteria)
        # Use timeout to prevent hanging
        if generation_thread is not None:
            generation_thread.join(timeout=5.0)  # 5 second timeout
        
        # Clean up GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()


def generate_with_persona_streaming_parallel(
    tokenizer,
    model,
    question: str,
    persona: str = "low",
    num_responses: int = 1,
    max_new_tokens: int = 1024,
    stop_flag=None,
):
    """Generate multiple responses in parallel using streaming. Yields chunks from all responses as they arrive.
    
    Args:
        tokenizer: Tokenizer for the model
        model: Model to generate with
        question: Question to answer
        persona: Persona mode ('low' or 'high')
        num_responses: Number of responses to generate in parallel
        max_new_tokens: Maximum number of tokens to generate
        stop_flag: Optional mutable flag (list or dict) to force stop generation
    
    Yields:
        Tuple of (response_id, chunk) for each chunk from any response
    """
    if persona == "low":
        sys_prompt = (
            "System: You are a curious 5-year-old child. "
            "Please explain things using very simple words and basic intuition."
        )
    elif persona == "high":
        sys_prompt = (
            "System: You are a world-renowned expert and Fields Medalist. "
            "Please provide a rigorous, comprehensive, and theoretically profound explanation."
        )
    else:
        sys_prompt = ""

    full_prompt = f"{sys_prompt}\nUser: {question}\nAssistant:"
    
    # Create stopping criteria
    stop_criteria = StopOnPatternCriteria(tokenizer)
    stopping_criteria_list = [stop_criteria]
    
    # Add force stop criteria if stop_flag is provided
    if stop_flag is not None:
        force_stop_criteria = ForceStopCriteria(stop_flag)
        stopping_criteria_list.append(force_stop_criteria)
    
    stopping_criteria = StoppingCriteriaList(stopping_criteria_list)
    
    # Create streamers and queues for each response
    streamers = []
    queues = []
    threads = []
    accumulated_texts = [""] * num_responses
    
    # Create a queue to collect chunks from all streams
    chunk_queue = queue.Queue()
    active_responses = set(range(num_responses))
    
    def stream_reader(streamer, response_id, chunk_queue, accumulated_texts, active_responses):
        """Read from a streamer and put chunks in the queue."""
        completion_sent = False
        try:
            stop_patterns = ["\nUser:", "\nSystem:", "User:", "System:"]
            for new_text in streamer:
                # Check stop flag - but don't stop immediately, let current generation finish
                # The stop flag will be checked by the main loop to mark as stopped
                # but we continue processing chunks to allow natural completion
                
                # Only process if this response is still active
                if response_id not in active_responses:
                    return
                
                accumulated_texts[response_id] += new_text
                
                # Check for stop patterns
                text_to_check = accumulated_texts[response_id].split("Assistant:")[-1] if "Assistant:" in accumulated_texts[response_id] else accumulated_texts[response_id]
                
                found_pattern = False
                earliest_idx = len(text_to_check)
                for pattern in stop_patterns:
                    idx = text_to_check.find(pattern)
                    if idx != -1 and idx < earliest_idx:
                        earliest_idx = idx
                        found_pattern = True
                
                if found_pattern:
                    # Truncate at stop pattern
                    if "Assistant:" in accumulated_texts[response_id]:
                        assistant_start = accumulated_texts[response_id].find("Assistant:") + len("Assistant:")
                        actual_stop_pos = assistant_start + earliest_idx
                        truncated = accumulated_texts[response_id][:actual_stop_pos].strip()
                    else:
                        truncated = accumulated_texts[response_id][:earliest_idx].strip()
                    
                    # Yield remaining part if any
                    already_yielded_len = len(accumulated_texts[response_id]) - len(new_text)
                    if len(truncated) > already_yielded_len:
                        chunk_queue.put((response_id, truncated[already_yielded_len:], False))
                    if not completion_sent:
                        chunk_queue.put((response_id, None, True))  # Done signal
                        completion_sent = True
                    return
                
                # Put chunk in queue - this response is still active
                chunk_queue.put((response_id, new_text, False))
            
            # Streamer finished naturally - signal completion only if not already sent
            if not completion_sent:
                chunk_queue.put((response_id, None, True))
                completion_sent = True
        except Exception as e:
            # Put error in queue and signal completion only if not already sent
            if not completion_sent:
                chunk_queue.put((response_id, f"Error: {str(e)}", True))
                completion_sent = True
    
    # Start all generation threads in parallel
    for resp_idx in range(num_responses):
        inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        streamers.append(streamer)
        
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "pad_token_id": tokenizer.pad_token_id,
            "streamer": streamer,
            "stopping_criteria": stopping_criteria,
        }
        
        # Start reader thread for this streamer
        reader_thread = Thread(target=stream_reader, args=(streamer, resp_idx, chunk_queue, accumulated_texts, active_responses), daemon=True)
        reader_thread.start()
        threads.append(reader_thread)
        
        # Start generation thread
        def generate_with_no_grad(kwargs=generation_kwargs):
            try:
                with torch.no_grad():
                    model.generate(**kwargs)
            except Exception:
                pass
        
        gen_thread = Thread(target=generate_with_no_grad, daemon=True)
        gen_thread.start()
        threads.append(gen_thread)
    
    # Yield chunks as they arrive from any stream
    try:
        while active_responses:
            try:
                # Use a longer timeout to ensure we don't miss chunks
                response_id, chunk, is_done = chunk_queue.get(timeout=1.0)
                
                if is_done:
                    if chunk:
                        # Error message
                        yield (response_id, chunk)
                    else:
                        # Completion signal
                        yield (response_id, None)
                    # Remove from active responses only after yielding completion
                    active_responses.discard(response_id)
                else:
                    # Regular chunk - yield it (response is still active at this point)
                    yield (response_id, chunk)
                    
            except queue.Empty:
                # Check if stop was requested
                # Don't break immediately - let active responses continue generating
                # The stop flag will be checked by individual reader threads
                if stop_flag is not None:
                    if isinstance(stop_flag, list) and stop_flag[0]:
                        # Mark all remaining as stopped, but don't break - let them finish naturally
                        # The reader threads will check the flag and exit when they see it
                        pass
                    elif isinstance(stop_flag, dict) and stop_flag.get('stop', False):
                        # Same as above - let responses finish naturally
                        pass
                # Continue waiting for chunks from remaining active responses
                continue
    
    finally:
        # Wait for all threads to finish
        for thread in threads:
            thread.join(timeout=2.0)
        
        # Clean up GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ==================== Evaluation Metrics ====================

def compute_jargon_score(text: str) -> float:
    """Compute jargon score based on academic/specialized vocabulary."""
    # Simple heuristic: count academic-sounding words
    academic_patterns = [
        r'\b(?:theorem|hypothesis|paradigm|methodology|framework|mechanism|phenomenon|'
        r'quantitative|qualitative|empirical|theoretical|analytical|systematic)\b',
        r'\b(?:-[a-z]+ly)\b',  # Adverbs like "systematically"
    ]
    matches = sum(len(re.findall(pattern, text.lower())) for pattern in academic_patterns)
    return matches / max(len(text.split()), 1)


def compute_verbosity_score(text: str) -> float:
    """Compute verbosity score (normalized token count)."""
    tokens = len(text.split())
    # Normalize to 0-1 scale (assuming typical range 50-500 tokens)
    return min(tokens / 500.0, 1.0)


def compute_logic_score(text: str) -> float:
    """Simple heuristic for logic depth (can be replaced with LLM judge)."""
    # Look for logical connectors and reasoning indicators
    logic_indicators = [
        r'\b(?:because|therefore|thus|hence|consequently|implies|follows|'
        r'if.*then|given that|assuming|suppose|consider|analyze|examine)\b',
        r'\b(?:first|second|third|finally|moreover|furthermore|additionally)\b',
    ]
    matches = sum(len(re.findall(pattern, text.lower())) for pattern in logic_indicators)
    return min(matches / 10.0, 1.0)  # Normalize


def compute_correctness_score(text: str, reference_answer: Optional[str] = None) -> float:
    """
    Simple heuristic for correctness (can be replaced with LLM judge or fact-checking).
    Looks for indicators of uncertainty, hedging, or incorrect statements.
    """
    # Indicators of uncertainty/incorrectness
    uncertainty_indicators = [
        r'\b(?:might|maybe|perhaps|possibly|probably|unlikely|doubtful|uncertain)\b',
        r'\b(?:i think|i believe|i guess|i suppose|not sure|not certain)\b',
        r'\b(?:wrong|incorrect|false|mistake|error|misconception)\b',
    ]
    uncertainty_matches = sum(len(re.findall(pattern, text.lower())) for pattern in uncertainty_indicators)
    
    # Indicators of confidence/correctness
    confidence_indicators = [
        r'\b(?:certainly|definitely|absolutely|precisely|exactly|correctly|accurately)\b',
        r'\b(?:fact|truth|accurate|correct|verified|confirmed)\b',
    ]
    confidence_matches = sum(len(re.findall(pattern, text.lower())) for pattern in confidence_indicators)
    
    # Normalize: higher score = more correct
    uncertainty_score = min(uncertainty_matches / 5.0, 1.0)
    confidence_score = min(confidence_matches / 5.0, 1.0)
    
    correctness = confidence_score - uncertainty_score * 0.5
    return max(0.0, min(1.0, (correctness + 1.0) / 2.0))  # Normalize to [0, 1]


def evaluate_output(text: str, reference_answer: Optional[str] = None) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    return {
        'logic_score': compute_logic_score(text),
        'jargon_score': compute_jargon_score(text),
        'verbosity_score': compute_verbosity_score(text),
        'correctness_score': compute_correctness_score(text, reference_answer),
    }


# ==================== Demo Function ====================

def run_enhanced_demo(
    tokenizer,
    model,
    vectors_dict: Dict[int, Dict[str, np.ndarray]],
    alpha_cap: float,
    alpha_style: float,
    alpha_density: float,
    alpha_correctness: float,
    question: str,
):
    """Run comprehensive comparison with all variants."""
    
    print("=" * 80)
    print("=== Enhanced Multi-Dimensional Steering Demo ===")
    print("=" * 80)
    
    # 1. Baseline Low
    print("\n" + "=" * 80)
    print("=== 1. Baseline Low Persona ===")
    print("=" * 80)
    baseline_low = generate_with_persona(tokenizer, model, question, persona="low")
    print(baseline_low)
    metrics_low = evaluate_output(baseline_low)
    print(f"\nMetrics: Logic={metrics_low['logic_score']:.3f}, "
          f"Jargon={metrics_low['jargon_score']:.3f}, "
          f"Verbosity={metrics_low['verbosity_score']:.3f}, "
          f"Correctness={metrics_low['correctness_score']:.3f}")
    
    # 2. Low + Steering
    print("\n" + "=" * 80)
    print(f"=== 2. Low Persona + Steering ===")
    print(f"alpha_cap={alpha_cap}, alpha_style={alpha_style}, alpha_density={alpha_density}, alpha_correctness={alpha_correctness}")
    print("=" * 80)
    with MultiVectorActivationSteerer(model, vectors_dict, alpha_cap, alpha_style, alpha_density, alpha_correctness):
        steered_low = generate_with_persona(tokenizer, model, question, persona="low")
    print(steered_low)
    metrics_steered_low = evaluate_output(steered_low)
    print(f"\nMetrics: Logic={metrics_steered_low['logic_score']:.3f}, "
          f"Jargon={metrics_steered_low['jargon_score']:.3f}, "
          f"Verbosity={metrics_steered_low['verbosity_score']:.3f}, "
          f"Correctness={metrics_steered_low['correctness_score']:.3f}")
    
    # 3. Baseline High
    print("\n" + "=" * 80)
    print("=== 3. Baseline High Persona ===")
    print("=" * 80)
    baseline_high = generate_with_persona(tokenizer, model, question, persona="high")
    print(baseline_high)
    metrics_high = evaluate_output(baseline_high)
    print(f"\nMetrics: Logic={metrics_high['logic_score']:.3f}, "
          f"Jargon={metrics_high['jargon_score']:.3f}, "
          f"Verbosity={metrics_high['verbosity_score']:.3f}, "
          f"Correctness={metrics_high['correctness_score']:.3f}")

    # 4. High - Steering
    print("\n" + "=" * 80)
    print(f"=== 4. High Persona - Steering ===")
    print(f"alpha_cap={-alpha_cap}, alpha_style={-alpha_style}, alpha_density={-alpha_density}, alpha_correctness={-alpha_correctness}")
    print("=" * 80)
    with MultiVectorActivationSteerer(model, vectors_dict, -alpha_cap, -alpha_style, -alpha_density, -alpha_correctness):
        steered_high = generate_with_persona(tokenizer, model, question, persona="high")
    print(steered_high)
    metrics_steered_high = evaluate_output(steered_high)
    print(f"\nMetrics: Logic={metrics_steered_high['logic_score']:.3f}, "
          f"Jargon={metrics_steered_high['jargon_score']:.3f}, "
          f"Verbosity={metrics_steered_high['verbosity_score']:.3f}, "
          f"Correctness={metrics_steered_high['correctness_score']:.3f}")
    
    # 4. Summary
    print("\n" + "=" * 80)
    print("=== Summary ===")
    print("=" * 80)
    print("Target: High Logic, Low Jargon, Low Verbosity, High Correctness (深入浅出)")
    print(f"\nBaseline Low:     Logic={metrics_low['logic_score']:.3f}, "
          f"Jargon={metrics_low['jargon_score']:.3f}, Verbosity={metrics_low['verbosity_score']:.3f}, "
          f"Correctness={metrics_low['correctness_score']:.3f}")
    print(f"Steered Low:      Logic={metrics_steered_low['logic_score']:.3f}, "
          f"Jargon={metrics_steered_low['jargon_score']:.3f}, Verbosity={metrics_steered_low['verbosity_score']:.3f}, "
          f"Correctness={metrics_steered_low['correctness_score']:.3f}")
    print(f"Baseline High:    Logic={metrics_high['logic_score']:.3f}, "
          f"Jargon={metrics_high['jargon_score']:.3f}, Verbosity={metrics_high['verbosity_score']:.3f}, "
          f"Correctness={metrics_high['correctness_score']:.3f}")
    
    logic_improvement = metrics_steered_low['logic_score'] - metrics_low['logic_score']
    jargon_change = metrics_steered_low['jargon_score'] - metrics_low['jargon_score']
    verbosity_change = metrics_steered_low['verbosity_score'] - metrics_low['verbosity_score']
    correctness_change = metrics_steered_low['correctness_score'] - metrics_low['correctness_score']
    
    print(f"\nImprovement: Logic +{logic_improvement:+.3f}, "
          f"Jargon {jargon_change:+.3f}, Verbosity {verbosity_change:+.3f}, "
          f"Correctness {correctness_change:+.3f}")
    print("=" * 80)


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Dimensional Activation Steering"
    )
    parser.add_argument(
        "--capability-file",
        type=str,
        default="data/expert_data/expert_data.json",
        help="Path to capability dataset JSON.",
    )
    parser.add_argument(
        "--layer-indices",
        type=int,
        nargs="+",
        default=[DEFAULT_LAYER_INDEX],
        help="Layer indices for vector extraction.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="Max samples per vector type (<=0 means no limit).",
    )
    parser.add_argument(
        "--alpha-cap",
        type=float,
        default=3.0,
        help="Steering strength for pure capability.",
    )
    parser.add_argument(
        "--alpha-style",
        type=float,
        default=-1.0,
        help="Steering strength for style (negative = suppress jargon).",
    )
    parser.add_argument(
        "--alpha-density",
        type=float,
        default=-1.0,
        help="Steering strength for density (negative = suppress verbosity).",
    )
    parser.add_argument(
        "--alpha-correctness",
        type=float,
        default=0.0,
        help="Steering strength for correctness (positive = correct, negative = incorrect).",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Explain quantum entanglement.",
        help="Demo question.",
    )
    parser.add_argument(
        "--vectors-dir",
        type=str,
        default="data/vectors_enhanced",
        help="Directory to save/load vectors.",
    )
    parser.add_argument(
        "--recompute-vectors",
        action="store_true",
        help="Force recomputation of all vectors.",
    )
    parser.add_argument(
        "--generate-prompt-variants",
        action="store_true",
        help="Generate prompt variants using Qwen (will be saved to data/prompt_variants.json).",
    )
    parser.add_argument(
        "--prompt-variants-file",
        type=str,
        default="data/prompt_variants.json",
        help="Path to prompt variants JSON file.",
    )
    parser.add_argument(
        "--num-variants-per-prompt",
        type=int,
        default=5,
        help="Number of variants to generate per prompt type.",
    )
    parser.add_argument(
        "--use-prompt-variants",
        action="store_true",
        default=True,
        help="Use prompt variants if available (default: True).",
    )
    parser.add_argument(
        "--cycle-variants",
        action="store_true",
        help="Cycle through variants deterministically instead of random selection.",
    )
    parser.add_argument(
        "--no-orthogonalize",
        action="store_true",
        help="Skip Gram-Schmidt orthogonalization and use raw vectors directly.",
    )

    args = parser.parse_args()

    # Load model first (needed for generating variants)
    print("Loading Qwen model...")
    tokenizer, model = load_qwen()

    # Generate prompt variants if requested
    if args.generate_prompt_variants:
        generate_all_prompt_variants(
            tokenizer,
            model,
            args.prompt_variants_file,
            args.num_variants_per_prompt,
            force_regenerate=True,
        )

    # Load data
    print("\nLoading capability data...")
    data = load_capability_data(args.capability_file)
    print(f"Loaded {len(data)} Q&A pairs")

    # Format pairs
    print("\nFormatting multi-axis pairs...")
    pairs = format_multi_axis_pairs(
        data,
        use_variants=args.use_prompt_variants,
        cycle_variants=args.cycle_variants,
    )

    layer_indices = sorted(set(args.layer_indices))
    print(f"Target layers: {layer_indices}")

    vectors_dir = Path(args.vectors_dir)
    vectors_dir.mkdir(parents=True, exist_ok=True)

    # Compute or load vectors
    need_recompute = args.recompute_vectors
    
    v_cap_path = vectors_dir / f"v_cap_layers_{'_'.join(map(str, layer_indices))}.npz"
    v_style_path = vectors_dir / f"v_style_layers_{'_'.join(map(str, layer_indices))}.npz"
    v_density_path = vectors_dir / f"v_density_layers_{'_'.join(map(str, layer_indices))}.npz"
    v_correctness_path = vectors_dir / f"v_correctness_layers_{'_'.join(map(str, layer_indices))}.npz"

    if not need_recompute and v_cap_path.exists() and v_style_path.exists() and v_density_path.exists():
        print("\nLoading existing vectors...")
        def load_vector_dict(path):
            data = np.load(path)
            return {int(k.split('_')[1]): data[k] for k in data.files if k.startswith('layer_')}
        v_cap_dict = load_vector_dict(v_cap_path)
        v_style_dict = load_vector_dict(v_style_path)
        v_density_dict = load_vector_dict(v_density_path)
        v_correctness_dict = load_vector_dict(v_correctness_path) if v_correctness_path.exists() else None
        
        # Check if all requested layers are present
        loaded_layers = set(v_cap_dict.keys()) & set(v_style_dict.keys()) & set(v_density_dict.keys())
        if v_correctness_dict:
            loaded_layers = loaded_layers & set(v_correctness_dict.keys())
        missing_layers = set(layer_indices) - loaded_layers
        if missing_layers:
            print(f"Warning: Missing layers {missing_layers} in saved files. Recomputing...")
            need_recompute = True
        else:
            # Filter to only requested layers
            v_cap_dict = {k: v_cap_dict[k] for k in layer_indices}
            v_style_dict = {k: v_style_dict[k] for k in layer_indices}
            v_density_dict = {k: v_density_dict[k] for k in layer_indices}
            if v_correctness_dict:
                v_correctness_dict = {k: v_correctness_dict[k] for k in layer_indices}
            print(f"Loaded vectors for layers: {sorted(v_cap_dict.keys())}")
    else:
        print("\nComputing vectors...")
        v_cap_dict = compute_vector(
            pairs["capability"], tokenizer, model, layer_indices, args.max_samples, "v_cap"
        )
        v_style_dict = compute_vector(
            pairs["style"], tokenizer, model, layer_indices, args.max_samples, "v_style"
        )
        v_density_dict = compute_vector(
            pairs["density"], tokenizer, model, layer_indices, args.max_samples, "v_density"
        )
        v_correctness_dict = compute_vector(
            pairs["correctness"], tokenizer, model, layer_indices, args.max_samples, "v_correctness"
        )
        
        # Save vectors
        np.savez(v_cap_path, **{f"layer_{k}": v for k, v in v_cap_dict.items()})
        np.savez(v_style_path, **{f"layer_{k}": v for k, v in v_style_dict.items()})
        np.savez(v_density_path, **{f"layer_{k}": v for k, v in v_density_dict.items()})
        np.savez(v_correctness_path, **{f"layer_{k}": v for k, v in v_correctness_dict.items()})
        print(f"\nSaved vectors to {vectors_dir}")

    # Prepare vectors (with or without orthogonalization)
    if args.no_orthogonalize:
        print("\n" + "=" * 80)
        print("Using raw vectors (no orthogonalization)...")
        print("=" * 80)
        vectors_dict = {}
        for layer_idx in layer_indices:
            vectors_dict[layer_idx] = {
                'v_cap': v_cap_dict[layer_idx],
                'v_style': v_style_dict[layer_idx],
                'v_density': v_density_dict[layer_idx],
                'v_cap_pure': v_cap_dict[layer_idx],  # Use raw v_cap as v_cap_pure when not orthogonalizing
            }
            if v_correctness_dict and layer_idx in v_correctness_dict:
                vectors_dict[layer_idx]['v_correctness'] = v_correctness_dict[layer_idx]
            
            print(f"\n[Layer {layer_idx} Raw Vectors]")
            print(f"  ||v_cap|| = {np.linalg.norm(v_cap_dict[layer_idx]):.4f}")
            print(f"  ||v_style|| = {np.linalg.norm(v_style_dict[layer_idx]):.4f}")
            print(f"  ||v_density|| = {np.linalg.norm(v_density_dict[layer_idx]):.4f}")
            if v_correctness_dict and layer_idx in v_correctness_dict:
                print(f"  ||v_correctness|| = {np.linalg.norm(v_correctness_dict[layer_idx]):.4f}")
    else:
        print("\n" + "=" * 80)
        print("Applying Gram-Schmidt orthogonalization...")
        print("=" * 80)
        vectors_dict = compute_orthogonalized_vectors(
            v_cap_dict, v_style_dict, v_density_dict, v_correctness_dict
        )

    # Run demo
    print("\n" + "=" * 80)
    print("Running enhanced demo...")
    print("=" * 80)
    run_enhanced_demo(
        tokenizer=tokenizer,
        model=model,
        vectors_dict=vectors_dict,
        alpha_cap=args.alpha_cap,
        alpha_style=args.alpha_style,
        alpha_density=args.alpha_density,
        alpha_correctness=args.alpha_correctness,
        question=args.question,
    )


if __name__ == "__main__":
    main()

