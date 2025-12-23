import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_generator import format_data_for_experiment


MODEL_ID = "./Qwen2.5-3B-Instruct"
DEFAULT_LAYER_INDEX = 20  # a mid-layer; can be overridden via CLI


def load_qwen(model_id: str = MODEL_ID):
    """
    Load Qwen 7B Instruct (or Qwen2.5-7B-Instruct) with hidden-states enabled.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def _extract_answer_hidden_states(
    tokenizer,
    model,
    text: str,
    answer_text: str,
    layer_index: int,
) -> np.ndarray:
    """
    Run a single teacher-forced pass and return the hidden states corresponding
    to the answer segment only, at the chosen layer.
    """
    with torch.no_grad():
        encoded_full = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded_ans = tokenizer(
            answer_text,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = encoded_full["input_ids"].to(model.device)
        attention_mask = encoded_full["attention_mask"].to(model.device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        hidden_states = out.hidden_states[layer_index]  # [1, seq_len, d_model]

        # Heuristic: treat the last len(answer_ids) tokens as belonging to the answer.
        ans_len = encoded_ans["input_ids"].shape[-1]
        answer_slice = hidden_states[:, -ans_len:, :]  # [1, T_ans, d_model]
        return answer_slice.squeeze(0).float().cpu().numpy()


def compute_v_cap(
    capability_pairs: List[Dict],
    tokenizer,
    model,
    layer_indices: List[int],
    max_samples: int = 128,
) -> Dict[int, np.ndarray]:
    """
    Compute the capability vector v_cap using teacher forcing for multiple layers independently.

    For each layer and each pair:
        v_i = mean_token( h_high(answer_tokens) - h_low(answer_tokens) )
    Then:
        v_cap[layer] = mean_i v_i

    Returns a dict mapping layer_index -> v_cap vector.
    """
    layer_diffs: Dict[int, List[np.ndarray]] = {layer_idx: [] for layer_idx in layer_indices}
    used = 0

    limit = max_samples if max_samples is not None and max_samples > 0 else None

    for item in tqdm(capability_pairs, desc=f"Computing v_cap for {len(layer_indices)} layers"):
        if limit is not None and used >= limit:
            break

        text_low = item["text_low"]
        text_high = item["text_high"]
        target_answer = item["target_answer"]

        # Extract hidden states for all layers in one forward pass
        try:
            with torch.no_grad():
                encoded_full = tokenizer(
                    text_low,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                encoded_ans = tokenizer(
                    target_answer,
                    return_tensors="pt",
                    add_special_tokens=False,
                )

                input_ids = encoded_full["input_ids"].to(model.device)
                attention_mask = encoded_full["attention_mask"].to(model.device)

                out_low = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

                # Process high persona
                encoded_full_high = tokenizer(
                    text_high,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
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

                    delta = h_high - h_low  # [T, D] : "becoming smarter"
                    v_i = delta.mean(axis=0)  # [D]
                    layer_diffs[layer_idx].append(v_i)

        except Exception as e:
            print(f"[compute_v_cap] Skipping one sample due to error: {e}")
            continue

        used += 1

    # Compute v_cap for each layer
    v_cap_dict: Dict[int, np.ndarray] = {}
    for layer_idx in layer_indices:
        diffs = layer_diffs[layer_idx]
        if not diffs:
            raise RuntimeError(f"No valid capability pairs were processed for layer {layer_idx}. v_cap is empty.")

        diffs_arr = np.stack(diffs, axis=0)  # [N, D]
        v_cap = diffs_arr.mean(axis=0)  # [D]
        v_cap_dict[layer_idx] = v_cap

        # ---- Diagnostics / analysis per layer ----
        per_sample_norms = np.linalg.norm(diffs_arr, axis=1)
        v_cap_norm = np.linalg.norm(v_cap)
        print(
            f"[v_cap Layer {layer_idx}] Used {len(diffs)} capability pairs "
            f"(requested max_samples={max_samples}, total available={len(capability_pairs)})."
        )
        print(
            f"[v_cap Layer {layer_idx}] Per-sample ||v_i||: "
            f"mean={per_sample_norms.mean():.4f}, "
            f"std={per_sample_norms.std():.4f}, "
            f"min={per_sample_norms.min():.4f}, "
            f"max={per_sample_norms.max():.4f}"
        )
        print(f"[v_cap Layer {layer_idx}] Aggregate ||v_cap|| = {v_cap_norm:.4f}")

    return v_cap_dict


def save_v_cap(v_cap_dict: Dict[int, np.ndarray], out_path: Path):
    """
    Save v_cap vectors for multiple layers.
    Uses .npz format to store multiple arrays.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as .npz with keys like "layer_20", "layer_15", etc.
    save_dict = {f"layer_{layer_idx}": v_cap for layer_idx, v_cap in v_cap_dict.items()}
    np.savez(out_path, **save_dict)
    print(f"Saved v_cap for {len(v_cap_dict)} layers to {out_path}")
    for layer_idx, v_cap in v_cap_dict.items():
        print(f"  Layer {layer_idx}: shape {v_cap.shape}, ||v_cap|| = {np.linalg.norm(v_cap):.4f}")


def load_v_cap(out_path: Path) -> Dict[int, np.ndarray]:
    """
    Load v_cap vectors. Supports both old format (single .npy) and new format (.npz with multiple layers).
    """
    if not out_path.exists():
        raise FileNotFoundError(f"v_cap file not found: {out_path}")
    
    if out_path.suffix == ".npz":
        # New multi-layer format
        data = np.load(out_path)
        v_cap_dict = {}
        for key in data.files:
            if key.startswith("layer_"):
                layer_idx = int(key.split("_")[1])
                v_cap_dict[layer_idx] = data[key]
        return v_cap_dict
    else:
        # Old single-layer format (backward compatibility)
        v_cap = np.load(out_path)
        # Try to infer layer index from filename or use default
        # For backward compatibility, we'll return a dict with a single entry
        # The caller should handle this appropriately
        return {DEFAULT_LAYER_INDEX: v_cap}


class ActivationSteerer:
    """
    Simple activation steering via a forward hook on a single Transformer layer.
    """

    def __init__(self, model, v_cap: np.ndarray, layer_index: int, alpha: float):
        self.model = model
        # Keep on CPU; move to the correct device inside the hook for each layer.
        self.v_cap = torch.tensor(v_cap, dtype=torch.float32, device="cpu")
        self.layer_index = layer_index
        self.alpha = alpha
        self.handle = None

    def __enter__(self):
        def hook(module, inputs, output):
            # output: [batch, seq, dim]
            if not isinstance(output, torch.Tensor):
                return output
            v_cap = self.v_cap.to(output.device, dtype=output.dtype)
            return output + self.alpha * v_cap.view(1, 1, -1)

        # Most HF decoder-only models expose transformer blocks here:
        block = self.model.model.layers[self.layer_index]
        self.handle = block.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class MultiLayerActivationSteerer:
    """
    Activation steering via forward hooks on multiple Transformer layers.
    Each layer gets its own independent v_cap vector and steering strength.
    """

    def __init__(self, model, v_cap_dict: Dict[int, np.ndarray], alpha: float):
        """
        Args:
            model: The transformer model
            v_cap_dict: Dict mapping layer_index -> v_cap vector for that layer
            alpha: Steering strength (applied uniformly to all layers)
        """
        self.model = model
        self.v_cap_dict = {
            layer_idx: torch.tensor(v_cap, dtype=torch.float32, device="cpu")
            for layer_idx, v_cap in v_cap_dict.items()
        }
        self.alpha = alpha
        self.handles = []

    def __enter__(self):
        for layer_idx, v_cap_tensor in self.v_cap_dict.items():
            # Create a closure that properly captures v_cap_tensor for this iteration
            # Using a lambda with default argument to capture by value
            def make_hook(layer_v_cap):
                def hook(module, inputs, output):
                    # output: [batch, seq, dim]
                    if not isinstance(output, torch.Tensor):
                        return output
                    v_cap_device = layer_v_cap.to(output.device, dtype=output.dtype)
                    return output + self.alpha * v_cap_device.view(1, 1, -1)
                return hook

            # Most HF decoder-only models expose transformer blocks here:
            block = self.model.model.layers[layer_idx]
            handle = block.register_forward_hook(make_hook(v_cap_tensor))
            self.handles.append((layer_idx, handle))
        
        print(f"[MultiLayerSteerer] Registered hooks on {len(self.handles)} layers: {sorted(self.v_cap_dict.keys())}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for layer_idx, handle in self.handles:
            handle.remove()
        self.handles.clear()


def generate_with_persona(
    tokenizer,
    model,
    question: str,
    persona: str = "low",
    max_new_tokens: int = 1024,
) -> str:
    """
    Simple generation helper for qualitative inspection.
    """
    if persona == "low":
        sys_prompt = (
            "System: You are a curious 5-year-old child. "
            "Please explain things using very simple words and basic intuition that a kindergartener would understand."
        )
    elif persona == "high":
        sys_prompt = (
            "System: You are a world-renowned expert and Fields Medalist in this field. "
            "Please provide a rigorous, comprehensive, and theoretically profound explanation."
        )
    else:
        sys_prompt = ""

    full_prompt = f"{sys_prompt}\nUser: {question}\nAssistant:"

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
        )

    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    
    # Extract only the Assistant's response
    split_tok = "Assistant:"
    if split_tok in generated:
        # Take everything after the last "Assistant:" occurrence
        parts = generated.split(split_tok)
        assistant_response = parts[-1].strip()
    else:
        assistant_response = generated.strip()
    
    # Remove any hallucinated User: or System: prompts
    # Check for various patterns and truncate at the first occurrence
    stop_patterns = [
        "\nUser:",      # New line followed by User:
        "\nSystem:",    # New line followed by System:
        "User:",        # User: at start or middle
        "System:",      # System: at start or middle
    ]
    
    # Find the earliest occurrence of any stop pattern
    earliest_idx = len(assistant_response)
    for pattern in stop_patterns:
        idx = assistant_response.find(pattern)
        if idx != -1 and idx < earliest_idx:
            earliest_idx = idx
    
    # Truncate if we found a stop pattern
    if earliest_idx < len(assistant_response):
        assistant_response = assistant_response[:earliest_idx].strip()
    
    # Additional cleanup: remove trailing incomplete sentences if they look like prompts
    # Remove lines that start with "User:" or "System:" even if not at the beginning
    lines = assistant_response.split('\n')
    cleaned_lines = []
    for line in lines:
        line_stripped = line.strip()
        # Skip lines that look like prompts
        if line_stripped.startswith(('User:', 'System:')):
            break
        cleaned_lines.append(line)
    
    assistant_response = '\n'.join(cleaned_lines).strip()
    
    return assistant_response


def run_demo(
    tokenizer,
    model,
    v_cap_dict: Dict[int, np.ndarray],
    alpha: float,
    question: str,
):
    """
    Compare all variants:
    1. Baseline low persona (no steering)
    2. Low persona + v_cap (steering to increase intelligence)
    3. Baseline high persona (no steering)
    4. High persona - v_cap (steering to decrease intelligence)
    
    Uses multi-layer steering if multiple layers are provided.
    """
    layers_str = f"{len(v_cap_dict)} layers: {sorted(v_cap_dict.keys())}"
    
    print("=" * 80)
    print("=== 1. Baseline Low Persona (no steering) ===")
    print("=" * 80)
    baseline_low = generate_with_persona(tokenizer, model, question, persona="low")
    print(baseline_low)
    
    print("\n" + "=" * 80)
    print(f"=== 2. Low Persona + v_cap (steering to increase intelligence) ===")
    print(f"Layers: {layers_str}, alpha={alpha}")
    print("=" * 80)
    with MultiLayerActivationSteerer(model, v_cap_dict, alpha):
        steered_low = generate_with_persona(tokenizer, model, question, persona="low")
    print(steered_low)
    
    print("\n" + "=" * 80)
    print("=== 3. Baseline High Persona (no steering) ===")
    print("=" * 80)
    baseline_high = generate_with_persona(tokenizer, model, question, persona="high")
    print(baseline_high)
    
    print("\n" + "=" * 80)
    print(f"=== 4. High Persona - v_cap (steering to decrease intelligence) ===")
    print(f"Layers: {layers_str}, alpha={-alpha} (negative)")
    print("=" * 80)
    with MultiLayerActivationSteerer(model, v_cap_dict, -alpha):
        steered_high = generate_with_persona(tokenizer, model, question, persona="high")
    print(steered_high)
    
    print("\n" + "=" * 80)
    print("=== Summary ===")
    print("=" * 80)
    print("1. Baseline Low:  Simple language, low depth")
    print("2. Low + v_cap:   Simple language, HIGH depth (target: '深入浅出')")
    print("3. Baseline High: Complex language, high depth")
    print("4. High - v_cap:  Complex language, LOW depth (reverse steering)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Operation Sandbagging with Qwen 7B Instruct (ignoring v_style)."
    )
    parser.add_argument(
        "--capability-file",
        type=str,
        default="data/expert_data/expert_data.json",
        help="Path to the capability dataset JSON.",
    )
    parser.add_argument(
        "--style-file",
        type=str,
        default=None,
        help="(Optional) Path to the style dataset JSON. Ignored for v_cap.",
    )
    parser.add_argument(
        "--layer-indices",
        type=int,
        nargs="+",
        default=[DEFAULT_LAYER_INDEX],
        help="Layer indices at which to measure and inject v_cap. Can specify multiple layers, e.g., --layer-indices 15 20 25",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=256,
        help="Max number of capability pairs used to compute v_cap (<=0 means no limit).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=3.0,
        help="Steering strength for intervention.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Explain quantum entanglement.",
        help="Demo question for qualitative comparison.",
    )
    parser.add_argument(
        "--v-cap-path",
        type=str,
        default="data/v_cap_qwen2_5_7b_instruct.npz",
        help="Where to save / load the capability vectors (.npz format for multi-layer).",
    )
    parser.add_argument(
        "--recompute-v-cap",
        action="store_true",
        help="Force recomputation of v_cap even if the file already exists.",
    )

    args = parser.parse_args()

    data_root = Path(".")
    capability_file = data_root / args.capability_file
    style_file = data_root / args.style_file if args.style_file else None
    v_cap_path = data_root / args.v_cap_path

    print("Loading formatted capability pairs (style data is optional / ignored)...")
    cap_pairs, _ = format_data_for_experiment(
        str(capability_file),
        str(style_file) if style_file is not None else None,
    )
    if not cap_pairs:
        raise RuntimeError("No capability pairs loaded. Check capability_file path.")

    print("Loading Qwen model...")
    tokenizer, model = load_qwen()

    layer_indices = sorted(set(args.layer_indices))  # Remove duplicates and sort
    print(f"Target layers: {layer_indices}")

    # Check if we need to recompute
    need_recompute = args.recompute_v_cap or not v_cap_path.exists()
    
    if not need_recompute:
        try:
            print(f"Loading existing v_cap from {v_cap_path}")
            loaded_v_cap_dict = load_v_cap(v_cap_path)
            # Check if all requested layers are present
            missing_layers = [idx for idx in layer_indices if idx not in loaded_v_cap_dict]
            if missing_layers:
                print(f"Warning: Missing layers {missing_layers} in saved file. Recomputing...")
                need_recompute = True
            else:
                # Use only the requested layers
                v_cap_dict = {idx: loaded_v_cap_dict[idx] for idx in layer_indices}
        except Exception as e:
            print(f"Error loading v_cap: {e}. Recomputing...")
            need_recompute = True

    if need_recompute:
        print("Computing v_cap via teacher forcing (ignoring v_style)...")
        v_cap_dict = compute_v_cap(
            capability_pairs=cap_pairs,
            tokenizer=tokenizer,
            model=model,
            layer_indices=layer_indices,
            max_samples=args.max_samples,
        )
        save_v_cap(v_cap_dict, v_cap_path)
    else:
        v_cap_dict = {idx: loaded_v_cap_dict[idx] for idx in layer_indices}

    print("\nRunning qualitative demo comparison...")
    run_demo(
        tokenizer=tokenizer,
        model=model,
        v_cap_dict=v_cap_dict,
        alpha=args.alpha,
        question=args.question,
    )


if __name__ == "__main__":
    main()


