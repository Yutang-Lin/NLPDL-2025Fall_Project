"""
Generate diverse prompt variants for each dimension using Qwen model.
This increases the robustness of vector extraction by using multiple phrasings.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_ID = "./Qwen2.5-3B-Instruct"


# Base prompt templates for each dimension
BASE_PROMPTS = {
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
}


def load_qwen(model_id: str = MODEL_ID):
    """Load Qwen model."""
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


def generate_prompt_variants(
    tokenizer,
    model,
    base_prompt: str,
    num_variants: int = 5,
    temperature: float = 0.8,
) -> List[str]:
    """
    Generate multiple paraphrased variants of a base prompt using Qwen.
    """
    generation_prompt = f"""Task: Generate {num_variants} different ways to express the following instruction, while keeping the core meaning exactly the same. Each variant should use different wording but convey the same intent.

Original instruction:
{base_prompt}

Generate {num_variants} variants, one per line. Each variant should:
1. Maintain the exact same meaning and intent
2. Use different wording and phrasing
3. Be clear and direct
4. Start with "System:"

Output format (one variant per line):
System: [variant 1]
System: [variant 2]
...
System: [variant {num_variants}]
"""

    inputs = tokenizer(generation_prompt, return_tensors="pt", add_special_tokens=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=512,
            num_return_sequences=1,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract variants from the generated text
    variants = []
    lines = generated_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('System:'):
            # Clean up the variant
            variant = line
            # Remove any trailing explanations or notes
            if ':' in variant and variant.count(':') > 1:
                # Keep only the first part before second colon if it looks like explanation
                parts = variant.split(':')
                if len(parts) > 2:
                    variant = ':'.join(parts[:2]) + ':'
            variants.append(variant)
    
    # If we didn't get enough variants, try generating more
    if len(variants) < num_variants:
        # Try a different approach: generate one at a time
        variants = []
        for i in range(num_variants):
            single_prompt = f"""Task: Rephrase the following instruction in a different way while keeping the exact same meaning.

Original:
{base_prompt}

Generate ONE rephrased version that:
- Uses different wording
- Maintains the same intent
- Starts with "System:"

Rephrased version:
System:"""

            inputs = tokenizer(single_prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature + 0.1 * i,  # Vary temperature slightly
                    top_p=0.95,
                    max_new_tokens=200,
                    num_return_sequences=1,
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the variant
            if 'System:' in generated:
                parts = generated.split('System:')
                if len(parts) > 1:
                    variant_text = 'System:' + parts[-1].strip()
                    # Clean up: remove any trailing "User:" or explanations
                    for stop_word in ['\nUser:', '\nTask:', '\nNote:', '\nOutput:']:
                        if stop_word in variant_text:
                            variant_text = variant_text.split(stop_word)[0].strip()
                    variants.append(variant_text)
    
    # Ensure we have at least the base prompt
    if not variants:
        variants = [base_prompt]
    elif len(variants) < num_variants:
        # Fill with base prompt if needed
        while len(variants) < num_variants:
            variants.append(base_prompt)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variants = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique_variants.append(v)
    
    return unique_variants[:num_variants]


def generate_all_variants(
    tokenizer,
    model,
    num_variants_per_prompt: int = 5,
    output_file: str = "data/prompt_variants.json",
):
    """
    Generate variants for all base prompts and save to file.
    """
    all_variants = {}
    
    print("Generating prompt variants for all dimensions...")
    print("=" * 80)
    
    for prompt_key, base_prompt in BASE_PROMPTS.items():
        print(f"\nGenerating variants for: {prompt_key}")
        print(f"Base prompt: {base_prompt[:80]}...")
        
        variants = generate_prompt_variants(
            tokenizer,
            model,
            base_prompt,
            num_variants=num_variants_per_prompt,
        )
        
        all_variants[prompt_key] = {
            "base": base_prompt,
            "variants": variants,
        }
        
        print(f"Generated {len(variants)} variants:")
        for i, variant in enumerate(variants, 1):
            print(f"  {i}. {variant[:100]}...")
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_variants, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Saved all variants to {output_path}")
    print(f"Total prompts: {len(BASE_PROMPTS)}")
    print(f"Variants per prompt: {num_variants_per_prompt}")
    print(f"Total variants: {sum(len(v['variants']) for v in all_variants.values())}")
    
    return all_variants


def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse prompt variants using Qwen"
    )
    parser.add_argument(
        "--num-variants",
        type=int,
        default=5,
        help="Number of variants to generate per prompt.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/prompt_variants.json",
        help="Output file path for saving variants.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help="Model ID to use for generation.",
    )

    args = parser.parse_args()

    print("Loading Qwen model...")
    tokenizer, model = load_qwen(args.model_id)

    print("\nGenerating prompt variants...")
    variants = generate_all_variants(
        tokenizer,
        model,
        num_variants_per_prompt=args.num_variants,
        output_file=args.output_file,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()

