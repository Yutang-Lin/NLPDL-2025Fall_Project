"""
Web interface for Multi-Dimensional Activation Steering
Provides a simple web UI with sliders to control four dimensions.
"""

import argparse
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from qwen_experiment_enhanced import (
    load_qwen,
    load_capability_data,
    format_multi_axis_pairs,
    compute_vector,
    compute_orthogonalized_vectors,
    MultiVectorActivationSteerer,
    generate_with_persona,
    evaluate_output,
    DEFAULT_LAYER_INDEX,
)

app = Flask(__name__)
CORS(app)

# Global variables to store model and vectors
tokenizer = None
model = None
vectors_dict = None
layer_indices = [DEFAULT_LAYER_INDEX]


def initialize_model_and_vectors(
    capability_file: str = "data/expert_data/expert_data.json",
    vectors_dir: str = "data/vectors_enhanced",
    layer_indices_list: list = None,
    max_samples: int = 128,
    use_prompt_variants: bool = True,
    no_orthogonalize: bool = False,
):
    """Initialize model and load/compute vectors."""
    global tokenizer, model, vectors_dict, layer_indices
    
    if layer_indices_list is None:
        layer_indices_list = [DEFAULT_LAYER_INDEX]
    
    layer_indices = sorted(set(layer_indices_list))
    
    print("Loading Qwen model...")
    tokenizer, model = load_qwen()
    
    print("Loading capability data...")
    data = load_capability_data(capability_file)
    print(f"Loaded {len(data)} Q&A pairs")
    
    print("Formatting multi-axis pairs...")
    pairs = format_multi_axis_pairs(
        data,
        use_variants=use_prompt_variants,
        cycle_variants=False,
    )
    
    vectors_dir_path = Path(vectors_dir)
    vectors_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Compute or load vectors
    v_cap_path = vectors_dir_path / f"v_cap_layers_{'_'.join(map(str, layer_indices))}.npz"
    v_style_path = vectors_dir_path / f"v_style_layers_{'_'.join(map(str, layer_indices))}.npz"
    v_density_path = vectors_dir_path / f"v_density_layers_{'_'.join(map(str, layer_indices))}.npz"
    v_correctness_path = vectors_dir_path / f"v_correctness_layers_{'_'.join(map(str, layer_indices))}.npz"
    
    import numpy as np
    
    if v_cap_path.exists() and v_style_path.exists() and v_density_path.exists():
        print("Loading existing vectors...")
        def load_vector_dict(path):
            data = np.load(path)
            return {int(k.split('_')[1]): data[k] for k in data.files if k.startswith('layer_')}
        
        v_cap_dict = load_vector_dict(v_cap_path)
        v_style_dict = load_vector_dict(v_style_path)
        v_density_dict = load_vector_dict(v_density_path)
        v_correctness_dict = load_vector_dict(v_correctness_path) if v_correctness_path.exists() else None
        
        # Filter to requested layers
        v_cap_dict = {k: v_cap_dict[k] for k in layer_indices if k in v_cap_dict}
        v_style_dict = {k: v_style_dict[k] for k in layer_indices if k in v_style_dict}
        v_density_dict = {k: v_density_dict[k] for k in layer_indices if k in v_density_dict}
        if v_correctness_dict:
            v_correctness_dict = {k: v_correctness_dict[k] for k in layer_indices if k in v_correctness_dict}
    else:
        print("Computing vectors...")
        v_cap_dict = compute_vector(
            pairs["capability"], tokenizer, model, layer_indices, max_samples, "v_cap"
        )
        v_style_dict = compute_vector(
            pairs["style"], tokenizer, model, layer_indices, max_samples, "v_style"
        )
        v_density_dict = compute_vector(
            pairs["density"], tokenizer, model, layer_indices, max_samples, "v_density"
        )
        v_correctness_dict = compute_vector(
            pairs["correctness"], tokenizer, model, layer_indices, max_samples, "v_correctness"
        )
        
        # Save vectors
        np.savez(v_cap_path, **{f"layer_{k}": v for k, v in v_cap_dict.items()})
        np.savez(v_style_path, **{f"layer_{k}": v for k, v in v_style_dict.items()})
        np.savez(v_density_path, **{f"layer_{k}": v for k, v in v_density_dict.items()})
        np.savez(v_correctness_path, **{f"layer_{k}": v for k, v in v_correctness_dict.items()})
    
    # Prepare vectors (with or without orthogonalization)
    if no_orthogonalize:
        print("Using raw vectors (no orthogonalization)...")
        vectors_dict = {}
        for layer_idx in layer_indices:
            vectors_dict[layer_idx] = {
                'v_cap': v_cap_dict[layer_idx],
                'v_style': v_style_dict[layer_idx],
                'v_density': v_density_dict[layer_idx],
                'v_cap_pure': v_cap_dict[layer_idx],
            }
            if v_correctness_dict and layer_idx in v_correctness_dict:
                vectors_dict[layer_idx]['v_correctness'] = v_correctness_dict[layer_idx]
    else:
        print("Applying Gram-Schmidt orthogonalization...")
        vectors_dict = compute_orthogonalized_vectors(
            v_cap_dict, v_style_dict, v_density_dict, v_correctness_dict
        )
    
    return vectors_dict


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API endpoint for text generation with steering."""
    global tokenizer, model, vectors_dict
    
    if tokenizer is None or model is None or vectors_dict is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        data = request.json
        question = data.get('question', 'Explain quantum entanglement.')
        alpha_cap = float(data.get('alpha_cap', 0.0))
        alpha_style = float(data.get('alpha_style', 0.0))
        alpha_density = float(data.get('alpha_density', 0.0))
        alpha_correctness = float(data.get('alpha_correctness', 0.0))
        persona = data.get('persona', 'low')
        
        # Generate with steering
        with MultiVectorActivationSteerer(
            model, vectors_dict, alpha_cap, alpha_style, alpha_density, alpha_correctness
        ):
            generated_text = generate_with_persona(tokenizer, model, question, persona=persona)
        
        # Evaluate output
        metrics = evaluate_output(generated_text)
        
        return jsonify({
            'text': generated_text,
            'metrics': metrics,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/status', methods=['GET'])
def api_status():
    """Check if model is ready."""
    global tokenizer, model, vectors_dict
    return jsonify({
        'ready': tokenizer is not None and model is not None and vectors_dict is not None,
        'layers': layer_indices
    })


def main():
    parser = argparse.ArgumentParser(description="Web interface for Activation Steering")
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    parser.add_argument(
        '--capability-file',
        type=str,
        default='data/expert_data/expert_data.json',
        help='Path to capability dataset'
    )
    parser.add_argument(
        '--vectors-dir',
        type=str,
        default='data/vectors_enhanced',
        help='Directory for vectors'
    )
    parser.add_argument(
        '--layer-indices',
        type=int,
        nargs='+',
        default=[DEFAULT_LAYER_INDEX],
        help='Layer indices'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=128,
        help='Max samples per vector type'
    )
    parser.add_argument(
        '--no-orthogonalize',
        action='store_true',
        help='Skip orthogonalization'
    )
    parser.add_argument(
        '--no-prompt-variants',
        action='store_true',
        help='Do not use prompt variants'
    )
    
    args = parser.parse_args()
    
    # Initialize model and vectors
    print("Initializing model and vectors...")
    global vectors_dict
    vectors_dict = initialize_model_and_vectors(
        capability_file=args.capability_file,
        vectors_dir=args.vectors_dir,
        layer_indices_list=args.layer_indices,
        max_samples=args.max_samples,
        use_prompt_variants=not args.no_prompt_variants,
        no_orthogonalize=args.no_orthogonalize,
    )
    
    print(f"\nWeb server starting on http://{args.host}:{args.port}")
    print("Open your browser and navigate to the URL above")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()

