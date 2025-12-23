"""
Web interface for Multi-Dimensional Activation Steering
Provides a simple web UI with sliders to control four dimensions.
"""

import argparse
import json
import threading
import gc
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import torch

from qwen_experiment_enhanced import (
    load_qwen,
    load_capability_data,
    format_multi_axis_pairs,
    compute_vector,
    compute_orthogonalized_vectors,
    MultiVectorActivationSteerer,
    generate_with_persona,
    generate_with_persona_batch,
    generate_with_persona_streaming,
    generate_with_persona_streaming_parallel,
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

# Track active generations for stopping
active_generations = {}
generation_lock = threading.Lock()

def stop_all_generations():
    """Stop all active generations immediately. Used for single-instance mode."""
    global active_generations, generation_lock
    
    with generation_lock:
        # Get all generation IDs before modifying
        all_gen_ids = list(active_generations.keys())
        
        for generation_id in all_gen_ids:
            gen_info = active_generations[generation_id]
            gen_info['stop_requested'] = True
            gen_info['force_stop'] = True
            
            # Set stop flag to stop generation thread
            stop_flag = gen_info.get('stop_flag')
            if stop_flag is not None:
                stop_flag[0] = True
            
            # Clean up steerer if it exists
            steerer_ref = gen_info.get('steerer')
            if steerer_ref and steerer_ref[0] is not None:
                try:
                    steerer = steerer_ref[0]
                    if hasattr(steerer, 'handles') and steerer.handles:
                        for layer_idx, handle in list(steerer.handles):
                            try:
                                handle.remove()
                            except:
                                pass
                        steerer.handles.clear()
                    steerer_ref[0] = None
                except Exception as e:
                    print(f"Error cleaning up steerer: {e}")
        
        # Clear all active generations
        active_generations.clear()
        
        # Clean up GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def stop_generation(generation_id):
    """Stop a specific generation and clean it up."""
    global active_generations, generation_lock
    
    with generation_lock:
        if generation_id not in active_generations:
            return
        
        gen_info = active_generations[generation_id]
        gen_info['stop_requested'] = True
        gen_info['force_stop'] = True
        
        # Set stop flag to stop generation thread
        stop_flag = gen_info.get('stop_flag')
        if stop_flag is not None:
            stop_flag[0] = True
        
        # Clean up steerer if it exists
        steerer_ref = gen_info.get('steerer')
        if steerer_ref and steerer_ref[0] is not None:
            try:
                steerer = steerer_ref[0]
                if hasattr(steerer, 'handles') and steerer.handles:
                    for layer_idx, handle in list(steerer.handles):
                        try:
                            handle.remove()
                        except:
                            pass
                    steerer.handles.clear()
                steerer_ref[0] = None
            except Exception as e:
                print(f"Error cleaning up steerer: {e}")
        
        # Remove from active generations
        del active_generations[generation_id]
        
        # Light cleanup - only if this was the last generation
        if not active_generations:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


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
    """API endpoint for text generation with steering. Supports parallel generation."""
    global tokenizer, model, vectors_dict, active_generations, generation_lock
    import uuid
    
    if tokenizer is None or model is None or vectors_dict is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    # Single-instance mode: stop all existing generations before starting new one
    stop_all_generations()
    
    try:
        data = request.json
        question = data.get('question', 'Explain quantum entanglement.')
        alpha_cap = float(data.get('alpha_cap', 0.0))
        alpha_style = float(data.get('alpha_style', 0.0))
        alpha_density = float(data.get('alpha_density', 0.0))
        alpha_correctness = float(data.get('alpha_correctness', 0.0))
        persona = data.get('persona', 'low')
        num_responses = int(data.get('num_responses', 1))
        generation_id = data.get('generation_id', str(uuid.uuid4()))
        
        # Initialize generation tracking
        with generation_lock:
            active_generations[generation_id] = {'stop_requested': False}
        
        steerer = None
        try:
            steerer = MultiVectorActivationSteerer(
                model, vectors_dict, alpha_cap, alpha_style, alpha_density, alpha_correctness
            )
            
            with steerer:
                # Check if stop was requested
                with generation_lock:
                    if active_generations.get(generation_id, {}).get('stop_requested', False):
                        return jsonify({'error': 'Generation stopped', 'success': False}), 400
                
                if num_responses == 1:
                    # Single response
                    generated_text = generate_with_persona(tokenizer, model, question, persona=persona)
                    
                    # Check stop after generation
                    with generation_lock:
                        if active_generations.get(generation_id, {}).get('stop_requested', False):
                            return jsonify({'error': 'Generation stopped', 'success': False}), 400
                    
                    metrics = evaluate_output(generated_text)
                    
                    return jsonify({
                        'text': generated_text,
                        'metrics': metrics,
                        'success': True,
                        'generation_id': generation_id,
                        'responses': [{
                            'text': generated_text,
                            'metrics': metrics,
                            'id': 0
                        }]
                    })
                else:
                    # Batched generation
                    generated_texts = generate_with_persona_batch(
                        tokenizer, model, question, persona=persona, num_responses=num_responses
                    )
                    
                    # Check stop after generation
                    with generation_lock:
                        if active_generations.get(generation_id, {}).get('stop_requested', False):
                            return jsonify({'error': 'Generation stopped', 'success': False}), 400
                    
                    # Process results
                    response_list = []
                    for i, generated_text in enumerate(generated_texts):
                        try:
                            metrics = evaluate_output(generated_text)
                            response_list.append({
                                'text': generated_text,
                                'metrics': metrics,
                                'id': i
                            })
                        except Exception as e:
                            response_list.append({
                                'text': f'Error evaluating response: {str(e)}',
                                'metrics': {},
                                'id': i,
                                'error': True
                            })
                    
                    return jsonify({
                        'success': True,
                        'generation_id': generation_id,
                        'responses': response_list,
                        'num_responses': len(response_list)
                    })
        
        finally:
            # Clean up steerer
            if steerer is not None:
                try:
                    # Context manager should handle cleanup, but ensure it's done
                    if hasattr(steerer, 'handles'):
                        for layer_idx, handle in list(steerer.handles):
                            try:
                                handle.remove()
                            except:
                                pass
                        steerer.handles.clear()
                except:
                    pass
            
            # Clean up generation tracking
            with generation_lock:
                if generation_id in active_generations:
                    del active_generations[generation_id]
            
            # Light cleanup only if no other generations are active
            with generation_lock:
                if not active_generations:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
    
    except Exception as e:
        # Clean up on error
        with generation_lock:
            if 'generation_id' in locals() and generation_id in active_generations:
                del active_generations[generation_id]
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/generate/stream', methods=['POST', 'GET'])
def api_generate_stream():
    """API endpoint for streaming text generation with steering."""
    global tokenizer, model, vectors_dict, active_generations, generation_lock
    import uuid
    
    if tokenizer is None or model is None or vectors_dict is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    # Single-instance mode: stop all existing generations before starting new one
    stop_all_generations()
    
    try:
        # Support both GET (for EventSource) and POST
        if request.method == 'GET':
            question = request.args.get('question', 'Explain quantum entanglement.')
            alpha_cap = float(request.args.get('alpha_cap', 0.0))
            alpha_style = float(request.args.get('alpha_style', 0.0))
            alpha_density = float(request.args.get('alpha_density', 0.0))
            alpha_correctness = float(request.args.get('alpha_correctness', 0.0))
            persona = request.args.get('persona', 'low')
            response_id = int(request.args.get('response_id', 0))
            num_responses = int(request.args.get('num_responses', 1))
            generation_id = request.args.get('generation_id', str(uuid.uuid4()))
        else:
            data = request.json
            question = data.get('question', 'Explain quantum entanglement.')
            alpha_cap = float(data.get('alpha_cap', 0.0))
            alpha_style = float(data.get('alpha_style', 0.0))
            alpha_density = float(data.get('alpha_density', 0.0))
            alpha_correctness = float(data.get('alpha_correctness', 0.0))
            persona = data.get('persona', 'low')
            response_id = int(data.get('response_id', 0))
            num_responses = int(data.get('num_responses', 1))
            generation_id = data.get('generation_id', str(uuid.uuid4()))
        
        # Initialize generation tracking with steerer reference for dynamic updates
        steerer_ref = [None]  # Use list to make it mutable in nested function
        stop_flag = [False]  # Use list to make it mutable for ForceStopCriteria
        
        with generation_lock:
            active_generations[generation_id] = {
                'stop_requested': False,
                'steerer': steerer_ref,
                'stop_flag': stop_flag  # Store stop flag for external control
            }
        
        def generate():
            steerer = None
            try:
                steerer = MultiVectorActivationSteerer(
                    model, vectors_dict, alpha_cap, alpha_style, alpha_density, alpha_correctness
                )
                steerer_ref[0] = steerer  # Store reference for dynamic updates
                
                with steerer:
                    if num_responses == 1:
                        # Single response - use simple streaming
                        for chunk in generate_with_persona_streaming(
                            tokenizer, model, question, persona=persona, stop_flag=stop_flag
                        ):
                            # Check if stop was requested
                            with generation_lock:
                                gen_info = active_generations.get(generation_id)
                                if not gen_info or gen_info.get('stop_requested', False) or gen_info.get('force_stop', False):
                                    stop_flag[0] = True
                                    yield f"data: {json.dumps({'chunk': '', 'response_id': 0, 'done': True, 'stopped': True, 'generation_id': generation_id})}\n\n"
                                    return
                            
                            yield f"data: {json.dumps({'chunk': chunk, 'response_id': 0, 'done': False, 'generation_id': generation_id})}\n\n"
                        
                        yield f"data: {json.dumps({'chunk': '', 'response_id': 0, 'done': True, 'generation_id': generation_id})}\n\n"
                    else:
                        # Multiple responses - use parallel streaming
                        # Track which responses should be marked as stopped (but let generation continue)
                        should_mark_stopped = set()
                        
                        for response_id, chunk in generate_with_persona_streaming_parallel(
                            tokenizer, model, question, persona=persona, 
                            num_responses=num_responses, stop_flag=None  # Don't use stop_flag to interrupt - let all finish
                        ):
                            # Check if stop was requested (mark for stopped display, but don't interrupt generation)
                            with generation_lock:
                                gen_info = active_generations.get(generation_id)
                                if gen_info and (gen_info.get('stop_requested', False) or gen_info.get('force_stop', False)):
                                    # Mark all responses to show as stopped, but let them continue generating
                                    should_mark_stopped = set(range(num_responses))
                            
                            if chunk is None:
                                # Completion signal - mark as stopped if stop was requested
                                if response_id in should_mark_stopped:
                                    yield f"data: {json.dumps({'chunk': '', 'response_id': response_id, 'done': True, 'stopped': True, 'generation_id': generation_id})}\n\n"
                                else:
                                    yield f"data: {json.dumps({'chunk': '', 'response_id': response_id, 'done': True, 'generation_id': generation_id})}\n\n"
                            else:
                                # Regular chunk - continue streaming even if marked for stop
                                yield f"data: {json.dumps({'chunk': chunk, 'response_id': response_id, 'done': False, 'generation_id': generation_id})}\n\n"
            except GeneratorExit:
                # Generator was closed (e.g., EventSource closed)
                pass
            except Exception as e:
                # Only send error if not force stopped
                with generation_lock:
                    gen_info = active_generations.get(generation_id, {})
                    if not gen_info.get('force_stop', False):
                        yield f"data: {json.dumps({'error': str(e), 'response_id': response_id, 'done': True, 'generation_id': generation_id})}\n\n"
            finally:
                # Clean up steerer
                if steerer is not None:
                    try:
                        if hasattr(steerer, 'handles') and steerer.handles:
                            for layer_idx, handle in list(steerer.handles):
                                try:
                                    handle.remove()
                                except:
                                    pass
                            steerer.handles.clear()
                        steerer_ref[0] = None
                    except Exception as e:
                        print(f"Error cleaning up steerer: {e}")
                
                # Clean up generation entry
                with generation_lock:
                    if generation_id in active_generations:
                        del active_generations[generation_id]
                
                # Light cleanup only if no other generations are active
                with generation_lock:
                    if not active_generations:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
        
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
    
    except Exception as e:
        # Clean up on error
        with generation_lock:
            if 'generation_id' in locals() and generation_id in active_generations:
                del active_generations[generation_id]
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/update_alphas', methods=['POST'])
def api_update_alphas():
    """Update alpha values for an ongoing streaming generation."""
    global active_generations, generation_lock
    
    try:
        data = request.json
        generation_id = data.get('generation_id')
        alpha_cap = data.get('alpha_cap')
        alpha_style = data.get('alpha_style')
        alpha_density = data.get('alpha_density')
        alpha_correctness = data.get('alpha_correctness')
        
        if not generation_id:
            return jsonify({'error': 'generation_id required', 'success': False}), 400
        
        with generation_lock:
            if generation_id not in active_generations:
                return jsonify({'error': 'Generation not found', 'success': False}), 404
            
            steerer_ref = active_generations[generation_id].get('steerer')
            if not steerer_ref or steerer_ref[0] is None:
                return jsonify({'error': 'Generation not in streaming mode', 'success': False}), 400
            
            # Get steerer outside lock to avoid holding lock during update
            steerer = steerer_ref[0]
        
        # Update alpha values outside lock to avoid blocking
        # The hook reads from steerer.alpha_*[0] which is thread-safe for reads/writes
        updated = steerer.update_alphas(
            alpha_cap=float(alpha_cap) if alpha_cap is not None else None,
            alpha_style=float(alpha_style) if alpha_style is not None else None,
            alpha_density=float(alpha_density) if alpha_density is not None else None,
            alpha_correctness=float(alpha_correctness) if alpha_correctness is not None else None,
        )
        
        if updated:
            # Get current values
            current_alphas = {
                'alpha_cap': steerer.alpha_cap[0],
                'alpha_style': steerer.alpha_style[0],
                'alpha_density': steerer.alpha_density[0],
                'alpha_correctness': steerer.alpha_correctness[0],
            }
            
            return jsonify({
                'success': True, 
                'message': 'Alpha values updated',
                'alphas': current_alphas
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No alpha values were updated'
            })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop ongoing generation forcefully."""
    global active_generations, generation_lock
    
    try:
        data = request.json
        generation_id = data.get('generation_id')
        force = data.get('force', False)
        
        if not generation_id:
            return jsonify({'error': 'generation_id required', 'success': False}), 400
        
        if force:
            # Force stop: clean up immediately
            stop_generation(generation_id)
            return jsonify({
                'success': True, 
                'message': 'Generation stopped and cleaned up',
                'force': True
            })
        else:
            # Soft stop: just set flag
            with generation_lock:
                if generation_id in active_generations:
                    active_generations[generation_id]['stop_requested'] = True
                    return jsonify({
                        'success': True, 
                        'message': 'Stop requested',
                        'force': False
                    })
                else:
                    return jsonify({
                        'success': True, 
                        'message': 'Generation not found (may have already completed)'
                    })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/save', methods=['POST'])
def api_save():
    """Save current outputs to a JSON file."""
    try:
        data = request.json
        outputs = data.get('outputs', [])
        question = data.get('question', '')
        parameters = data.get('parameters', {})
        
        if not outputs:
            return jsonify({'error': 'No outputs to save', 'success': False}), 400
        
        # Create output directory if it doesn't exist
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f'generation_{timestamp}.json'
        
        # Prepare data to save
        save_data = {
            'timestamp': timestamp,
            'question': question,
            'parameters': parameters,
            'outputs': outputs
        }
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'filename': str(filename),
            'message': f'Saved {len(outputs)} outputs to {filename}'
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

