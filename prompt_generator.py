import json

def format_data_for_experiment(capability_file, style_file):
    """
    Load JSON data and format it into experiment-ready prompt lists with English templates.
    """
    
    # 1. Load Data
    try:
        with open(capability_file, 'r', encoding='utf-8') as f:
            cap_data = json.load(f)
        with open(style_file, 'r', encoding='utf-8') as f:
            style_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return [], []

    # ==========================================
    # Group A: Capability Pairs (For extracting Intelligence Vector)
    # ==========================================
    # Goal: Force the model to process the SAME high-level answer under different personas.
    
    # Low-Level Persona: Focus on simplicity, intuition, and limited vocabulary.
    sys_prompt_low = (
        "System: You are a curious 5-year-old child. "
        "Please explain things using very simple words and basic intuition that a kindergartener would understand."
    )
    
    # High-Level Persona: Focus on rigor, depth, and academic authority.
    sys_prompt_high = (
        "System: You are a world-renowned expert and Fields Medalist in this field. "
        "Please provide a rigorous, comprehensive, and theoretically profound explanation."
    )
    
    cap_pairs_formatted = []
    for item in cap_data:
        # 适配之前代码生成的字段名 (json keys)
        # 如果是"核心概念"数据，keys 应该是 'question' 和 'answer'
        q = item.get('question', '')
        a = item.get('answer', '') 
        
        if not q or not a: continue # Skip empty items
        
        # Construct Full Text for Teacher Forcing
        # The 'Assistant' part is IDENTICAL (the expert answer) for both cases.
        # We measure the model's "surprise" or internal resistance to saying smart things when acting as a child.
        text_low = f"{sys_prompt_low}\nUser: {q}\nAssistant: {a}"
        text_high = f"{sys_prompt_high}\nUser: {q}\nAssistant: {a}"
        
        cap_pairs_formatted.append({
            "type": "capability",
            "text_low": text_low,
            "text_high": text_high,
            "target_answer": a 
        })

    # ==========================================
    # Group B: Style Pairs (For Orthogonal Projection / De-noising)
    # ==========================================
    # Goal: Extract a vector that represents pure "Style/Jargon" without semantic difference.
    
    sys_prompt_casual = "System: Rewrite the following sentence to be extremely casual, using slang and informal grammar."
    sys_prompt_formal = "System: Rewrite the following sentence to be extremely formal, academic, and sophisticated."
    
    style_pairs_formatted = []
    for item in style_data:
        # 适配之前代码生成的字段名
        # keys 应该是 'neutral' (作为输入), 'casual' (作为输出1), 'formal' (作为输出2)
        # 注意：这里可能需要根据你实际生成的 JSON key 做微调，下面按之前的 generate_style_pairs_dataset 代码逻辑适配
        content = item.get('neutral', '')       # The neutral input
        casual_out = item.get('casual', '')     # The slang output
        formal_out = item.get('formal', '')     # The academic output
        
        if not content or not casual_out: continue

        # For Style Pairs, we look at the difference in the Assistant's output.
        text_casual = f"{sys_prompt_casual}\nUser: {content}\nAssistant: {casual_out}"
        text_formal = f"{sys_prompt_formal}\nUser: {content}\nAssistant: {formal_out}"
        
        style_pairs_formatted.append({
            "type": "style",
            "text_casual": text_casual,
            "text_formal": text_formal
        })
        
    print(f"Loaded {len(cap_pairs_formatted)} capability pairs and {len(style_pairs_formatted)} style pairs.")
    return cap_pairs_formatted, style_pairs_formatted

# Example Usage:
# caps, styles = format_data_for_experiment('dataset_core_concepts_final.json', 'dataset_style_pairs_final.json')
# print(caps[0]['text_low'])