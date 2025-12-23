import os
import google.generativeai as genai
import json
import time
from tqdm import tqdm

# ================= 配置区域 =================
os.environ["GOOGLE_API_KEY"] = ""
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = genai.GenerativeModel('gemini-3-flash-preview')

# 定义 10 个生活化/通用话题，确保覆盖面广
TOPICS = [
    # 1. 厨房里的科学 (化学/物理基础，但不涉及高深公式)
    "Cooking Science",
    
    # 2. 自然现象 (地球科学基础)
    "Weather Phenomena",
    
    # 3. 身体与健康 (生物/医学基础)
    "Human Body",
    
    # 4. 交通与机械 (工程学基础)
    "Transportation",
    
    # 5. 金钱与交易 (经济学基础)
    "Money & Value",
    
    # 6. 建筑与结构 (力学基础)
    "Architecture",
    
    # 7. 声音与光 (波的基础)
    "Sound & Light",
    
    # 8. 互联网与通信 (信息论基础)
    "Digital Communication",
    
    # 9. 动植物行为 (生态学基础)
    "Nature & Wildlife",
    
    # 10. 社会规则 (社会学/心理学基础)
    "Social Dynamics"
]

TARGET_PER_TOPIC = 50
BATCH_SIZE = 5

# ================= 核心生成逻辑 =================

def generate_style_pairs_dataset():
    all_data = []
    
    print(f"🚀 开始生成风格差异数据 (Style Pairs)")
    print(f"🌍 话题数量: {len(TOPICS)}")
    print(f"🎯 目标总数: {len(TOPICS) * TARGET_PER_TOPIC}")
    
    for topic in TOPICS:
        print(f"\nProcessing Topic: {topic}...")
        topic_data = []
        num_batches = TARGET_PER_TOPIC // BATCH_SIZE
        
        for i in tqdm(range(num_batches), desc=f"Generating {topic}"):
            
            # ====================================================
            # 风格差异专用 PROMPT
            # ====================================================
            prompt = f"""
            Role: You are an expert educator and science communicator with mastery over different pedagogical tones.
            
            Task: Generate {BATCH_SIZE} triples of sentences related to the topic: "{topic}".
            
            Structure for each triple:
            1.  **Neutral:** A simple, factual, standard English sentence (The core meaning).
            2.  **Simple_Explanation:** The same meaning, but rewritten for a target audience of **K12 children or the elderly**. The tone should be warm, patient, encouraging, and extremely easy to follow.
            3.  **Advanced_Explanation:** The same meaning, but rewritten for a target audience of **Graduate Students or Peers**. The tone should be rigorous, dense, precise, and assume high cognitive load capacity.
            
            Guidelines:
            - The **meaning** must remain exactly the same across all three versions.
            - **Simple_Explanation:** Use short sentences, active voice, simple analogies, and a "gentle/guiding" tone. Avoid jargon. (e.g., "Think of it like...", "This helps us to...")
            - **Advanced_Explanation:** Use domain-specific terminology (appropriate for the topic), complex syntactic structures, and precise definitions. (e.g., "The mechanism implies...", "Correlation suggests...")
            - Strictly English.
            - Return a RAW JSON list.
            
            Output Example:
            [
                {{
                    "topic": "{topic}",
                    "neutral": "Plants need sunlight to grow.",
                    "simple_explanation": "Just like you need food to get big and strong, plants need sunshine to make their own energy! It's their favorite snack.",
                    "advanced_explanation": "Photosynthesis is the fundamental endothermic reaction by which autotrophs convert light energy into chemical energy to sustain growth."
                }}
            ]
            """
            
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.8 # 稍微高一点，让风格差异更夸张
                    }
                )
                
                batch_items = json.loads(response.text)
                if isinstance(batch_items, list):
                    topic_data.extend(batch_items)
                
                time.sleep(1.0)
                
            except Exception as e:
                print(f"⚠️ Error in batch {i}: {e}")
                time.sleep(2)

        all_data.extend(topic_data)
        # 断点保存
        save_to_file(all_data, filename="dataset_style_pairs_partial.json")

    # 最终保存
    save_to_file(all_data, filename="dataset_style_pairs_final.json")
    print(f"\n✅ 全部完成！共生成 {len(all_data)} 条风格数据。")

def save_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generate_style_pairs_dataset()