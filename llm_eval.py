# llm_eval.py
import json
import re
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from util import retry, generate

# 复制 EVAL_ROUND_PROMPT (5个维度，含 Cognitive Alignment)
EVAL_ROUND_PROMPT = '''You will play the role of a psychologist who is well versed in emotional support. There will be a dialog between the help seeker (i.e., the person seeking support) and the supporter (i.e., the person providing support). The dialog will consist of the help-seeker asking the supporter for help with some emotional problem, and the supporter will give the appropriate emotional support and advice.

Your task is to act as an impartial scoring judge and rate the last response of supporter in the dialog in terms of the following aspects in order to evaluate the quality of the response. Please read and fully understand the following scoring criteria.

Note that please give the scores in the specified format, just the serial number and the relevant dimension score from the list of questions, without repeating the question itself. Also, do not add other extraneous prefixes and control characters.

## Evaluation Criteria:

(1) Empathy: Focusing on the comprehension of user emotions and the delineation of the underlying logical framework of user emotions.

### Options:
4 points: The system exhibits a high degree of anthropomorphism, going so far as to console users in a friendly manner and assist them in analyzing the underlying logic of emotions.
3 points: Providing emotional comfort during conversations and assisting users in analyzing the underlying logical framework of their emotions.
2 points: The lack of understanding of user emotions or the absence of mechanisms to analyze user emotions are the main factors.
1 point: The lack of understanding of user emotions and the absence of mechanisms to analyze user emotions are the main factors.
0 points: The disregard for user concerns, the absence of assistance in analyzing user issues, and even the imposition of negative effects on user emotions.

(2) Information: Focusing on Evaluating the Reasonableness and Quantity of Recommendations Provided by Emotion Assistants.

### Options:
4 points: There are many suggestions, and all of them are effective.
3 points: There are more than five suggestions, but some of them are ineffective. There are fewer than five suggestions, but all of them are very effective.
2 points: The suggestions are fewer than five, and some suggestions are effective, while others provide numerous suggestions, but none of them touch the root of the problem.
1 point: Have suggestions but ineffective, as well as no suggestions.
0 points: Suggestions were provided, but all of them were ineffective, and some even gave advice that could potentially harm the user.

(3) Humanoid: Focus on the differences between emotional assistants and humans.

### Options:
4 points: There is no apparent difference from human friends.
3 points: 1-2 traces can reveal that the AI assistant is a language model.
2 points: More than two traces can reveal that the AI assistant is a language model.
1 point: Structured responses, or responses in the form of ’As a large language model’ or robot-like replies.
0 points: The dialogue exhibits rigidity and lacks comprehension in terms of internalizing the content.

(4) Strategies: Evaluating the Accuracy and Appropriateness of Emotional Support Strategies Used by Assistants

### Options:
4 points: The strategies are numerous, well-tailored to the user's emotional state, and demonstrate high empathy and effectiveness in addressing the user's concerns.
3 points: More than five strategies are provided, but some lack empathy or relevance. Alternatively, fewer than five strategies are shared, but they are highly empathetic and directly address the user's core emotional needs.
2 points: Fewer than five strategies are provided, and they are a mix of relevant and irrelevant approaches. Alternatively, a large number of strategies are given, but they fail to address the user's emotional root issues.
1 point: Strategies are present but lack empathy or relevance. Some may appear dismissive or insufficiently supportive in the context of the user's concerns.
0 points: Strategies are counterproductive, exacerbating the user's distress or dismissing their concerns. Some suggestions may inadvertently harm the user's emotional well-being.

(5) Cognitive Alignment: Evaluate whether the response aligns with key cognitive support principles derived from prefrontal regulation theory.

### Options:
4 points: The response fully reflects all five cognitive principles: (a) accurately gauges emotional intensity, (b) provides a reasonable and non-judgmental attribution, (c) validates the user's experience without questioning their methods or judgments, (d) shows appropriate risk awareness (e.g., suggests professional help when needed), and (e) demonstrates cognitively appropriate pacing, such as not moving into advice or reframing before adequate validation (e.g., does not give premature advice).
3 points: The response reflects at least four of the five principles well; minor omissions but overall cognitively sound.
2 points: The response reflects two or three principles; noticeable gaps in cognitive alignment.
1 point: The response reflects only one principle or none, with clear cognitive misalignment (e.g., misattributes, questions user's methods, ignores risks).
0 points: The response actively violates cognitive principles (e.g., gives harmful advice, dismisses user's emotions).

## Assessment Steps:

1. Read the conversation carefully to identify major topics and key points.
2. Read the Evaluation Criteria and compare them to the content of the conversation.
3. Based on the Evaluation Criteria, rate each aspect on a scale of 0 to 4, with 0 being the lowest and 4 being the highest.

What you need to do to evaluate this document:
{chat_history}

Please follow the response format below strictly, avoiding any positional bias and not letting the length of your response affect your evaluation. Evaluate the areas as objectively as possible.

## Answer format:

<Question number>: <Score>
'''

# 2. 评估单条对话的核心函数
@retry()
def get_llm_scores(full_conversation):
    query = EVAL_ROUND_PROMPT.format(chat_history=full_conversation)
    # 直接调用你的 util.py 中的 generate
    output = generate(query)
    
    scores = {}
    for i in range(1, 6):
        pattern = rf'\(?{i}\)?:\s*(\d+)'
        matches = re.search(pattern, output, re.I)
        if matches:
            scores[f"dim_{i}"] = int(matches.group(1))
        else:
            print(f"Warning: Failed to parse score for dimension {i} in output:\n{repr(output)}")
            scores[f"dim_{i}"] = 0
            
    return scores

def parse_llama_factory_prompt(prompt_text):
    """清洗 Llama-Factory 的指令格式，提取纯粹的 Conversation 文本。"""
    try:
        chat_part = prompt_text.split("Conversation:\n")[1]
        chat_history_clean = chat_part.split("<|eot_id|>")[0].strip()
        chat_history_clean = re.sub(r'Supporter:\s*$', '', chat_history_clean).strip()
        return chat_history_clean
    except Exception as e:
        return prompt_text  

# 多线程任务的单条处理逻辑
def process_line(line):
    data = json.loads(line)
    clean_history = parse_llama_factory_prompt(data['prompt'])
    full_conversation = f"{clean_history}\nSupporter: {data['predict'].strip()}"
    
    # 获取评分
    scores = get_llm_scores(full_conversation)
    
    # 将评分写回当前字典中，用于之后保存详情
    data['eval_scores'] = scores
    return data, scores

# 3. 批量评估与统计函数
def evaluate_file(file_path, max_workers=10):
    print(f"Evaluating {file_path} with {max_workers} threads...")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    total_scores = {"dim_1": 0, "dim_2": 0, "dim_3": 0, "dim_4": 0, "dim_5": 0}
    valid_count = 0
    results_detailed = []

    # 多线程并发执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_line, line): line for line in lines}
        
        for future in tqdm(as_completed(futures), total=len(lines), desc="Scoring"):
            try:
                data, scores = future.result()
                results_detailed.append(data)
                
                total_scores["dim_1"] += scores["dim_1"]
                total_scores["dim_2"] += scores["dim_2"]
                total_scores["dim_3"] += scores["dim_3"]
                total_scores["dim_4"] += scores["dim_4"]
                total_scores["dim_5"] += scores["dim_5"]
                valid_count += 1
            except Exception as exc:
                print(f"Line processing generated an exception: {exc}")
        
    if valid_count == 0:
        return None

    avg_scores = {k: v / valid_count for k, v in total_scores.items()}
    
    # 构造输出文件名
    base_dir, ext = os.path.splitext(file_path)

    os.makedirs("ablation_experiments_qwen", exist_ok=True)

    name = os.path.basename(os.path.dirname(file_path))
    
    # detail_file = f"ablation_experiments_qwen/{base_dir}_eval_details_qwen.jsonl"
    # summary_file = f"ablation_experiments_qwen/{base_dir}_eval_summary_qwen.txt"
    
    detail_file = f"ablation_experiments_qwen/{name}_eval_details_qwen.jsonl"
    summary_file = f"ablation_experiments_qwen/{name}_eval_summary_qwen.txt"

    # 生成控制台/Summary文件的输出文本
    summary_text = (
        f"{'='*50}\n"
        f"Results for {file_path} (Total: {valid_count})\n"
        f"{'='*50}\n"
        f">>> 1. Empathy:             {avg_scores['dim_1']:>4.2f} / 4.0\n"
        f">>> 2. Information:         {avg_scores['dim_2']:>4.2f} / 4.0\n"
        f">>> 3. Humanoid:            {avg_scores['dim_3']:>4.2f} / 4.0\n"
        f">>> 4. Strategies:          {avg_scores['dim_4']:>4.2f} / 4.0\n"
        f">>> 5. Cognitive Alignment: {avg_scores['dim_5']:>4.2f} / 4.0\n"
        f"{'='*50}\n"
    )
    
    print(f"\n{summary_text}")
    
    # 保存 Summary 文本文件
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
        
    # 保存包含评分详情的 jsonl 文件
    with open(detail_file, 'w', encoding='utf-8') as f:
        for item in results_detailed:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Details saved to: {detail_file}")
    print(f"Summary saved to: {summary_file}\n")
    
    return avg_scores

if __name__ == "__main__":
    BASE_FILE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base_qwens/generated_predictions.jsonl"
    COGNITIVE_FILE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_cognitive_qwens/generated_predictions.jsonl"
    ABLATION_FILE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_ablation_qwen/generated_predictions.jsonl"
    
    # 如果你是测试阶段，可以用你原来注掉的路径
    # BASE_FILE = "test_base.jsonl"
    # COGNITIVE_FILE = "test_cognitive.jsonl" 
    
    print("Start LLM-as-a-Judge Evaluation...")
    print("Using util.py for API calls and retry mechanisms.\n")
    
    # 开始评估，max_workers 建议设在 10~20 之间，以免瞬间并发过高被 API 提供商限流
    print(">>> Evaluating BASE")
    eval_base = evaluate_file(BASE_FILE, max_workers=30)

    print(">>> Evaluating FULL COGNITIVE")
    eval_cog = evaluate_file(COGNITIVE_FILE, max_workers=30)

    print(">>> Evaluating ABLATION (no validation-first)")
    eval_abl = evaluate_file(ABLATION_FILE, max_workers=30)
