"""
Centralized path configuration for supplemental experiments.
Set environment variables to override defaults, or edit this file directly.
"""
import os

# Root of the repo (one level up from this file)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _p(*parts):
    return os.path.join(REPO_ROOT, *parts)

RESPONSE_FILES = {
    "Llama-Base":      os.getenv("LLAMA_BASE_PREDICTIONS",      _p("data/predictions/llama_base.jsonl")),
    "Llama-Cognitive": os.getenv("LLAMA_COGNITIVE_PREDICTIONS", _p("data/predictions/llama_cognitive.jsonl")),
    "Qwen-Base":       os.getenv("QWEN_BASE_PREDICTIONS",       _p("data/predictions/qwen_base.jsonl")),
    "Qwen-Cognitive":  os.getenv("QWEN_COGNITIVE_PREDICTIONS",  _p("data/predictions/qwen_cognitive.jsonl")),
    "GPT-4o":          os.getenv("GPT4O_PREDICTIONS",           _p("data/predictions/gpt4o.jsonl")),
    "Claude-Sonnet":   os.getenv("CLAUDE_PREDICTIONS",          _p("data/predictions/claude_sonnet.jsonl")),
    "GPT-4o-mini":     os.getenv("GPT4O_MINI_PREDICTIONS",      _p("data/predictions/gpt4o_mini.jsonl")),
}

POINTWISE_FILES = {
    "Llama-Base":      os.getenv("LLAMA_BASE_POINTWISE",      _p("eval_with_baselines_results/eval_base_eval_details.jsonl")),
    "Llama-Cognitive": os.getenv("LLAMA_COGNITIVE_POINTWISE", _p("eval_with_baselines_results/eval_cognitive_fix_eval_details.jsonl")),
    "Qwen-Base":       os.getenv("QWEN_BASE_POINTWISE",       _p("eval_with_baselines_results/eval_base_qwens_eval_details.jsonl")),
    "Qwen-Cognitive":  os.getenv("QWEN_COGNITIVE_POINTWISE",  _p("eval_with_baselines_results/eval_cognitive_qwens_eval_details.jsonl")),
    "GPT-4o":          os.getenv("GPT4O_POINTWISE",           _p("eval_with_baselines_results/gpt4o_eval_details.jsonl")),
    "Claude-Sonnet":   os.getenv("CLAUDE_POINTWISE",          _p("eval_with_baselines_results/claude_sonnet_eval_details.jsonl")),
    "GPT-4o-mini":     os.getenv("GPT4O_MINI_POINTWISE",      _p("eval_with_baselines_results/gpt4o_mini_eval_details.jsonl")),
}
