#!/bin/bash
# 等待生成完成后自动运行评估
TASK_OUTPUT="/tmp/claude-0/-root-baseline-CSO-main/57377e4a-4f97-4438-9339-3111f1a3d12c/tasks/blaseh7v1.output"
BASELINES_DIR="/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines"

echo "[$(date)] Waiting for generation to complete..."

while true; do
    # 检查后台任务是否完成（输出中出现"All generations completed"）
    if grep -q "All generations completed" "$TASK_OUTPUT" 2>/dev/null; then
        echo "[$(date)] Generation completed! Starting evaluations..."
        break
    fi
    sleep 60
done

cd /root/baseline/CSO-main
echo "[$(date)] Running all evaluations..."
python3 run_all_evaluations.py 2>&1 | tee /root/baseline/CSO-main/eval_rerun.log
echo "[$(date)] All evaluations done."
