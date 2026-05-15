#!/bin/bash

echo "Waiting for generation to complete..."

# 等待生成进程结束
while ps aux | grep -q "[g]enerate_strong_baseline_responses.py"; do
    sleep 60
    python check_progress.py
done

echo ""
echo "Generation completed! Starting evaluation pipeline..."
echo ""

# 运行评估
python run_all_evaluations.py

