#!/bin/bash
# 安装GGUF模型依赖

echo "安装llama-cpp-python库..."

# 安装llama-cpp-python（支持GPU）
pip install llama-cpp-python

# 如果需要GPU支持，可能需要编译安装
# CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

echo "依赖安装完成！"
echo "运行测试: python /home/developer/ai-infra-rag/scripts/test_local_gguf.py"
