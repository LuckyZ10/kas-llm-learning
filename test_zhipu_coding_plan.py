#!/usr/bin/env python3
"""
KAS + Zhipu AI Coding Plan 快速测试脚本

使用方式:
1. 设置环境变量:
   export ANTHROPIC_AUTH_TOKEN="your_api_key"
   export ANTHROPIC_BASE_URL="https://open.bigmodel.cn/api/anthropic"
   export API_TIMEOUT_MS="3000000"

2. 运行测试:
   python test_zhipu_coding_plan.py
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "kas"))

from kas.core.llm_client import LLMClient, create_llm_client_from_config
from kas.core.config import get_config, setup_wizard


def test_environment_config():
    """测试环境变量配置"""
    print("\n" + "="*60)
    print("测试 1: 环境变量配置")
    print("="*60)
    
    # 检查环境变量
    env_token = os.getenv('ANTHROPIC_AUTH_TOKEN')
    env_url = os.getenv('ANTHROPIC_BASE_URL')
    env_timeout = os.getenv('API_TIMEOUT_MS')
    
    print(f"ANTHROPIC_AUTH_TOKEN: {'已设置' if env_token else '未设置'}")
    print(f"ANTHROPIC_BASE_URL: {env_url or '未设置'}")
    print(f"API_TIMEOUT_MS: {env_timeout or '未设置'}")
    
    if not env_token:
        print("\n提示: 请设置 ANTHROPIC_AUTH_TOKEN 环境变量")
        print("export ANTHROPIC_AUTH_TOKEN='your_api_key'")
        return False
    
    return True


def test_client_creation():
    """测试客户端创建"""
    print("\n" + "="*60)
    print("测试 2: 客户端创建")
    print("="*60)
    
    try:
        client = LLMClient(
            api_key=os.getenv('ANTHROPIC_AUTH_TOKEN'),
            provider='zhipu',
            base_url=os.getenv('ANTHROPIC_BASE_URL', 'https://open.bigmodel.cn/api/anthropic'),
            timeout=float(os.getenv('API_TIMEOUT_MS', '3000000')) / 1000
        )
        
        print(f"✓ Provider: {client.provider}")
        print(f"✓ Model: {client.model}")
        print(f"✓ Base URL: {client.base_url}")
        print(f"✓ Timeout: {client.timeout}s")
        print("\n客户端创建成功!")
        return client
        
    except Exception as e:
        print(f"✗ 客户端创建失败: {e}")
        return None


def test_simple_chat(client):
    """测试简单对话"""
    print("\n" + "="*60)
    print("测试 3: 简单对话")
    print("="*60)
    
    try:
        response = client.chat(
            system_prompt="You are a helpful Python coding assistant.",
            user_message="Write a Python function to calculate fibonacci numbers.",
            temperature=0.7,
            max_tokens=500
        )
        
        print(f"✓ 响应长度: {len(response.content)} 字符")
        print(f"✓ Token 使用: {response.usage}")
        print(f"✓ 延迟: {response.latency_ms:.2f}ms")
        print(f"\n响应内容:\n{response.content[:500]}...")
        return True
        
    except Exception as e:
        print(f"✗ 对话失败: {e}")
        return False


def test_code_generation(client):
    """测试代码生成"""
    print("\n" + "="*60)
    print("测试 4: 代码生成")
    print("="*60)
    
    try:
        response = client.chat(
            system_prompt="You are an expert Python developer. Generate clean, efficient code.",
            user_message="""
Create a Python class that implements a simple REST API client with:
1. GET, POST, PUT, DELETE methods
2. Error handling
3. Retry logic with exponential backoff
4. Request/response logging
""",
            temperature=0.3,  # 更低的温度，更确定性的输出
            max_tokens=2000
        )
        
        print(f"✓ 响应长度: {len(response.content)} 字符")
        print(f"✓ Token 使用: {response.usage}")
        print(f"✓ 延迟: {response.latency_ms:.2f}ms")
        
        # 保存生成的代码
        output_file = project_root / "generated_code.py"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response.content)
        print(f"\n✓ 代码已保存到: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ 代码生成失败: {e}")
        return False


def test_streaming_chat(client):
    """测试流式对话"""
    print("\n" + "="*60)
    print("测试 5: 流式对话")
    print("="*60)
    
    try:
        print("流式响应: ", end="", flush=True)
        
        for chunk in client.chat_stream(
            system_prompt="You are a helpful assistant.",
            user_message="Count from 1 to 10, one number per line.",
            max_tokens=100
        ):
            print(chunk, end="", flush=True)
        
        print("\n\n✓ 流式对话成功!")
        return True
        
    except Exception as e:
        print(f"\n✗ 流式对话失败: {e}")
        return False


def test_config_integration():
    """测试配置集成"""
    print("\n" + "="*60)
    print("测试 6: 配置系统集成")
    print("="*60)
    
    try:
        config = get_config()
        
        # 设置智谱 AI 配置
        config.llm.provider = 'zhipu'
        config.llm.base_url = os.getenv('ANTHROPIC_BASE_URL')
        config.llm.timeout_ms = int(os.getenv('API_TIMEOUT_MS', '3000000'))
        
        # 设置 API Key
        if os.getenv('ANTHROPIC_AUTH_TOKEN'):
            config.set_api_key('zhipu', os.getenv('ANTHROPIC_AUTH_TOKEN'))
        
        print(f"✓ Provider: {config.llm.provider}")
        print(f"✓ Base URL: {config.llm.base_url}")
        print(f"✓ Timeout: {config.get_timeout()}s")
        print(f"✓ API Key 已配置: {bool(config.get_api_key('zhipu'))}")
        
        # 从配置创建客户端
        client = create_llm_client_from_config(config)
        
        if client:
            print("\n✓ 从配置创建客户端成功!")
            return True
        else:
            print("\n✗ 从配置创建客户端失败")
            return False
        
    except Exception as e:
        print(f"✗ 配置集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("KAS + Zhipu AI Coding Plan 测试")
    print("="*60)
    
    # 测试 1: 环境变量
    if not test_environment_config():
        print("\n跳过后续测试（需要配置 API Key）")
        return
    
    # 测试 2: 客户端创建
    client = test_client_creation()
    if not client:
        print("\n跳过后续测试（客户端创建失败）")
        return
    
    # 测试 3-5: 需要实际 API 调用，可选
    run_api_tests = os.getenv('RUN_API_TESTS', 'false').lower() == 'true'
    
    if run_api_tests:
        test_simple_chat(client)
        test_code_generation(client)
        test_streaming_chat(client)
    else:
        print("\n提示: 设置 RUN_API_TESTS=true 运行 API 调用测试")
    
    # 测试 6: 配置集成
    test_config_integration()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()
