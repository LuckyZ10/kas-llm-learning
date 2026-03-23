#!/usr/bin/env python3
"""Simple test for Zhipu AI Coding Plan"""
import os
import sys

# Set environment
os.environ['ANTHROPIC_AUTH_TOKEN'] = '25e14035e3ca42f2b4ef9f7c3fef3b02.d9Nl65FEuxW6MiQp'
os.environ['ANTHROPIC_BASE_URL'] = 'https://open.bigmodel.cn/api/anthropic'
os.environ['API_TIMEOUT_MS'] = '3000000'

# Add project path
sys.path.insert(0, 'kas')

from kas.core.llm_client import LLMClient

print("="*60)
print("Zhipu AI Coding Plan Test")
print("="*60)

# Test 1: Client creation
print("\nTest 1: Client Creation")
try:
    client = LLMClient(
        api_key=os.environ['ANTHROPIC_AUTH_TOKEN'],
        provider='zhipu',
        base_url=os.environ['ANTHROPIC_BASE_URL'],
        timeout=3000.0
    )
    print(f"  Provider: {client.provider}")
    print(f"  Model: {client.model}")
    print(f"  Base URL: {client.base_url}")
    print(f"  Timeout: {client.timeout}s")
    print("  [OK] Client created successfully!")
except Exception as e:
    print(f"  [FAIL] {e}")
    sys.exit(1)

# Test 2: Simple chat
print("\nTest 2: Simple Chat")
try:
    response = client.chat(
        system_prompt="You are a helpful assistant.",
        user_message="Say hello in one sentence.",
        temperature=0.7,
        max_tokens=100
    )
    print(f"  Response: {response.content[:100]}...")
    print(f"  Tokens: {response.usage}")
    print(f"  Latency: {response.latency_ms:.0f}ms")
    print("  [OK] Chat successful!")
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

# Test 3: Code generation
print("\nTest 3: Code Generation")
try:
    response = client.chat(
        system_prompt="You are a Python expert. Generate clean code.",
        user_message="Write a Python function to reverse a string.",
        temperature=0.3,
        max_tokens=500
    )
    print(f"  Response length: {len(response.content)} chars")
    print(f"  Tokens: {response.usage}")
    print("  [OK] Code generation successful!")
except Exception as e:
    print(f"  [FAIL] {e}")

print("\n" + "="*60)
print("All tests completed!")
print("="*60)
