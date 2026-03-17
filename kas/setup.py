"""
KAS - Klaw Agent Studio
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="klaw-agent-studio",
    version="0.2.0",
    author="Yilin.zhang",
    author_email="zhangyilin210@gmail.com",
    description="专业开发者的 CLI-first Agent 孵化平台",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LuckyZ10/kas-llm-learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
    ],
    extras_require={
        'llm': ['openai>=1.0.0'],
        'cloud': ['fastapi>=0.109.0', 'uvicorn>=0.27.0', 'pyjwt>=2.8.0'],
        'dashboard': ['flask>=2.0.0'],
        'knowledge': ['chromadb>=0.4.0', 'requests>=2.28.0'],
        'dev': ['pytest>=7.0.0', 'black>=23.0.0'],
        'all': ['openai>=1.0.0', 'fastapi>=0.109.0', 'uvicorn>=0.27.0', 
                'pyjwt>=2.8.0', 'flask>=2.0.0', 'chromadb>=0.4.0'],
    },
    entry_points={
        'console_scripts': [
            'kas=kas.cli.main:cli',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
