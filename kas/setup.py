"""
KAS - Klaw Agent Studio
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kimi-agent-studio",
    version="0.1.0",
    author="KAS Team",
    description="专业开发者的 CLI-first Agent 孵化平台",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kas-team/kas",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        'llm': ['openai>=1.0.0'],
        'dev': ['pytest>=7.0.0', 'black>=23.0.0'],
    },
    entry_points={
        'console_scripts': [
            'kas=kas.cli.main:main',
        ],
    },
)
