from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hpc-connector",
    version="1.0.0",
    author="DFT Platform Team",
    description="A production-grade HPC cluster connection and job scheduling system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/hpc-connector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Clustering",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "asyncssh>=2.0.0",
        "pyyaml>=5.0",
    ],
    extras_require={
        "aws": ["boto3>=1.20.0"],
        "aliyun": ["aliyun-python-sdk-core>=2.13.0"],
        "tencent": ["qcloud-sdk-python>=3.0.0"],
        "monitoring": ["prometheus-client>=0.12.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=2.12.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hpc-connector=hpc_connector.cli:main",
        ],
    },
)
