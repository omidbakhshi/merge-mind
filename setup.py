# setup.py
"""
Setup configuration for Merge Mind
Location: setup.py (root directory)
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="merge-mind",
    version="1.0.0",
    author="Your Team",
    author_email="ai-reviewer@yourcompany.com",
    description="AI-powered code review assistant for GitLab - Merge Mind",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/merge-mind",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Quality Assurance",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "python-gitlab>=4.4.0",
        "openai>=0.28.0",
        "tiktoken>=0.5.2",
        "chromadb>=0.4.22",
        "pydantic>=2.5.3",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "aiofiles>=23.2.1",
        "httpx>=0.26.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.4",
            "pytest-asyncio>=0.23.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
        "qdrant": ["qdrant-client>=1.7.0"],
        "pinecone": ["pinecone-client>=3.0.0"],
    },
    entry_points={
        "console_scripts": [
            "gitlab-reviewer=src.main:main",
            "learn-codebase=scripts.learn_codebase:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
)