#!/usr/bin/env python

import asyncio
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_manager import ConfigManager
from src.vector_store import QdrantStore, CodeMemoryManager


class ProgressBar:
    """Simple CLI progress bar"""

    def __init__(self, total: int, prefix: str = "Progress", suffix: str = "Complete", length: int = 50):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.current = 0
        self.start_time = time.time()

    def update(self, current: Optional[int] = None):
        """Update progress bar"""
        if current is not None:
            self.current = current
        else:
            self.current += 1

        percent = (self.current / self.total) * 100
        filled_length = int(self.length * self.current // self.total)
        bar = "█" * filled_length + "-" * (self.length - filled_length)

        elapsed_time = time.time() - self.start_time
        eta = (elapsed_time / self.current) * (self.total - self.current) if self.current > 0 else 0

        print(f"\r{self.prefix}: |{bar}| {percent:.1f}% {self.suffix} (ETA: {eta:.1f}s)", end="", flush=True)

        if self.current >= self.total:
            total_time = time.time() - self.start_time
            print(f"\r{self.prefix}: |{'█' * self.length}| 100.0% {self.suffix} ({total_time:.1f}s)\n", flush=True)


class LocalCodeLearner:
    """Learn from local codebase instead of GitLab"""

    def __init__(self, vector_store, project_name: str):
        self.vector_store = vector_store
        self.project_name = project_name

    async def learn_from_local_directory(
        self,
        local_path: str,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if exclude_patterns is None:
            exclude_patterns = ['.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build']
        if file_patterns is None:
            file_patterns = []
        """Learn from a local directory

        Args:
            local_path: Path to local directory
            file_patterns: File extensions to include (e.g., ['.py', '.js'])
            exclude_patterns: Paths to exclude
        """
        if not exclude_patterns:
            exclude_patterns = ['.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build']

        stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "total_chunks": 0,
            "errors": [],
        }

        local_path_obj = Path(local_path)
        if not local_path_obj.exists():
            raise ValueError(f"Local path does not exist: {local_path_obj}")

        collection_name = f"gitlab_code_{self.project_name}"
        await self.vector_store.initialize(collection_name)

        print(f"📚 Learning from local directory: {local_path_obj}")

        # Count total files first for progress bar
        total_files = 0
        for root, dirs, files in os.walk(local_path_obj):
            dirs[:] = [d for d in dirs if not any(excl in os.path.join(root, d) for excl in exclude_patterns)]
            for file in files:
                file_path = Path(root) / file
                if file_patterns and not any(file_path.suffix == ext for ext in file_patterns):
                    continue
                if any(excl in str(file_path) for excl in exclude_patterns):
                    continue
                if file_path.stat().st_size > 500000:
                    continue
                total_files += 1

        print(f"Found {total_files} files to process")
        progress_bar = ProgressBar(total_files, prefix="Learning", suffix="files processed")

        # Walk through directory again for processing
        processed_count = 0
        for root, dirs, files in os.walk(local_path_obj):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(excl in os.path.join(root, d) for excl in exclude_patterns)]

            for file in files:
                file_path = Path(root) / file

                # Check file patterns
                if file_patterns and not any(file_path.suffix == ext for ext in file_patterns):
                    stats["files_skipped"] += 1
                    continue

                # Skip excluded files
                if any(excl in str(file_path) for excl in exclude_patterns):
                    stats["files_skipped"] += 1
                    continue

                # Skip large files
                if file_path.stat().st_size > 500000:  # 500KB
                    stats["files_skipped"] += 1
                    continue

                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    if content.strip():
                        # Create chunks from the file
                        chunks = self._create_code_chunks(content, str(file_path.relative_to(local_path)))

                        for chunk_idx, chunk in enumerate(chunks):
                            # Generate a simple integer ID based on file path and chunk index
                            file_hash = hash(str(file_path.relative_to(local_path_obj)))
                            doc_id = abs(file_hash + chunk_idx)  # Ensure positive integer

                            doc = {
                                "id": doc_id,
                                "content": chunk,
                                "metadata": {
                                    "project_name": self.project_name,
                                    "file_path": str(file_path.relative_to(local_path_obj)),
                                    "chunk_index": chunk_idx,
                                    "language": self._detect_language(file_path),
                                    "source": "local",
                                },
                            }

                            # Add document to vector store
                            await self.vector_store.add_documents([doc], collection_name)

                        stats["files_processed"] += 1
                        stats["total_chunks"] += len(chunks)

                        processed_count += 1
                        progress_bar.update(processed_count)

                except Exception as e:
                    error_msg = f"{file_path}: {str(e)}"
                    stats["errors"].append(error_msg)
                    print(f"❌ Error processing {file_path}: {e}")
                    processed_count += 1
                    progress_bar.update(processed_count)

        # Store learning stats
        self.learning_stats = {
            "learned_at": "local_training",
            "stats": stats,
        }

        return stats

    def _create_code_chunks(
        self, content: str, file_path: str, chunk_size: int = 100, overlap: int = 20
    ) -> List[str]:
        """Split code into meaningful chunks for embedding"""
        lines = content.split("\n")
        chunks = []

        # Add file header as context
        header = f"File: {file_path}\n"

        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i : i + chunk_size]
            chunk = header + "\n".join(chunk_lines)
            chunks.append(chunk)

        # If file is small, just use the whole file
        if len(chunks) == 1 and len(lines) < 50:
            chunks = [content]

        return chunks

    def _detect_language(self, file_path: Path) -> str:
        """Detect language from file extension"""
        ext = file_path.suffix.lower()

        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
            ".sql": "sql",
            ".html": "html",
            ".css": "css",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".md": "markdown",
        }

        return language_map.get(ext, "unknown")


async def learn_local_project(project_name: str, local_path: str, file_extensions: Optional[List[str]] = None):
    """Learn from a local project directory"""

    # Load configuration
    config = ConfigManager()

    # Initialize vector store (Qdrant)
    vector_config = config.get_global_setting("vector_store")
    openai_key = config.get_global_setting("openai", "api_key")

    if vector_config.get("type") == "qdrant":
        qdrant_config = vector_config.get("qdrant", {})
        vector_store = QdrantStore(
            host=qdrant_config.get("host", "qdrant"),
            port=qdrant_config.get("port", 6333),
            openai_api_key=openai_key,
        )
    else:
        raise ValueError("Only Qdrant vector store is supported for local learning")

    # Initialize local learner
    learner = LocalCodeLearner(vector_store, project_name)

    print(f"🚀 Starting local learning for project: {project_name}")
    print(f"📁 Local path: {local_path}")

    # Learn from local directory
    stats = await learner.learn_from_local_directory(
        local_path,
        file_patterns=file_extensions,
        exclude_patterns=['.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build', '*.pyc']
    )

    print(f"\n✅ Local learning complete!")
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Files skipped: {stats['files_skipped']}")
    print(f"  Total chunks: {stats['total_chunks']}")

    if stats['errors']:
        print(f"\n⚠️ Errors encountered:")
        for error in stats['errors'][:5]:
            print(f"  - {error}")


def main():
    parser = argparse.ArgumentParser(description='Learn from local codebase')
    parser.add_argument('project_name', help='Name for the project/collection')
    parser.add_argument('local_path', help='Path to local codebase directory')
    parser.add_argument('--extensions', nargs='+', help='File extensions to include (e.g., .py .js .ts)')

    args = parser.parse_args()

    # Default extensions if not specified
    extensions = args.extensions if args.extensions is not None else ['.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c', '.cs', '.rb', '.php']

    # Run async function
    asyncio.run(learn_local_project(args.project_name, args.local_path, extensions))


if __name__ == '__main__':
    main()