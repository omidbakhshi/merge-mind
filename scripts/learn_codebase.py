#!/usr/bin/env python

import asyncio
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_manager import ConfigManager
from src.gitlab_client import GitLabClient
from src.vector_store import ChromaDBStore, CodeMemoryManager


async def learn_project(project_id: int, branch: str = 'main'):
    """Learn from a specific project"""

    # Load configuration
    config = ConfigManager()

    # Initialize GitLab client
    gitlab_client = GitLabClient(
        config.get_global_setting('gitlab', 'url'),
        config.get_global_setting('gitlab', 'token')
    )

    # Initialize vector store
    vector_store = ChromaDBStore(
        path=config.get_global_setting('vector_store', 'path'),
        openai_api_key=config.get_global_setting('openai', 'api_key')
    )

    # Initialize memory manager
    memory_manager = CodeMemoryManager(vector_store, gitlab_client)

    # Get project configuration
    project_config = config.get_project_config(project_id)
    if not project_config:
        print(f"❌ Project {project_id} not configured")
        return

    print(f"📚 Learning from project: {project_config.name}")
    print(f"🌿 Branch: {branch}")

    # Learn from repository
    stats = await memory_manager.learn_from_repository(
        project_id,
        project_config.name,
        branch,
        file_patterns=project_config.included_extensions
    )

    print(f"\n✅ Learning complete!")
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Files skipped: {stats['files_skipped']}")
    print(f"  Total chunks: {stats['total_chunks']}")

    if stats['errors']:
        print(f"\n⚠️ Errors encountered:")
        for error in stats['errors'][:5]:
            print(f"  - {error}")


def main():
    parser = argparse.ArgumentParser(description='Learn from GitLab repositories')
    parser.add_argument('project_id', type=int, help='GitLab project ID')
    parser.add_argument('--branch', default='main', help='Branch to learn from')

    args = parser.parse_args()

    # Run async function
    asyncio.run(learn_project(args.project_id, args.branch))


if __name__ == '__main__':
    main()