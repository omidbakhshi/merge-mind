#!/usr/bin/env python

import asyncio
import argparse
import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_manager import ConfigManager
from src.gitlab_client import GitLabClient
from src.vector_store import QdrantStore, CodeMemoryManager


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
    vector_store_type = config.get_global_setting('vector_store', 'type', 'chromadb') or 'chromadb'
    openai_api_key = config.get_global_setting('openai', 'api_key')

    logger.info(f"Using vector store type: {vector_store_type}")

    if vector_store_type == 'qdrant':
        from src.vector_store import QdrantStore
        vector_store = QdrantStore(
            host=config.get_global_setting('vector_store', 'qdrant', 'host'),
            port=config.get_global_setting('vector_store', 'qdrant', 'port'),
            openai_api_key=openai_api_key
        )
    elif vector_store_type == 'chromadb':
        from src.vector_store import ChromaDBStore
        vector_store = ChromaDBStore(
            path=config.get_global_setting('vector_store', 'path', './storage/vectordb'),
            openai_api_key=openai_api_key
        )
    else:
        raise ValueError(f"Unsupported vector store type: {vector_store_type}")

    # Initialize memory manager
    memory_manager = CodeMemoryManager(vector_store, gitlab_client)

    # Get project configuration
    project_config = config.get_project_config(project_id)
    if not project_config:
        print(f"❌ Project {project_id} not configured")
        return

    print(f"📚 Learning from project: {project_config.name}")
    print(f"🌿 Branch: {branch}")

    # Learn from repository with better error handling
    try:
        stats = await memory_manager.learn_from_repository(
            project_id,
            project_config.name,
            branch,
            included_extensions=project_config.included_extensions
        )

        print(f"\n✅ Learning complete!")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Files skipped: {stats['files_skipped']}")
        print(f"  Total chunks: {stats['total_chunks']}")

        if stats['errors']:
            print(f"\n⚠️ Errors encountered ({len(stats['errors'])}):")
            for error in stats['errors'][:10]:  # Show first 10
                print(f"  - {error}")

    except Exception as e:
        logger.error(f"Failed to learn from repository: {e}", exc_info=True)
        print(f"\n❌ Learning failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Learn from GitLab repositories')
    parser.add_argument('project_id', type=int, help='GitLab project ID')
    parser.add_argument('--branch', default='main', help='Branch to learn from')

    args = parser.parse_args()

    # Run async function
    asyncio.run(learn_project(args.project_id, args.branch))


if __name__ == '__main__':
    main()