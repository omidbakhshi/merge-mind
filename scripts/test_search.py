#!/usr/bin/env python

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_manager import ConfigManager
from src.vector_store import QdrantStore


async def test_search():
    """Test searching the vector store"""

    # Load configuration
    config = ConfigManager()

    # Initialize vector store
    vector_config = config.get_global_setting("vector_store")
    openai_key = config.get_global_setting("openai", "api_key")

    qdrant_config = vector_config.get("qdrant", {})
    vector_store = QdrantStore(
        host=qdrant_config.get("host", "qdrant"),
        port=qdrant_config.get("port", 6333),
        openai_api_key=openai_key,
    )

    collection_name = "gitlab_code_my_test_project"

    # Test search
    results = await vector_store.search_similar("hello world function", collection_name, limit=5)

    print(f"Search results for 'hello world function':")
    for result in results:
        print(f"  - {result['file']}: {result['code'][:100]}...")

    # Get collection stats
    if hasattr(vector_store, 'get_collection_stats'):
        stats = vector_store.get_collection_stats(collection_name)
        print(f"\nCollection stats: {stats}")


if __name__ == '__main__':
    asyncio.run(test_search())