# src/vector_store.py
"""
Vector store implementation for code embeddings and similarity search
Supports multiple backends: ChromaDB, Qdrant, Pinecone
Location: src/vector_store.py
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional, cast, Union, TypedDict
from abc import ABC, abstractmethod
from datetime import datetime

# Vector store backend imports (install as needed)

class LearningStats(TypedDict):
    files_processed: int
    files_skipped: int
    total_chunks: int
    errors: List[str]

try:
    from openai import OpenAI
    import openai
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Circuit breaker import
try:
    from src.circuit_breaker import get_qdrant_circuit_breaker, CircuitBreakerOpenException
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False


class VectorStoreBase(ABC):
    """Abstract base class for vector stores"""

    @abstractmethod
    async def initialize(self, collection_name: str):
        """Initialize a collection"""
        pass

    @abstractmethod
    async def add_documents(self, documents: List[Dict], collection_name: str):
        """Add documents to the store"""
        pass

    @abstractmethod
    async def search_similar(
        self, query: str, collection_name: str, limit: int = 5
    ) -> List[Dict]:
        """Search for similar documents"""
        pass

    @abstractmethod
    async def update_document(self, doc_id: str, document: Dict, collection_name: str):
        """Update a document"""
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str):
        """Delete a collection"""
        pass


class QdrantStore(VectorStoreBase):
    """Qdrant implementation for vector storage with performance optimizations"""

    def __init__(self, host: str = "localhost", port: int = 6333, openai_api_key: Optional[str] = None,
                 connection_pool_size: int = 10, max_batch_size: int = 100):
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client is not installed. Run: pip install qdrant-client")

        self.host = host
        self.port = port
        self.openai_api_key = openai_api_key
        self.connection_pool_size = connection_pool_size
        self.max_batch_size = max_batch_size

        # Initialize Qdrant client with connection pooling
        self.client = QdrantClient(
            host=host,
            port=port,
            prefer_grpc=True,  # Use gRPC for better performance
            timeout=30.0
        )

        # Set up OpenAI for embeddings
        self.openai_client = None
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)

        # Embedding cache for performance
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_max_size = 1000

        logger.info(f"Initialized Qdrant store at {host}:{port} with pool_size={connection_pool_size}, max_batch={max_batch_size}")

    async def initialize(self, collection_name: str):
        """Initialize or get a collection"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if collection_name not in collection_names:
                # Create collection with vector parameters
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                logger.info(f"Created collection: {collection_name}")
            else:
                logger.info(f"Collection {collection_name} already exists")

        except Exception as e:
            logger.error(f"Failed to initialize collection {collection_name}: {e}")
            raise

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI with caching"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        # Check approximate token count (rough estimate: 1 token ≈ 4 characters)
        approx_tokens = len(text) // 4
        if approx_tokens > 8000:  # Leave some buffer below 8192 limit
            logger.warning(f"Text too long for embedding ({approx_tokens} tokens), skipping")
            raise ValueError(f"Text exceeds token limit: {approx_tokens} tokens")

        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash in self._embedding_cache:
            logger.debug("Using cached embedding")
            return self._embedding_cache[text_hash]

        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            embedding = response['data'][0]['embedding']

            # Cache the embedding
            if len(self._embedding_cache) < self._cache_max_size:
                self._embedding_cache[text_hash] = embedding
            elif len(self._embedding_cache) >= self._cache_max_size:
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
                self._embedding_cache[text_hash] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def add_documents(self, documents: List[Dict], collection_name: str):
        """Add code documents to the store with optimized batching

        Args:
            documents: List of dicts with keys: id, content, metadata
            collection_name: Name of the collection
        """
        if CIRCUIT_BREAKER_AVAILABLE:
            circuit_breaker = get_qdrant_circuit_breaker()
            try:
                await circuit_breaker.call(self._add_documents_direct, documents, collection_name)
            except CircuitBreakerOpenException as e:
                logger.error(f"Qdrant circuit breaker is OPEN: {e}")
                raise Exception("Vector store is temporarily unavailable. Please try again later.")
        else:
            await self._add_documents_direct(documents, collection_name)

    async def _add_documents_direct(self, documents: List[Dict], collection_name: str):
        """Direct document addition with optimized batching"""
        await self.initialize(collection_name)

        # Process documents in optimized batches
        total_processed = 0
        for i in range(0, len(documents), self.max_batch_size):
            batch = documents[i:i + self.max_batch_size]
            points = []

            logger.debug(f"Processing batch {i//self.max_batch_size + 1} with {len(batch)} documents")

            for doc in batch:
                doc_id = doc.get("id") or hashlib.md5(doc["content"].encode()).hexdigest()

                try:
                    # Generate embedding (with caching)
                    embedding = self._generate_embedding(doc["content"])
                except Exception as e:
                    logger.warning(f"Skipping document due to embedding failure: {e}")
                    continue

                # Prepare metadata
                metadata = doc.get("metadata", {})
                metadata["added_at"] = datetime.now().isoformat()
                metadata["content_hash"] = hashlib.md5(doc["content"].encode()).hexdigest()

                point = PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        "content": doc["content"],
                        **metadata
                    }
                )
                points.append(point)

            # Skip if no points (all embeddings failed)
            if not points:
                logger.debug(f"Skipping empty batch {i//self.max_batch_size + 1}")
                continue

            # Add batch to Qdrant
            try:
                self.client.upsert(collection_name=collection_name, points=points)
                total_processed += len(batch)
                logger.debug(f"Added batch of {len(batch)} documents to {collection_name}")
            except Exception as e:
                logger.error(f"Failed to add batch to {collection_name}: {e}")
                raise

        logger.info(f"Successfully added {total_processed} documents to {collection_name} in {len(documents)//self.max_batch_size + 1} batches")

    async def search_similar(
        self, query: str, collection_name: str, limit: int = 5
    ) -> List[Dict]:
        """Search for similar code snippets

        Args:
            query: Code snippet or description to search for
            collection_name: Collection to search in
            limit: Maximum number of results
        """
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)

            # Search
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit
            )

            # Format results
            formatted_results = []
            for hit in search_result:
                if hit.payload is None:
                    continue
                formatted_results.append(
                    {
                        "id": hit.id,
                        "code": hit.payload.get("content", ""),
                        "metadata": {k: v for k, v in hit.payload.items() if k != "content"},
                        "distance": hit.score,
                        "file": hit.payload.get("file_path", "unknown"),
                    }
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def update_document(self, doc_id: str, document: Dict, collection_name: str):
        """Update a document in the store"""
        await self.initialize(collection_name)

        try:
            # Generate embedding for new content
            embedding = self._generate_embedding(document["content"])

            # Prepare metadata
            metadata = document.get("metadata", {})
            metadata["updated_at"] = datetime.now().isoformat()
            metadata["content_hash"] = hashlib.md5(document["content"].encode()).hexdigest()

            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "content": document["content"],
                    **metadata
                }
            )

            # Upsert (update if exists, insert if not)
            self.client.upsert(collection_name=collection_name, points=[point])

            logger.info(f"Updated document {doc_id} in {collection_name}")
        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            raise

    async def delete_collection(self, collection_name: str):
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about a collection"""
        try:
            collection_info = self.client.get_collection(collection_name)
            count = self.client.count(collection_name).count

            return {
                "name": collection_name,
                "document_count": count,
                "metadata": collection_info.config.params,
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

from pathlib import Path

logger = logging.getLogger(__name__)


class CodeMemoryManager:
    """High-level manager for code learning and memory"""

    def __init__(self, vector_store: VectorStoreBase, gitlab_client=None):
        """Initialize the memory manager

        Args:
            vector_store: Vector store backend
            gitlab_client: GitLab client for fetching code
        """
        self.vector_store = vector_store
        self.gitlab_client = gitlab_client
        self.learning_stats: Dict[str, Dict[str, Any]] = {}

    async def learn_from_repository(
        self,
        project_id: int,
        project_name: str,
        branch: str = "main",
        included_extensions: Optional[List[str]] = None,
        excluded_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Learn from an entire repository

        Args:
            project_id: GitLab project ID
            project_name: Project name for collection naming
            branch: Branch to learn from
            included_extensions: File extensions to include (e.g., ['.py', '.js'])
            excluded_paths: Path patterns to exclude (e.g., ['vendor/', 'node_modules/'])
        """
        if not self.gitlab_client:
            logger.error("GitLab client required for repository learning")
            raise ValueError("GitLab client required for repository learning")

        collection_name = f"gitlab_code_{project_name}"
        start_time = datetime.now()

        logger.info(f"Starting learning from {project_name} (project_id: {project_id}, branch: {branch})")
        logger.info(f"Included extensions: {included_extensions}")
        logger.info(f"Excluded paths: {excluded_paths}")

        stats: Dict[str, Any] = {
            "files_processed": 0,
            "files_skipped": 0,
            "total_chunks": 0,
            "errors": [],
        }

        try:
            # Get project
            logger.debug(f"Fetching project {project_id} from GitLab")
            project = self.gitlab_client.get_project(project_id)

            # Get repository tree
            logger.debug(f"Fetching repository tree for branch {branch}")
            items = project.repository_tree(ref=branch, recursive=True, iterator=True)

            documents = []
            processed_count = 0

            for item in items:
                if item["type"] != "blob":
                    continue

                file_path = item["path"]
                processed_count += 1

                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} items, {stats['files_processed']} files learned")

                # Skip large files
                if item.get("size", 0) > 500000:  # 500KB
                    logger.debug(f"Skipping large file: {file_path} ({item.get('size', 0)} bytes)")
                    stats["files_skipped"] += 1
                    continue

                # Check excluded paths
                if excluded_paths:
                    if any(excluded_path in file_path for excluded_path in excluded_paths):
                        logger.debug(f"Skipping excluded path: {file_path}")
                        stats["files_skipped"] += 1
                        continue

                # Check included extensions
                if included_extensions:
                    if not any(file_path.endswith(ext) for ext in included_extensions):
                        logger.debug(f"Skipping non-matching extension: {file_path}")
                        stats["files_skipped"] += 1
                        continue

                try:
                    # Get file content
                    content = await self.gitlab_client.get_file_content_async(
                        project_id, file_path, branch
                    )

                    if content:
                        # Create chunks from the file
                        chunks = self._create_code_chunks(content, file_path)

                        for chunk_idx, chunk in enumerate(chunks):
                            # Generate a unique ID using hash of file path and chunk index
                            doc_id = hashlib.md5(f"{project_id}_{file_path}_{chunk_idx}".encode()).hexdigest()
                            doc = {
                                "id": doc_id,
                                "content": chunk,
                                "metadata": {
                                    "project_id": project_id,
                                    "project_name": project_name,
                                    "file_path": file_path,
                                    "branch": branch,
                                    "chunk_index": chunk_idx,
                                    "language": self._detect_language(file_path),
                                },
                            }
                            documents.append(doc)

                        stats["files_processed"] += 1
                        stats["total_chunks"] += len(chunks)

                        logger.debug(f"Processed {file_path}: {len(chunks)} chunks")

                        # Add in batches
                        if len(documents) >= 10:
                            logger.debug(f"Adding batch of {len(documents)} documents to vector store")
                            await self.vector_store.add_documents(
                                documents, collection_name
                            )
                            documents = []

                except Exception as e:
                    error_msg = f"{file_path}: {str(e)}"
                    logger.error(f"Failed to process {file_path}: {e}")
                    stats["errors"].append(error_msg)

            # Add remaining documents
            if documents:
                logger.debug(f"Adding final batch of {len(documents)} documents to vector store")
                await self.vector_store.add_documents(documents, collection_name)

            # Store learning stats
            duration = (datetime.now() - start_time).total_seconds()
            self.learning_stats[project_name] = {
                "learned_at": datetime.now().isoformat(),
                "stats": stats,
                "duration_seconds": duration,
            }

            logger.info(f"Completed learning from {project_name} in {duration:.2f}s: {stats}")
            return cast(Dict[str, Any], stats)

        except Exception as e:
            logger.error(f"Failed to learn from repository {project_name}: {e}", exc_info=True)
            raise

    def _create_code_chunks(
        self, content: str, file_path: str, chunk_size: int = 50, overlap: int = 10
    ) -> List[str]:
        """Split code into meaningful chunks for embedding

        Args:
            content: File content
            file_path: Path to the file
            chunk_size: Number of lines per chunk
            overlap: Number of overlapping lines between chunks
        """
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

    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension"""
        import os

        ext = os.path.splitext(file_path)[1].lower()

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
        }

        return language_map.get(ext, "unknown")

    async def update_from_merged_code(
        self, project_id: int, project_name: str, file_changes: List[Dict]
    ) -> bool:
        """Update knowledge base with newly merged code

        Args:
            project_id: GitLab project ID
            project_name: Project name
            file_changes: List of file changes from merged MR
        """
        collection_name = f"gitlab_code_{project_name}"

        try:
            for change in file_changes:
                file_path = change["file_path"]
                content = change.get("new_content")

                if not content:
                    continue

                # Create chunks
                chunks = self._create_code_chunks(content, file_path)

                # Update documents
                for chunk_idx, chunk in enumerate(chunks):
                    doc_id = hashlib.md5(f"{project_id}_{file_path}_{chunk_idx}".encode()).hexdigest()
                    doc = {
                        "content": chunk,
                        "metadata": {
                            "project_id": project_id,
                            "project_name": project_name,
                            "file_path": file_path,
                            "updated_at": datetime.now().isoformat(),
                            "chunk_index": chunk_idx,
                        },
                    }

                    await self.vector_store.update_document(
                        doc_id, doc, collection_name
                    )

            logger.info(
                f"Updated knowledge base for {project_name} with {len(file_changes)} files"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to update knowledge base: {e}")
            return False

    async def extract_coding_patterns(
        self, project_name: str, limit: int = 100
    ) -> List[str]:
        """Extract common coding patterns from a project

        Returns list of pattern descriptions
        """
        # This would use more sophisticated pattern detection
        # For now, return placeholder patterns
        patterns = [
            "Use async/await for all database operations",
            "Follow PEP 8 naming conventions",
            "Add type hints to all function signatures",
            "Include docstrings for public methods",
            "Use dependency injection for service classes",
        ]

        return patterns
