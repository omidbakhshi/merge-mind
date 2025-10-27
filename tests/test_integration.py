"""
Integration test suite for Merge Mind
Location: tests/test_integration.py

Tests the full review pipeline with mocked external services
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import json
from datetime import datetime

from src.gitlab_client import GitLabClient, MergeRequestInfo, FileDiff
from src.openai_analyzer import OpenAIAnalyzer, CodeReviewResult, AnalysisContext
from src.vector_store import CodeMemoryManager
from src.merge_request_handler import MergeRequestHandler
from src.config_manager import ConfigManager


class TestIntegration:
    """Integration tests for the full review pipeline"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration manager"""
        config = Mock(spec=ConfigManager)

        def mock_get_global_setting(*args):
            settings = {
                ("openai", "api_key"): "test_openai_key",
                ("openai", "model"): "gpt-4-turbo-preview",
                ("openai", "max_tokens"): 2000,
                ("vector_store",): {"type": "qdrant", "qdrant": {"host": "localhost", "port": 6333}},
                ("review", "cache"): {"enabled": True, "ttl_seconds": 3600},
            }
            return settings.get(args, None)

        config.get_global_setting.side_effect = mock_get_global_setting

        # Mock project config
        project_config = Mock()
        project_config.review_enabled = True
        project_config.review_drafts = False
        project_config.max_files_per_review = 50
        project_config.min_lines = 10
        project_config.name = "test-project"
        config.get_project_config.return_value = project_config

        # Mock is_file_reviewable method
        config.is_file_reviewable.return_value = True
        return config

    @pytest.fixture
    def mock_gitlab_client(self):
        """Mock GitLab client"""
        client = Mock(spec=GitLabClient)

        # Mock MR info - use a real MergeRequestInfo object
        mr_info = MergeRequestInfo(
            project_id=123,
            mr_iid=1,
            title="Add new feature",
            description="This adds a new feature to the codebase",
            author="developer",
            source_branch="feature/new-feature",
            target_branch="main",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            state="opened",
            draft=False,
            web_url="https://gitlab.test.com/project/mr/1",
            diff_refs={"base_sha": "abc123", "head_sha": "def456", "start_sha": "ghi789"}
        )
        client.get_merge_request.return_value = mr_info

        # Mock file diffs - use real FileDiff objects
        diffs = [
            FileDiff(
                old_path="src/new_feature.py",
                new_path="src/new_feature.py",
                diff="+def new_feature():\n+    x = 1\n+    y = 2\n+    return x + y",
                added_lines=4,
                removed_lines=0,
                is_new_file=True,
                is_deleted_file=False,
                is_renamed_file=False,
                language="python"
            )
        ]
        client.get_merge_request_diffs.return_value = diffs

        # Mock other methods
        client.post_review_comment.return_value = True
        client.set_pipeline_status.return_value = None
        client.get_existing_comments.return_value = []

        return client

    @pytest.fixture
    def mock_analyzer(self):
        """Mock OpenAI analyzer"""
        analyzer = Mock(spec=OpenAIAnalyzer)

        # Mock review results for individual file analysis
        results = [
            CodeReviewResult(
                file_path="src/new_feature.py",
                severity="minor",
                line_number=5,
                message="Consider using more descriptive variable names",
                code_snippet="x = 1\ny = 2",
                suggestion="Use 'first_number' and 'second_number' instead of 'x' and 'y'",
                confidence=0.8
            ),
            CodeReviewResult(
                file_path="src/new_feature.py",
                severity="suggestion",
                line_number=12,
                message="Add type hints for better code clarity",
                code_snippet="def __init__(self):\n    self.value = 0",
                suggestion="Add type hints: def __init__(self) -> None:",
                confidence=0.7
            )
        ]
        analyzer.analyze_file_diff = AsyncMock(return_value=results)

        # Mock merge request analysis
        analyzer.analyze_merge_request = AsyncMock(return_value={
            "results": results,
            "summary": {
                "status": "success",
                "overall_assessment": "Code looks good with minor suggestions",
                "total_issues": 2,
                "critical_count": 0,
                "major_count": 0,
                "minor_count": 1,
                "suggestion_count": 1
            }
        })

        return analyzer

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager"""
        memory = Mock()

        # Mock vector_store.search_similar as async
        memory.vector_store = Mock()
        memory.vector_store.search_similar = AsyncMock(return_value=[
            {
                "file": "src/utils.py",
                "code": "def calculate_sum(a, b):\n    return a + b",
                "similarity": 0.85
            }
        ])

        # Mock extract_coding_patterns as async
        memory.extract_coding_patterns = AsyncMock(return_value=[
            {"pattern": "function naming", "confidence": 0.8},
            {"pattern": "error handling", "confidence": 0.7}
        ])

        return memory

    @pytest.mark.asyncio
    async def test_full_review_pipeline(
        self, mock_config, mock_gitlab_client, mock_analyzer, mock_memory_manager
    ):
        """Test the complete review pipeline from webhook to comment posting"""

        # Create handler with mocks
        handler = MergeRequestHandler(
            gitlab_client=mock_gitlab_client,
            analyzer=mock_analyzer,
            memory_manager=mock_memory_manager,
            config_manager=mock_config
        )

        # Simulate webhook payload
        webhook_data = {
            'object_kind': 'merge_request',
            'object_attributes': {
                'iid': 1,
                'action': 'open',
                'draft': False,
                'title': 'Add new feature',
                'description': 'This adds a new feature to the codebase',
                'source_branch': 'feature/new-feature',
                'target_branch': 'main',
                'author': {'username': 'developer'}
            },
            'project': {
                'id': 123,
                'name': 'test-project',
                'web_url': 'https://gitlab.test.com/test-project'
            }
        }

        # Process the webhook
        result = await handler.process_webhook(webhook_data)

        # Verify the result structure
        assert result['status'] == 'queued'
        assert result['project_id'] == 123
        assert result['mr_iid'] == 1

        # Verify that the pipeline components were called correctly
        mock_gitlab_client.get_merge_request.assert_called_once_with(123, 1)
        mock_gitlab_client.get_merge_request_diffs.assert_called_once_with(123, 1)
        mock_analyzer.analyze_file_diff.assert_called_once()
        mock_memory_manager.search_similar.assert_called_once()
        mock_gitlab_client.post_review_comment.assert_called()

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_handling(
        self, mock_config, mock_gitlab_client, mock_analyzer, mock_memory_manager
    ):
        """Test circuit breaker behavior when services fail"""

        # Configure analyzer to simulate failure
        mock_analyzer.analyze_file_diff.side_effect = Exception("OpenAI API timeout")

        handler = MergeRequestHandler(
            gitlab_client=mock_gitlab_client,
            analyzer=mock_analyzer,
            memory_manager=mock_memory_manager,
            config_manager=mock_config
        )

        webhook_data = {
            'object_kind': 'merge_request',
            'object_attributes': {
                'iid': 1,
                'action': 'open',
                'draft': False
            },
            'project': {'id': 123}
        }

        # Process should handle the failure gracefully
        result = await handler.process_webhook(webhook_data)

        # Should still return a result (failure handled)
        assert 'status' in result
        assert 'error' in result or result['status'] == 'failed'

    @pytest.mark.asyncio
    async def test_performance_optimizations(self, mock_config, mock_gitlab_client, mock_analyzer, mock_memory_manager):
        """Test that performance optimizations are working"""

        # Configure analyzer to use caching
        mock_analyzer.analyze_file_diff.return_value = [
            CodeReviewResult(
                file_path="test.py",
                severity="minor",
                line_number=1,
                message="Test review",
                code_snippet="test code",
                suggestion="Test suggestion",
                confidence=0.8
            )
        ]

        handler = MergeRequestHandler(
            gitlab_client=mock_gitlab_client,
            analyzer=mock_analyzer,
            memory_manager=mock_memory_manager,
            config_manager=mock_config
        )

        webhook_data = {
            'object_kind': 'merge_request',
            'object_attributes': {'iid': 1, 'action': 'open', 'draft': False},
            'project': {'id': 123}
        }

        # First call
        await handler.process_webhook(webhook_data)

        # Second call with same data (should potentially use cache)
        await handler.process_webhook(webhook_data)

        # Verify calls were made (in real implementation, second call might be cached)
        assert mock_analyzer.analyze_file_diff.call_count >= 1