import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from src.gitlab_client import GitLabClient, MergeRequestInfo, FileDiff


class TestGitLabClient:
    """Test GitLab client functionality"""

    @pytest.fixture
    def mock_gitlab(self):
        """Create mock GitLab instance"""
        with patch('src.gitlab_client.gitlab') as mock:
            mock_gl = MagicMock()
            mock_gl.auth.return_value = None
            mock_gl.user.username = "test_bot"
            mock.Gitlab.return_value = mock_gl
            yield mock_gl

    @pytest.fixture
    def client(self, mock_gitlab):
        """Create GitLab client instance"""
        return GitLabClient("https://gitlab.test.com", "test_token")

    def test_initialization(self, client):
        """Test client initialization"""
        assert client.url == "https://gitlab.test.com"
        assert client.current_user.username == "test_bot"

    def test_get_merge_request(self, client, mock_gitlab):
        """Test fetching merge request"""
        # Setup mock
        mock_project = MagicMock()
        mock_mr = MagicMock()
        mock_mr.title = "Test MR"
        mock_mr.description = "Test description"
        mock_mr.author = {"username": "developer"}
        mock_mr.source_branch = "feature"
        mock_mr.target_branch = "main"
        mock_mr.created_at = "2024-01-01T00:00:00"
        mock_mr.updated_at = "2024-01-02T00:00:00"
        mock_mr.state = "opened"
        mock_mr.draft = False
        mock_mr.web_url = "https://gitlab.test.com/project/mr/1"
        mock_mr.diff_refs = {
            "base_sha": "abc123",
            "head_sha": "def456",
            "start_sha": "ghi789"
        }

        mock_project.mergerequests.get.return_value = mock_mr
        mock_gitlab.projects.get.return_value = mock_project

        # Test
        mr_info = client.get_merge_request(1, 1)

        assert isinstance(mr_info, MergeRequestInfo)
        assert mr_info.title == "Test MR"
        assert mr_info.author == "developer"
        assert mr_info.source_branch == "feature"

    def test_get_merge_request_diffs(self, client, mock_gitlab):
        """Test fetching MR diffs"""
        # Setup mock
        mock_project = MagicMock()
        mock_mr = MagicMock()

        mock_diff = MagicMock()
        mock_diff.diffs = [
            {
                'old_path': 'test.py',
                'new_path': 'test.py',
                'diff': '+def test():\n+    pass\n',
                'new_file': False,
                'deleted_file': False,
                'renamed_file': False
            }
        ]

        mock_mr.diffs.list.return_value = [mock_diff]
        mock_project.mergerequests.get.return_value = mock_mr
        mock_gitlab.projects.get.return_value = mock_project

        # Test
        diffs = client.get_merge_request_diffs(1, 1)

        assert len(diffs) == 1
        assert isinstance(diffs[0], FileDiff)
        assert diffs[0].new_path == 'test.py'
        assert diffs[0].added_lines == 2
        assert diffs[0].language == 'python'

    def test_post_review_comment(self, client, mock_gitlab):
        """Test posting review comment"""
        # Setup mock
        mock_project = MagicMock()
        mock_mr = MagicMock()
        mock_mr.diff_refs = {
            'base_sha': 'abc123',
            'head_sha': 'def456',
            'start_sha': 'ghi789'
        }
        mock_mr.notes.create.return_value = MagicMock()

        mock_project.mergerequests.get.return_value = mock_mr
        mock_gitlab.projects.get.return_value = mock_project

        # Test general comment
        result = client.post_review_comment(
            1, 1, "Test comment", severity="suggestion"
        )

        assert result is True
        mock_mr.notes.create.assert_called_once()

    def test_language_detection(self, client):
        """Test file language detection"""
        assert client._detect_language("test.py") == "python"
        assert client._detect_language("app.js") == "javascript"
        assert client._detect_language("main.go") == "go"
        assert client._detect_language("unknown.xyz") is None


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test async operations"""

    async def test_webhook_processing(self):
        """Test webhook event processing"""
        from src.merge_request_handler import MergeRequestHandler

        # Create mocks
        mock_gitlab = Mock()
        mock_analyzer = Mock()
        mock_memory = Mock()
        mock_config = Mock()

        handler = MergeRequestHandler(
            mock_gitlab, mock_analyzer, mock_memory, mock_config
        )

        # Test webhook data
        webhook_data = {
            'object_kind': 'merge_request',
            'object_attributes': {
                'iid': 1,
                'action': 'open',
                'draft': False
            },
            'project': {
                'id': 1
            }
        }

        mock_config.get_project_config.return_value = Mock(
            review_enabled=True,
            review_drafts=False
        )

        result = await handler.process_webhook(webhook_data)

        assert result['status'] == 'queued'
        assert result['project_id'] == 1
        assert result['mr_iid'] == 1