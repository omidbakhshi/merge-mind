"""
GitLab API client for interacting with self-hosted GitLab instances
Handles MR fetching, commenting, and pipeline integration
Location: src/gitlab_client.py
"""

import gitlab
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from src.circuit_breaker import get_gitlab_circuit_breaker, CircuitBreakerOpenException

logger = logging.getLogger(__name__)


@dataclass
class MergeRequestInfo:
    """Structured information about a merge request"""

    project_id: int
    mr_iid: int
    title: str
    description: str
    author: str
    source_branch: str
    target_branch: str
    created_at: datetime
    updated_at: datetime
    state: str
    draft: bool
    web_url: str
    diff_refs: Dict[str, str]


@dataclass
class FileDiff:
    """Represents a single file diff in a merge request"""

    old_path: str
    new_path: str
    diff: str
    added_lines: int
    removed_lines: int
    is_new_file: bool
    is_deleted_file: bool
    is_renamed_file: bool
    language: Optional[str] = None

    @property
    def total_changes(self) -> int:
        return self.added_lines + self.removed_lines


class GitLabClient:
    """Client for interacting with GitLab API"""

    # Comment signature to identify our bot comments
    BOT_SIGNATURE = "\n\n---\n*🤖 AI Code Review Bot*"
    REVIEW_THREAD_PREFIX = "ai-review"

    def __init__(self, url: str, private_token: str):
        """Initialize GitLab client

        Args:
            url: GitLab instance URL
            private_token: GitLab personal access token
        """
        self.url = url
        self.gl = gitlab.Gitlab(url, private_token=private_token)
        try:
            self.gl.auth()
            self.current_user = self.gl.user
            logger.info(f"Successfully authenticated as {self.current_user.username}")  # type: ignore[union-attr]
        except Exception as e:
            logger.error(f"Failed to authenticate with GitLab: {e}")
            raise

        # Type assertion since we raise exception above if auth fails
        assert self.current_user is not None

    def get_project(self, project_id: int):
        """Get a GitLab project object"""
        try:
            return self.gl.projects.get(project_id)
        except gitlab.exceptions.GitlabGetError as e:
            logger.error(f"Failed to get project {project_id}: {e}")
            raise

    def get_merge_request(self, project_id: int, mr_iid: int) -> MergeRequestInfo:
        """Get detailed information about a merge request"""
        circuit_breaker = get_gitlab_circuit_breaker()

        try:
            project = self.get_project(project_id)
            mr = project.mergerequests.get(mr_iid)

            return MergeRequestInfo(
                project_id=project_id,
                mr_iid=mr_iid,
                title=mr.title,
                description=mr.description or "",
                author=mr.author["username"],
                source_branch=mr.source_branch,
                target_branch=mr.target_branch,
                created_at=datetime.fromisoformat(mr.created_at),
                updated_at=datetime.fromisoformat(mr.updated_at),
                state=mr.state,
                draft=mr.draft,
                web_url=mr.web_url,
                diff_refs={
                    "base_sha": mr.diff_refs["base_sha"],
                    "head_sha": mr.diff_refs["head_sha"],
                    "start_sha": mr.diff_refs["start_sha"],
                },
            )

        except CircuitBreakerOpenException as e:
            logger.error(f"GitLab circuit breaker is OPEN: {e}")
            raise Exception("GitLab service is temporarily unavailable. Please try again later.")
        except Exception as e:
            logger.error(f"Failed to get merge request {project_id}/{mr_iid}: {e}")
            raise

    def get_merge_request_diffs(self, project_id: int, mr_iid: int) -> List[FileDiff]:
        """Get all file diffs for a merge request"""
        project = self.get_project(project_id)
        mr = project.mergerequests.get(mr_iid)
        diffs = mr.diffs.list(get_all=True)

        if not diffs:
            logger.warning(f"No diffs found for MR {mr_iid}")
            return []

        # Get the latest diff version
        latest_diff = diffs[0]
        file_diffs = []

        for file_diff in latest_diff.diffs:
            # Parse the diff to count added/removed lines
            added_lines = 0
            removed_lines = 0

            if file_diff.get("diff"):
                for line in file_diff["diff"].split("\n"):
                    if line.startswith("+") and not line.startswith("+++"):
                        added_lines += 1
                    elif line.startswith("-") and not line.startswith("---"):
                        removed_lines += 1

            # Detect file language from extension
            language = self._detect_language(
                file_diff.get("new_path", file_diff.get("old_path", ""))
            )

            file_diffs.append(
                FileDiff(
                    old_path=file_diff.get("old_path", ""),
                    new_path=file_diff.get("new_path", ""),
                    diff=file_diff.get("diff", ""),
                    added_lines=added_lines,
                    removed_lines=removed_lines,
                    is_new_file=file_diff.get("new_file", False),
                    is_deleted_file=file_diff.get("deleted_file", False),
                    is_renamed_file=file_diff.get("renamed_file", False),
                    language=language,
                )
            )

        return file_diffs

    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension"""
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
            ".sql": "sql",
            ".sh": "bash",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
        }

        import os

        _, ext = os.path.splitext(file_path)
        return language_map.get(ext.lower())

    def post_review_comment(
        self,
        project_id: int,
        mr_iid: int,
        comment: str,
        file_path: Optional[str] = None,
        line: Optional[int] = None,
        severity: str = "suggestion",
    ) -> bool:
        """Post a review comment on a merge request

        Args:
            project_id: GitLab project ID
            mr_iid: Merge request IID
            comment: The comment text
            file_path: Optional file path for inline comments
            line: Optional line number for inline comments
            severity: Comment severity level
        """
        try:
            project = self.get_project(project_id)
            mr = project.mergerequests.get(mr_iid)

            # Add bot signature and severity tag
            formatted_comment = (
                f"**[{severity.upper()}]** {comment}{self.BOT_SIGNATURE}"
            )

            if file_path and line:
                # Create an inline comment (discussion)
                position = {
                    "base_sha": mr.diff_refs["base_sha"],
                    "head_sha": mr.diff_refs["head_sha"],
                    "start_sha": mr.diff_refs["start_sha"],
                    "new_path": file_path,
                    "new_line": line,
                    "position_type": "text",
                }

                _discussion = mr.discussions.create(
                    {"body": formatted_comment, "position": position}
                )
                logger.info(f"Created inline comment on {file_path}:{line}")
            else:
                # Create a general comment
                _note = mr.notes.create({"body": formatted_comment})
                logger.info(f"Created general comment on MR {mr_iid}")

            return True

        except Exception as e:
            logger.error(f"Failed to post comment: {e}")
            return False

    def post_review_summary(
        self, project_id: int, mr_iid: int, summary: Dict[str, Any]
    ) -> bool:
        """Post a comprehensive review summary

        Args:
            project_id: GitLab project ID
            mr_iid: Merge request IID
            summary: Dictionary containing review summary with stats and findings
        """
        try:
            # Format the summary comment
            comment = self._format_review_summary(summary)

            # Check if we already posted a summary and update it
            existing_comment_id = self._find_existing_summary(project_id, mr_iid)

            project = self.get_project(project_id)
            mr = project.mergerequests.get(mr_iid)

            if existing_comment_id:
                # Update existing comment
                note = mr.notes.get(existing_comment_id)
                note.body = comment
                note.save()
                logger.info(f"Updated review summary for MR {mr_iid}")
            else:
                # Create new comment
                mr.notes.create({"body": comment})
                logger.info(f"Posted review summary for MR {mr_iid}")

            return True

        except Exception as e:
            logger.error(f"Failed to post review summary: {e}")
            return False

    def _format_review_summary(self, summary: Dict[str, Any]) -> str:
        """Format review summary for posting"""
        comment = "## 🔍 AI Code Review Summary\n\n"

        # Add statistics
        stats = summary.get("statistics", {})
        comment += f"**Files Reviewed:** {stats.get('files_reviewed', 0)}\n"
        comment += f"**Total Changes:** +{stats.get('lines_added', 0)} / -{stats.get('lines_removed', 0)}\n\n"

        # Add findings by severity
        findings = summary.get("findings", {})

        if findings.get("critical"):
            comment += "### 🔴 Critical Issues\n"
            for finding in findings["critical"]:
                comment += f"- {finding}\n"
            comment += "\n"

        if findings.get("major"):
            comment += "### 🟡 Major Issues\n"
            for finding in findings["major"]:
                comment += f"- {finding}\n"
            comment += "\n"

        if findings.get("minor"):
            comment += "### 🟢 Minor Issues\n"
            for finding in findings["minor"]:
                comment += f"- {finding}\n"
            comment += "\n"

        if findings.get("suggestions"):
            comment += "### 💡 Suggestions\n"
            for finding in findings["suggestions"]:
                comment += f"- {finding}\n"
            comment += "\n"

        # Add overall assessment
        if summary.get("overall_assessment"):
            comment += f"### Overall Assessment\n{summary['overall_assessment']}\n\n"

        # Add review status
        status = summary.get("status", "completed")
        if status == "approved":
            comment += "✅ **This merge request looks good to merge!**\n"
        elif status == "needs_work":
            comment += "⚠️ **This merge request needs attention before merging.**\n"

        comment += self.BOT_SIGNATURE
        return comment

    def _find_existing_summary(self, project_id: int, mr_iid: int) -> Optional[int]:
        """Find existing review summary comment"""
        try:
            project = self.get_project(project_id)
            mr = project.mergerequests.get(mr_iid)
            notes = mr.notes.list(get_all=True)

            for note in notes:
                if (
                    note.author["username"] == self.current_user.username  # type: ignore[union-attr]
                    and "## 🔍 AI Code Review Summary" in note.body
                ):
                    return note.id

            return None

        except Exception as e:
            logger.error(f"Error finding existing summary: {e}")
            return None

    def get_existing_comments(
        self, project_id: int, mr_iid: int
    ) -> List[Dict[str, Any]]:
        """Get all existing review comments from the bot"""
        try:
            project = self.get_project(project_id)
            mr = project.mergerequests.get(mr_iid)

            bot_comments = []

            # Get all discussions
            discussions = mr.discussions.list(get_all=True)
            for discussion in discussions:
                for note in discussion.notes:
                    if                     note["author"][
                        "username"
                    ] == self.current_user.username and self.BOT_SIGNATURE in note.get(
                        "body", ""
                    ):  # type: ignore[union-attr]
                        bot_comments.append(
                            {
                                "id": note["id"],
                                "body": note["body"],
                                "created_at": note["created_at"],
                                "position": discussion.position
                                if hasattr(discussion, "position")
                                else None,
                            }
                        )

            return bot_comments

        except Exception as e:
            logger.error(f"Error getting existing comments: {e}")
            return []

    def set_pipeline_status(
        self,
        project_id: int,
        sha: str,
        state: str,
        name: str = "AI Code Review",
        description: Optional[str] = None,
    ) -> bool:
        """Update pipeline status for a commit

        Args:
            project_id: GitLab project ID
            sha: Commit SHA
            state: Pipeline state (pending, running, success, failed, canceled)
            name: Status name
            description: Optional description
        """
        try:
            project = self.get_project(project_id)

            status_data = {
                "state": state,
                "name": name,
                "target_url": f"{self.url}/ai-review/{project_id}/{sha}",
            }

            if description:
                status_data["description"] = description

            project.commits.get(sha).statuses.create(status_data)
            logger.info(f"Set pipeline status to {state} for {sha}")
            return True

        except Exception as e:
            logger.error(f"Failed to set pipeline status: {e}")
            return False

    async def get_file_content_async(
        self, project_id: int, file_path: str, ref: str = "main"
    ) -> Optional[str]:
        """Get content of a file from the repository asynchronously

        Used for getting context files or learning from the codebase
        """
        circuit_breaker = get_gitlab_circuit_breaker()

        try:
            # Use circuit breaker for async operation
            content = await circuit_breaker.call(
                self._get_file_content_direct,
                project_id,
                file_path,
                ref
            )
            return content

        except CircuitBreakerOpenException as e:
            logger.error(f"GitLab circuit breaker is OPEN for file content: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get file content for {file_path}: {e}")
            return None

    def _get_file_content_direct(
        self, project_id: int, file_path: str, ref: str = "main"
    ) -> Optional[str]:
        """Direct synchronous file content retrieval"""
        try:
            project = self.get_project(project_id)
            file = project.files.get(file_path=file_path, ref=ref)

            # Get raw content
            content = file.content if hasattr(file, 'content') else file

            # Decode if bytes
            if isinstance(content, bytes):
                try:
                    return content.decode('utf-8')
                except UnicodeDecodeError:
                    logger.debug(f"Skipping binary file: {file_path}")
                    return None
            else:
                return str(content)

        except Exception as e:
            logger.error(f"Failed to get file content for {file_path}: {e}")
            return None

    # Keep the synchronous version for backward compatibility
    def get_file_content(
        self, project_id: int, file_path: str, ref: str = "main"
    ) -> Optional[str]:
        """Get content of a file from the repository (synchronous)

        Used for getting context files or learning from the codebase
        """
        import asyncio
        try:
            # Run async version in new event loop if needed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.get_file_content_async(project_id, file_path, ref))
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Failed to get file content synchronously for {file_path}: {e}")
            return None
