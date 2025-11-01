import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from src.gitlab_client import GitLabClient, MergeRequestInfo
from src.openai_analyzer import OpenAIAnalyzer, AnalysisContext, CodeReviewResult
from src.vector_store import CodeMemoryManager
from src.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class MergeRequestHandler:
    """Orchestrates the review process for merge requests"""

    def __init__(
        self,
        gitlab_client: GitLabClient,
        analyzer: OpenAIAnalyzer,
        memory_manager: CodeMemoryManager,
        config_manager: ConfigManager,
    ):
        self.gitlab = gitlab_client
        self.analyzer = analyzer
        self.memory = memory_manager
        self.config = config_manager

        # Track processing state
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.active_reviews: Dict[str, bool] = {}

    async def process_webhook(self, webhook_data: Dict[str, Any], wait_for_completion: bool = False) -> Dict[str, Any]:
        """Process GitLab webhook event

        Args:
            webhook_data: Webhook payload from GitLab
        """
        event_type = webhook_data.get("object_kind")

        if event_type != "merge_request":
            logger.info(f"Ignoring non-MR event: {event_type}")
            return {"status": "ignored", "reason": "not_merge_request"}

        # Extract MR information
        mr_data = webhook_data.get("object_attributes", {})
        project_id = webhook_data.get("project", {}).get("id")
        mr_iid = mr_data.get("iid")
        action = mr_data.get("action")

        # Check if we should process this event
        if action not in ["open", "reopen", "update"]:
            logger.info(f"Ignoring MR action: {action}")
            return {"status": "ignored", "reason": f"action_{action}_not_configured"}

        # Check project configuration
        project_config = self.config.get_project_config(project_id)
        if not project_config or not project_config.review_enabled:
            logger.info(f"Review disabled for project {project_id}")
            return {"status": "ignored", "reason": "review_disabled"}

        # Check if it's a draft and we should skip
        if mr_data.get("draft") and not project_config.review_drafts:
            logger.info(f"Skipping draft MR {mr_iid}")
            return {"status": "ignored", "reason": "draft_mr"}

        # Add to processing queue
        review_request = {
            "project_id": project_id,
            "mr_iid": mr_iid,
            "action": action,
            "timestamp": datetime.now(),
            "webhook_data": webhook_data,
        }

        if wait_for_completion:
            # Process synchronously for testing
            return await self._process_review_sync(review_request)
        else:
            # Add to processing queue for async processing
            await self.processing_queue.put(review_request)
            # Start async processing
            asyncio.create_task(self._process_review_async(review_request))
            return {"status": "queued", "project_id": project_id, "mr_iid": mr_iid}

    async def _process_review_sync(self, review_request: Dict[str, Any]) -> Dict[str, Any]:
        """Process review request synchronously (for testing)"""
        project_id = review_request["project_id"]
        mr_iid = review_request["mr_iid"]

        review_key = f"{project_id}_{mr_iid}"

        # Check if already processing
        if review_key in self.active_reviews:
            logger.info(f"Review already in progress for {review_key}")
            return {"status": "already_processing"}

        self.active_reviews[review_key] = True

        try:
            # Get MR information
            mr_info = self.gitlab.get_merge_request(project_id, mr_iid)
            self.gitlab.set_pipeline_status(
                project_id,
                mr_info.diff_refs["head_sha"],
                "running",
                name="AI Code Review",
                description="Analyzing code changes...",
            )

            # Perform the review
            result = await self.review_merge_request(project_id, mr_iid)

            # Set final pipeline status
            if result["status"] == "success":
                pipeline_status = "success"
                if result["summary"]["status"] == "needs_work":
                    pipeline_status = "failed"

                self.gitlab.set_pipeline_status(
                    project_id,
                    mr_info.diff_refs["head_sha"],
                    pipeline_status,
                    name="AI Code Review",
                    description=result["summary"].get(
                        "overall_assessment", "Review complete"
                    ),
                )
            else:
                self.gitlab.set_pipeline_status(
                    project_id,
                    mr_info.diff_refs["head_sha"],
                    "failed",
                    name="AI Code Review",
                    description="Review failed - check logs",
                )

            return result

        except Exception as e:
            logger.error(f"Error processing review for {review_key}: {e}")

            # Set error status
            try:
                mr_info = self.gitlab.get_merge_request(project_id, mr_iid)
                self.gitlab.set_pipeline_status(
                    project_id,
                    mr_info.diff_refs["head_sha"],
                    "failed",
                    name="AI Code Review",
                    description=f"Review failed: {str(e)}",
                )
            except Exception as e2:
                logger.error(f"Failed to set error status: {e2}")

            return {"status": "error", "message": str(e)}

        finally:
            # Clean up
            if review_key in self.active_reviews:
                del self.active_reviews[review_key]

    async def _process_review_async(self, review_request: Dict[str, Any]):
        """Process review request asynchronously"""
        project_id = review_request["project_id"]
        mr_iid = review_request["mr_iid"]

        review_key = f"{project_id}_{mr_iid}"

        # Check if already processing
        if review_key in self.active_reviews:
            logger.info(f"Review already in progress for {review_key}")
            return

        self.active_reviews[review_key] = True

        try:
            # Set pipeline status to running
            mr_info = self.gitlab.get_merge_request(project_id, mr_iid)
            self.gitlab.set_pipeline_status(
                project_id,
                mr_info.diff_refs["head_sha"],
                "running",
                name="AI Code Review",
                description="Analyzing code changes...",
            )

            # Perform the review
            result = await self.review_merge_request(project_id, mr_iid)

            # Set final pipeline status
            if result["status"] == "success":
                pipeline_status = "success"
                if result["summary"]["status"] == "needs_work":
                    pipeline_status = "failed"

                self.gitlab.set_pipeline_status(
                    project_id,
                    mr_info.diff_refs["head_sha"],
                    pipeline_status,
                    name="AI Code Review",
                    description=result["summary"].get(
                        "overall_assessment", "Review complete"
                    ),
                )
            else:
                self.gitlab.set_pipeline_status(
                    project_id,
                    mr_info.diff_refs["head_sha"],
                    "failed",
                    name="AI Code Review",
                    description="Review failed - check logs",
                )

        except Exception as e:
            logger.error(f"Error processing review for {review_key}: {e}")

            # Set error status
            try:
                mr_info = self.gitlab.get_merge_request(project_id, mr_iid)
                self.gitlab.set_pipeline_status(
                    project_id,
                    mr_info.diff_refs["head_sha"],
                    "failed",
                    name="AI Code Review",
                    description=f"Error: {str(e)[:100]}",
                )
            except Exception as e:
                logger.warning(f"Failed to set pipeline status: {e}")

        finally:
            # Remove from active reviews
            del self.active_reviews[review_key]

    async def review_merge_request(
        self, project_id: int, mr_iid: int
    ) -> Dict[str, Any]:
        """Perform complete review of a merge request"""

        logger.info(f"Starting review for project {project_id}, MR {mr_iid}")

        try:
            # Get MR information
            mr_info = self.gitlab.get_merge_request(project_id, mr_iid)

            # Get project configuration
            project_config = self.config.get_project_config(project_id)
            if not project_config:
                logger.error(f"No configuration found for project {project_id}")
                return {"status": "error", "message": "Project not configured"}

            # Type assertion for mypy
            assert project_config is not None

            # Get file diffs
            file_diffs = self.gitlab.get_merge_request_diffs(project_id, mr_iid)

            if not file_diffs:
                logger.info(f"No diffs found for MR {mr_iid}")
                return {"status": "success", "message": "No files to review"}

            # Filter files based on configuration
            files_to_review = []
            for diff in file_diffs:
                if self.config.is_file_reviewable(
                    project_id, diff.new_path or diff.old_path, diff.total_changes
                ):
                    files_to_review.append(diff)

            if not files_to_review:
                logger.info(f"No reviewable files in MR {mr_iid}")
                return {"status": "success", "message": "No reviewable files found"}

            # Limit number of files if configured
            if len(files_to_review) > project_config.max_files_per_review:
                logger.warning(
                    f"Too many files ({len(files_to_review)}), limiting to {project_config.max_files_per_review}"
                )
                files_to_review = files_to_review[: project_config.max_files_per_review]

            # Build analysis context
            context = await self._build_analysis_context(
                project_id, project_config.name, mr_info
            )

            # Prepare file data for analysis
            file_data = [
                (diff.new_path or diff.old_path, diff.diff, diff.language)
                for diff in files_to_review
            ]

            # Analyze with OpenAI
            analysis_result = await self.analyzer.analyze_merge_request(
                file_data,
                context,
                batch_size=project_config.max_files_per_review // 5
                if project_config.max_files_per_review > 5
                else 1,
            )

            # Post review comments
            await self._post_review_feedback(
                project_id,
                mr_iid,
                analysis_result["results"],
                analysis_result["summary"],
            )

            logger.info(f"Review completed for MR {mr_iid}")

            return {
                "status": "success",
                "files_reviewed": len(files_to_review),
                "issues_found": len(analysis_result["results"]),
                "summary": analysis_result["summary"],
            }

        except Exception as e:
            logger.error(f"Failed to review MR {mr_iid}: {e}")
            return {"status": "error", "error": str(e)}

    async def _build_analysis_context(
        self, project_id: int, project_name: str, mr_info: MergeRequestInfo
    ) -> AnalysisContext:
        """Build context for code analysis"""

        # Get similar code examples from vector store
        similar_code = []
        if self.memory:
            # Get a sample of the changes to find similar code
            diffs = self.gitlab.get_merge_request_diffs(project_id, mr_info.mr_iid)
            if diffs and len(diffs) > 0:
                sample_diff = diffs[0].diff[:1000]  # First 1000 chars
                collection_name = f"gitlab_code_{project_name}"
                similar_code = await self.memory.vector_store.search_similar(
                    sample_diff, collection_name, limit=3
                )

        # Extract project patterns
        patterns = []
        if self.memory:
            patterns = await self.memory.extract_coding_patterns(project_name, limit=5)

        # Get recent review comments for consistency
        recent_reviews = []
        existing_comments = self.gitlab.get_existing_comments(
            project_id, mr_info.mr_iid
        )
        for comment in existing_comments[:3]:
            recent_reviews.append(comment["body"][:200])

        return AnalysisContext(
            project_name=project_name,
            merge_request_title=mr_info.title,
            merge_request_description=mr_info.description,
            target_branch=mr_info.target_branch,
            similar_code_examples=similar_code,
            project_patterns=patterns,
            recent_reviews=recent_reviews,
        )

    async def _post_review_feedback(
        self,
        project_id: int,
        mr_iid: int,
        results: List[CodeReviewResult],
        summary: Dict[str, Any],
    ):
        """Post review feedback to GitLab"""

        # Group results by file and severity
        file_comments: Dict[str, List[CodeReviewResult]] = {}
        for result in results:
            if result.file_path not in file_comments:
                file_comments[result.file_path] = []
            file_comments[result.file_path].append(result)

        # Post inline comments for critical and major issues
        for file_path, comments in file_comments.items():
            for comment in comments:
                if comment.severity in ["critical", "major"]:
                    # Post inline comment
                    full_comment = comment.message
                    if comment.suggestion:
                        full_comment += (
                            f"\n\n**Suggestion:**\n```\n{comment.suggestion}\n```"
                        )

                    self.gitlab.post_review_comment(
                        project_id,
                        mr_iid,
                        full_comment,
                        file_path=file_path,
                        line=comment.line_number,
                        severity=comment.severity,
                    )

        # Post summary comment
        self.gitlab.post_review_summary(project_id, mr_iid, summary)

    async def learn_from_merge(self, project_id: int, mr_iid: int):
        """Learn from a merged MR to update knowledge base"""

        if not self.memory:
            logger.warning("No memory manager configured for learning")
            return

        try:
            # Get MR info
            mr_info = self.gitlab.get_merge_request(project_id, mr_iid)

            if mr_info.state != "merged":
                logger.info(f"MR {mr_iid} not merged, skipping learning")
                return

            # Get project config
            project_config = self.config.get_project_config(project_id)
            if not project_config:
                return

            # Get final file contents after merge
            file_changes = []
            diffs = self.gitlab.get_merge_request_diffs(project_id, mr_iid)

            for diff in diffs:
                if diff.is_deleted_file:
                    continue

                file_path = diff.new_path or diff.old_path
                content = self.gitlab.get_file_content(
                    project_id, file_path, mr_info.target_branch
                )

                if content:
                    file_changes.append(
                        {"file_path": file_path, "new_content": content}
                    )

            # Update knowledge base
            await self.memory.update_from_merged_code(
                project_id, project_config.name, file_changes
            )

            logger.info(f"Learned from merged MR {mr_iid}")

        except Exception as e:
            logger.error(f"Failed to learn from MR {mr_iid}: {e}")
