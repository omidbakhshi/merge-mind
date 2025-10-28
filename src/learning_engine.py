"""
Learning engine for continuous improvement based on feedback and merged code
Tracks patterns, learns from successful merges, and adapts to team preferences
Location: src/learning_engine.py
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
from pathlib import Path

from src.vector_store import VectorStoreBase
from src.gitlab_client import GitLabClient
from src.config_manager import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class ReviewFeedback:
    """Feedback on a review"""

    project_id: int
    mr_iid: int
    file_path: str
    comment_id: str
    feedback_type: str  # 'helpful', 'not_helpful', 'false_positive'
    timestamp: datetime
    user: str
    original_severity: str
    resolution: Optional[str] = None  # 'fixed', 'ignored', 'disputed'


@dataclass
class CodingPattern:
    """A learned coding pattern"""

    pattern_id: str
    project_id: int
    pattern_type: str  # 'naming', 'structure', 'import', 'error_handling', etc.
    description: str
    examples: List[str]
    confidence: float
    occurrences: int
    last_seen: datetime

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["last_seen"] = self.last_seen.isoformat()
        return data


@dataclass
class TeamPreference:
    """Team's coding preferences learned from reviews"""

    project_id: int
    preference_type: str
    description: str
    examples: List[Dict[str, Any]]
    weight: float  # Importance weight


class LearningEngine:
    """Manages continuous learning from code reviews and feedback"""

    def __init__(self, vector_store: VectorStoreBase, gitlab_client: GitLabClient, config_manager: ConfigManager) -> None:
        self.vector_store = vector_store
        self.gitlab_client = gitlab_client
        self.config = config_manager

        # Storage paths
        self.data_dir = Path("storage/learning")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # In-memory caches
        self.patterns_cache: Dict[int, List[CodingPattern]] = defaultdict(list)
        self.feedback_cache: List[ReviewFeedback] = []
        self.preferences_cache: Dict[int, List[TeamPreference]] = defaultdict(list)

        # Learning statistics
        self.stats = {
            "patterns_learned": 0,
            "feedback_processed": 0,
            "accuracy_metrics": {},
            "last_learning_cycle": None,
        }

        # Load existing data
        self._load_learning_data()

    def _load_learning_data(self):
        """Load existing learning data from disk"""
        patterns_file = self.data_dir / "patterns.json"
        feedback_file = self.data_dir / "feedback.json"

        # Load patterns
        if patterns_file.exists():
            try:
                with open(patterns_file, "r") as f:
                    data = json.load(f)
                    for project_id, patterns in data.items():
                        for pattern_data in patterns:
                            pattern = CodingPattern(**pattern_data)
                            pattern.last_seen = datetime.fromisoformat(
                                pattern_data["last_seen"]
                            )
                            self.patterns_cache[int(project_id)].append(pattern)
                logger.info(
                    f"Loaded {sum(len(p) for p in self.patterns_cache.values())} patterns"
                )
            except Exception as e:
                logger.error(f"Failed to load patterns: {e}")

        # Load feedback
        if feedback_file.exists():
            try:
                with open(feedback_file, "r") as f:
                    data = json.load(f)
                    for feedback_data in data:
                        feedback = ReviewFeedback(**feedback_data)
                        feedback.timestamp = datetime.fromisoformat(
                            feedback_data["timestamp"]
                        )
                        self.feedback_cache.append(feedback)
                logger.info(f"Loaded {len(self.feedback_cache)} feedback entries")
            except Exception as e:
                logger.error(f"Failed to load feedback: {e}")

    def _save_learning_data(self):
        """Persist learning data to disk"""
        patterns_file = self.data_dir / "patterns.json"
        feedback_file = self.data_dir / "feedback.json"

        # Save patterns
        patterns_data = {}
        for project_id, patterns in self.patterns_cache.items():
            patterns_data[str(project_id)] = [p.to_dict() for p in patterns]

        with open(patterns_file, "w") as f:
            json.dump(patterns_data, f, indent=2)

        # Save feedback
        feedback_data = []
        for fb in self.feedback_cache:
            fb_dict = asdict(fb)
            fb_dict["timestamp"] = fb.timestamp.isoformat()
            feedback_data.append(fb_dict)

        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f, indent=2)

        logger.info("Saved learning data to disk")

    async def learn_from_merged_mr(
        self, project_id: int, mr_iid: int
    ) -> Dict[str, Any]:
        """Learn patterns from a successfully merged MR

        Args:
            project_id: GitLab project ID
            mr_iid: Merge request IID
        """
        logger.info(f"Learning from merged MR {mr_iid} in project {project_id}")

        try:
            # Get MR details
            mr_info = self.gitlab_client.get_merge_request(project_id, mr_iid)

            if mr_info.state != "merged":
                return {"status": "skipped", "reason": "not_merged"}

            # Get the diffs
            diffs = self.gitlab_client.get_merge_request_diffs(project_id, mr_iid)

            patterns_learned = []

            for diff in diffs:
                if diff.is_deleted_file:
                    continue

                # Extract patterns from the diff
                patterns = self._extract_patterns_from_diff(
                    project_id, diff.new_path or diff.old_path, diff.diff, diff.language
                )

                patterns_learned.extend(patterns)

            # Update pattern statistics
            self._update_pattern_confidence(project_id, patterns_learned)

            # Learn from review comments that were addressed
            addressed_patterns = await self._learn_from_addressed_comments(
                project_id, mr_iid
            )

            # Update vector store with new patterns
            if patterns_learned:
                await self._update_vector_store_patterns(project_id, patterns_learned)

            # Save data
            self._save_learning_data()

            return {
                "status": "success",
                "patterns_learned": len(patterns_learned),
                "addressed_comments": len(addressed_patterns),
            }

        except Exception as e:
            logger.error(f"Failed to learn from MR {mr_iid}: {e}")
            return {"status": "error", "error": str(e)}

    def _extract_patterns_from_diff(
        self, project_id: int, file_path: str, diff: str, language: Optional[str]
    ) -> List[CodingPattern]:
        """Extract coding patterns from a diff"""
        patterns = []

        # Extract different types of patterns based on language
        if language == "python":
            patterns.extend(self._extract_python_patterns(project_id, file_path, diff))
        elif language in ["javascript", "typescript"]:
            patterns.extend(self._extract_js_patterns(project_id, file_path, diff))

        # Extract general patterns (applicable to all languages)
        patterns.extend(self._extract_general_patterns(project_id, file_path, diff))

        return patterns

    def _extract_python_patterns(
        self, project_id: int, file_path: str, diff: str
    ) -> List[CodingPattern]:
        """Extract Python-specific patterns"""
        patterns = []

        import re

        # Extract function naming patterns
        func_pattern = re.compile(r"\+def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
        for match in func_pattern.finditer(diff):
            func_name = match.group(1)

            # Detect naming convention
            if "_" in func_name:
                pattern_type = "naming_snake_case"
                description = "Functions use snake_case naming"
            elif func_name[0].isupper():
                pattern_type = "naming_pascal_case"
                description = "Classes/constructors use PascalCase"
            else:
                pattern_type = "naming_camel_case"
                description = "Functions use camelCase naming"

            pattern_id = hashlib.md5(
                f"{project_id}_{pattern_type}".encode()
            ).hexdigest()[:8]

            pattern = CodingPattern(
                pattern_id=pattern_id,
                project_id=project_id,
                pattern_type=pattern_type,
                description=description,
                examples=[func_name],
                confidence=0.7,
                occurrences=1,
                last_seen=datetime.now(),
            )

            # Check if pattern exists and update
            existing = self._find_pattern(project_id, pattern_id)
            if existing:
                existing.occurrences += 1
                existing.last_seen = datetime.now()
                if func_name not in existing.examples:
                    existing.examples.append(func_name)
                existing.confidence = min(0.95, existing.confidence + 0.05)
            else:
                patterns.append(pattern)
                self.patterns_cache[project_id].append(pattern)

        # Extract import patterns
        import_pattern = re.compile(r"\+(?:from\s+([^\s]+)\s+)?import\s+([^\s]+)")
        for match in import_pattern.finditer(diff):
            module = match.group(1) or match.group(2)

            pattern_id = hashlib.md5(
                f"{project_id}_import_{module}".encode()
            ).hexdigest()[:8]

            pattern = CodingPattern(
                pattern_id=pattern_id,
                project_id=project_id,
                pattern_type="import_style",
                description=f"Common import: {module}",
                examples=[match.group(0).strip("+")],
                confidence=0.6,
                occurrences=1,
                last_seen=datetime.now(),
            )

            existing = self._find_pattern(project_id, pattern_id)
            if existing:
                existing.occurrences += 1
                existing.confidence = min(0.95, existing.confidence + 0.05)
            else:
                patterns.append(pattern)
                self.patterns_cache[project_id].append(pattern)

        # Extract error handling patterns
        try_pattern = re.compile(
            r"\+\s*try:.*?\n.*?\+\s*except\s+([^:]+):", re.MULTILINE | re.DOTALL
        )
        for match in try_pattern.finditer(diff):
            exception_type = match.group(1).strip()

            pattern_id = hashlib.md5(
                f"{project_id}_exception_{exception_type}".encode()
            ).hexdigest()[:8]

            pattern = CodingPattern(
                pattern_id=pattern_id,
                project_id=project_id,
                pattern_type="error_handling",
                description=f"Handles {exception_type} exceptions",
                examples=[exception_type],
                confidence=0.7,
                occurrences=1,
                last_seen=datetime.now(),
            )

            existing = self._find_pattern(project_id, pattern_id)
            if existing:
                existing.occurrences += 1
                existing.confidence = min(0.95, existing.confidence + 0.05)
            else:
                patterns.append(pattern)
                self.patterns_cache[project_id].append(pattern)

        return patterns

    def _extract_js_patterns(
        self, project_id: int, file_path: str, diff: str
    ) -> List[CodingPattern]:
        """Extract JavaScript/TypeScript patterns"""
        patterns = []

        import re

        # Extract async/await usage
        async_pattern = re.compile(r"\+\s*async\s+")
        if async_pattern.search(diff):
            pattern_id = hashlib.md5(f"{project_id}_async_await".encode()).hexdigest()[
                :8
            ]

            pattern = CodingPattern(
                pattern_id=pattern_id,
                project_id=project_id,
                pattern_type="async_pattern",
                description="Uses async/await for asynchronous operations",
                examples=["async/await"],
                confidence=0.8,
                occurrences=1,
                last_seen=datetime.now(),
            )

            existing = self._find_pattern(project_id, pattern_id)
            if existing:
                existing.occurrences += 1
                existing.confidence = min(0.95, existing.confidence + 0.05)
            else:
                patterns.append(pattern)
                self.patterns_cache[project_id].append(pattern)

        # Extract arrow function vs regular function usage
        arrow_pattern = re.compile(r"\+.*?=>\s*{")
        function_pattern = re.compile(r"\+\s*function\s+")

        if arrow_pattern.search(diff):
            pattern_type = "arrow_functions"
            description = "Prefers arrow functions"
        elif function_pattern.search(diff):
            pattern_type = "regular_functions"
            description = "Uses regular function declarations"
        else:
            pattern_type = None

        if pattern_type:
            pattern_id = hashlib.md5(
                f"{project_id}_{pattern_type}".encode()
            ).hexdigest()[:8]

            pattern = CodingPattern(
                pattern_id=pattern_id,
                project_id=project_id,
                pattern_type=pattern_type,
                description=description,
                examples=[],
                confidence=0.7,
                occurrences=1,
                last_seen=datetime.now(),
            )

            existing = self._find_pattern(project_id, pattern_id)
            if existing:
                existing.occurrences += 1
                existing.confidence = min(0.95, existing.confidence + 0.05)
            else:
                patterns.append(pattern)
                self.patterns_cache[project_id].append(pattern)

        return patterns

    def _extract_general_patterns(
        self, project_id: int, file_path: str, diff: str
    ) -> List[CodingPattern]:
        """Extract language-agnostic patterns"""
        patterns = []

        import re

        # Extract comment style patterns
        single_line_comment = re.compile(r"\+\s*//.*")
        multi_line_comment = re.compile(r"\+\s*/\*.*?\*/", re.DOTALL)
        hash_comment = re.compile(r"\+\s*#.*")

        comment_style = None
        if single_line_comment.search(diff):
            comment_style = "double_slash"
            description = "Uses // for comments"
        elif multi_line_comment.search(diff):
            comment_style = "multi_line"
            description = "Uses /* */ for comments"
        elif hash_comment.search(diff):
            comment_style = "hash"
            description = "Uses # for comments"

        if comment_style:
            pattern_id = hashlib.md5(
                f"{project_id}_comment_{comment_style}".encode()
            ).hexdigest()[:8]

            pattern = CodingPattern(
                pattern_id=pattern_id,
                project_id=project_id,
                pattern_type="comment_style",
                description=description,
                examples=[],
                confidence=0.6,
                occurrences=1,
                last_seen=datetime.now(),
            )

            existing = self._find_pattern(project_id, pattern_id)
            if existing:
                existing.occurrences += 1
                existing.confidence = min(0.95, existing.confidence + 0.05)
            else:
                patterns.append(pattern)
                self.patterns_cache[project_id].append(pattern)

        # Extract indentation patterns
        two_space = re.compile(r"\n\+  [^\s]")
        four_space = re.compile(r"\n\+    [^\s]")
        tab = re.compile(r"\n\+\t")

        indent_style = None
        if two_space.search(diff):
            indent_style = "two_spaces"
            description = "Uses 2 spaces for indentation"
        elif four_space.search(diff):
            indent_style = "four_spaces"
            description = "Uses 4 spaces for indentation"
        elif tab.search(diff):
            indent_style = "tabs"
            description = "Uses tabs for indentation"

        if indent_style:
            pattern_id = hashlib.md5(
                f"{project_id}_indent_{indent_style}".encode()
            ).hexdigest()[:8]

            pattern = CodingPattern(
                pattern_id=pattern_id,
                project_id=project_id,
                pattern_type="indentation",
                description=description,
                examples=[],
                confidence=0.8,
                occurrences=1,
                last_seen=datetime.now(),
            )

            existing = self._find_pattern(project_id, pattern_id)
            if existing:
                existing.occurrences += 1
                existing.confidence = min(0.95, existing.confidence + 0.05)
            else:
                patterns.append(pattern)
                self.patterns_cache[project_id].append(pattern)

        return patterns

    def _find_pattern(
        self, project_id: int, pattern_id: str
    ) -> Optional[CodingPattern]:
        """Find an existing pattern by ID"""
        for pattern in self.patterns_cache[project_id]:
            if pattern.pattern_id == pattern_id:
                return pattern
        return None

    def _update_pattern_confidence(
        self, project_id: int, patterns: List[CodingPattern]
    ):
        """Update confidence scores based on occurrence frequency"""
        total_occurrences = {}

        # Count total occurrences by type
        for pattern in self.patterns_cache[project_id]:
            if pattern.pattern_type not in total_occurrences:
                total_occurrences[pattern.pattern_type] = 0
            total_occurrences[pattern.pattern_type] += pattern.occurrences

        # Update confidence based on relative frequency
        for pattern in self.patterns_cache[project_id]:
            if pattern.pattern_type in total_occurrences:
                relative_freq = (
                    pattern.occurrences / total_occurrences[pattern.pattern_type]
                )
                pattern.confidence = min(0.95, 0.5 + relative_freq * 0.45)

    async def _learn_from_addressed_comments(
        self, project_id: int, mr_iid: int
    ) -> List[TeamPreference]:
        """Learn from review comments that were addressed"""
        preferences = []

        try:
            # Get all bot comments on the MR
            comments = self.gitlab_client.get_existing_comments(project_id, mr_iid)

            # Get the final merged code
            _mr_info = self.gitlab_client.get_merge_request(project_id, mr_iid)
            _diffs = self.gitlab_client.get_merge_request_diffs(project_id, mr_iid)

            for comment in comments:
                # Check if comment was about a specific issue
                if "CRITICAL" in comment["body"] or "MAJOR" in comment["body"]:
                    # Check if the issue was addressed in final code
                    # This is simplified - in reality you'd do more sophisticated checking
                    preference = TeamPreference(
                        project_id=project_id,
                        preference_type="addressed_issue",
                        description=f"Team addresses {comment['body'][:100]}",
                        examples=[{"mr_iid": mr_iid, "comment": comment["body"][:200]}],
                        weight=0.8,
                    )
                    preferences.append(preference)
                    self.preferences_cache[project_id].append(preference)

        except Exception as e:
            logger.error(f"Failed to learn from addressed comments: {e}")

        return preferences

    async def _update_vector_store_patterns(
        self, project_id: int, patterns: List[CodingPattern]
    ):
        """Update vector store with learned patterns"""
        if not self.vector_store:
            return

        project_config = self.config.get_project_config(project_id)
        if not project_config:
            return

        collection_name = f"gitlab_patterns_{project_config.name}"

        documents = []
        for pattern in patterns:
            doc = {
                "id": pattern.pattern_id,
                "content": f"{pattern.description}\nExamples: {', '.join(pattern.examples[:5])}",
                "metadata": {
                    "project_id": project_id,
                    "pattern_type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "occurrences": pattern.occurrences,
                },
            }
            documents.append(doc)

        if documents:
            await self.vector_store.add_documents(documents, collection_name)

    def record_feedback(self, feedback: ReviewFeedback):
        """Record feedback on a review"""
        self.feedback_cache.append(feedback)
        self.stats["feedback_processed"] += 1

        # Adjust patterns based on feedback
        if feedback.feedback_type == "false_positive":
            # Reduce confidence in patterns that led to this review
            self._adjust_pattern_confidence_from_feedback(feedback)

        self._save_learning_data()

    def _adjust_pattern_confidence_from_feedback(self, feedback: ReviewFeedback):
        """Adjust pattern confidence based on feedback"""
        # Find patterns related to this file type and reduce confidence slightly
        for pattern in self.patterns_cache[feedback.project_id]:
            if pattern.pattern_type in ["error_handling", "code_style"]:
                pattern.confidence = max(0.3, pattern.confidence - 0.1)

    def get_relevant_patterns(
        self, project_id: int, file_path: str, language: Optional[str]
    ) -> List[CodingPattern]:
        """Get relevant patterns for a file being reviewed"""
        relevant_patterns = []

        for pattern in self.patterns_cache[project_id]:
            # Filter by confidence threshold
            if pattern.confidence < 0.5:
                continue

            # Filter by recency (patterns seen in last 30 days)
            if (datetime.now() - pattern.last_seen).days > 30:
                continue

            relevant_patterns.append(pattern)

        # Sort by confidence and occurrences
        relevant_patterns.sort(
            key=lambda p: (p.confidence, p.occurrences), reverse=True
        )

        return relevant_patterns[:10]  # Return top 10 most relevant patterns

    def get_team_preferences(self, project_id: int) -> List[TeamPreference]:
        """Get team preferences for a project"""
        return self.preferences_cache[project_id]

    async def run_learning_cycle(self):
        """Run a complete learning cycle across all projects"""
        logger.info("Starting learning cycle")

        for project_id in self.config.projects.keys():
            project_config = self.config.get_project_config(project_id)
            if not project_config or not project_config.review_enabled:
                continue

            try:
                # Get recently merged MRs
                project = self.gitlab_client.get_project(project_id)

                # Get MRs merged in last 7 days
                since = (datetime.now() - timedelta(days=7)).isoformat()
                mrs = project.mergerequests.list(
                    state="merged", updated_after=since, per_page=20
                )

                for mr in mrs:
                    await self.learn_from_merged_mr(project_id, mr.iid)

                logger.info(
                    f"Learned from {len(mrs)} merged MRs in project {project_id}"
                )

            except Exception as e:
                logger.error(f"Failed learning cycle for project {project_id}: {e}")

        self.stats["last_learning_cycle"] = datetime.now()
        logger.info("Learning cycle complete")

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        stats = dict(self.stats)
        stats["total_patterns"] = sum(len(p) for p in self.patterns_cache.values())
        stats["projects_with_patterns"] = len(self.patterns_cache)
        stats["total_feedback"] = len(self.feedback_cache)

        # Calculate accuracy metrics
        if self.feedback_cache:
            total_feedback = len(self.feedback_cache)
            false_positives = sum(
                1 for f in self.feedback_cache if f.feedback_type == "false_positive"
            )
            helpful = sum(
                1 for f in self.feedback_cache if f.feedback_type == "helpful"
            )

            stats["accuracy_metrics"] = {
                "false_positive_rate": false_positives / total_feedback,
                "helpful_rate": helpful / total_feedback,
            }

        return stats
