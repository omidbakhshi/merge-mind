import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


@dataclass
class ReviewMetrics:
    """Metrics for a single review operation"""
    project_id: int
    mr_iid: int
    start_time: float
    end_time: Optional[float] = None
    files_reviewed: int = 0
    issues_found: int = 0
    processing_time: float = 0.0
    status: str = "in_progress"

    def complete(self, files_reviewed: int = 0, issues_found: int = 0):
        """Mark review as completed"""
        self.end_time = time.time()
        self.processing_time = self.end_time - self.start_time
        self.files_reviewed = files_reviewed
        self.issues_found = issues_found
        self.status = "completed"

    def fail(self, error: str = ""):
        """Mark review as failed"""
        self.end_time = time.time()
        self.processing_time = self.end_time - self.start_time
        self.status = f"failed: {error}"


@dataclass
class LearningMetrics:
    """Metrics for learning operations"""
    project_name: str
    start_time: float
    end_time: Optional[float] = None
    files_processed: int = 0
    chunks_created: int = 0
    processing_time: float = 0.0
    status: str = "in_progress"

    def complete(self, files_processed: int = 0, chunks_created: int = 0):
        """Mark learning as completed"""
        self.end_time = time.time()
        self.processing_time = self.end_time - self.start_time
        self.files_processed = files_processed
        self.chunks_created = chunks_created
        self.status = "completed"

    def fail(self, error: str = ""):
        """Mark learning as failed"""
        self.end_time = time.time()
        self.processing_time = self.end_time - self.start_time
        self.status = f"failed: {error}"


class MetricsCollector:
    """Simple metrics collector for the application"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.review_metrics: Dict[str, ReviewMetrics] = {}
        self.learning_metrics: Dict[str, LearningMetrics] = {}
        self.counters: Dict[str, int] = {
            "total_reviews": 0,
            "successful_reviews": 0,
            "failed_reviews": 0,
            "total_learning_operations": 0,
            "successful_learning_operations": 0,
            "failed_learning_operations": 0,
            "api_calls_openai": 0,
            "api_calls_gitlab": 0,
        }
        self.start_time = time.time()

    def start_review(self, project_id: int, mr_iid: int) -> str:
        """Start tracking a review operation"""
        review_id = f"review_{project_id}_{mr_iid}_{int(time.time())}"
        with self._lock:
            self.review_metrics[review_id] = ReviewMetrics(
                project_id=project_id,
                mr_iid=mr_iid,
                start_time=time.time()
            )
        logger.debug(f"Started tracking review {review_id}")
        return review_id

    def complete_review(self, review_id: str, files_reviewed: int = 0, issues_found: int = 0):
        """Complete a review operation"""
        with self._lock:
            if review_id in self.review_metrics:
                self.review_metrics[review_id].complete(files_reviewed, issues_found)
                self.counters["total_reviews"] += 1
                self.counters["successful_reviews"] += 1
                logger.info(f"Completed review {review_id}: {files_reviewed} files, {issues_found} issues")

    def fail_review(self, review_id: str, error: str = ""):
        """Mark a review as failed"""
        with self._lock:
            if review_id in self.review_metrics:
                self.review_metrics[review_id].fail(error)
                self.counters["total_reviews"] += 1
                self.counters["failed_reviews"] += 1
                logger.warning(f"Failed review {review_id}: {error}")

    def start_learning(self, project_name: str) -> str:
        """Start tracking a learning operation"""
        learning_id = f"learn_{project_name}_{int(time.time())}"
        with self._lock:
            self.learning_metrics[learning_id] = LearningMetrics(
                project_name=project_name,
                start_time=time.time()
            )
        logger.debug(f"Started tracking learning {learning_id}")
        return learning_id

    def complete_learning(self, learning_id: str, files_processed: int = 0, chunks_created: int = 0):
        """Complete a learning operation"""
        with self._lock:
            if learning_id in self.learning_metrics:
                self.learning_metrics[learning_id].complete(files_processed, chunks_created)
                self.counters["total_learning_operations"] += 1
                self.counters["successful_learning_operations"] += 1
                logger.info(f"Completed learning {learning_id}: {files_processed} files, {chunks_created} chunks")

    def fail_learning(self, learning_id: str, error: str = ""):
        """Mark a learning operation as failed"""
        with self._lock:
            if learning_id in self.learning_metrics:
                self.learning_metrics[learning_id].fail(error)
                self.counters["total_learning_operations"] += 1
                self.counters["failed_learning_operations"] += 1
                logger.warning(f"Failed learning {learning_id}: {error}")

    def increment_counter(self, counter_name: str, amount: int = 1):
        """Increment a counter"""
        with self._lock:
            self.counters[counter_name] = self.counters.get(counter_name, 0) + amount

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        with self._lock:
            uptime_seconds = time.time() - self.start_time

            # Calculate averages
            review_times = [m.processing_time for m in self.review_metrics.values() if m.end_time]
            avg_review_time = sum(review_times) / len(review_times) if review_times else 0

            learning_times = [m.processing_time for m in self.learning_metrics.values() if m.end_time]
            avg_learning_time = sum(learning_times) / len(learning_times) if learning_times else 0

            return {
                "uptime_seconds": uptime_seconds,
                "total_reviews": self.counters["total_reviews"],
                "successful_reviews": self.counters["successful_reviews"],
                "failed_reviews": self.counters["failed_reviews"],
                "total_learning_operations": self.counters["total_learning_operations"],
                "successful_learning_operations": self.counters["successful_learning_operations"],
                "failed_learning_operations": self.counters["failed_learning_operations"],
                "average_review_time_seconds": avg_review_time,
                "average_learning_time_seconds": avg_learning_time,
                "api_calls_openai": self.counters["api_calls_openai"],
                "api_calls_gitlab": self.counters["api_calls_gitlab"],
                "active_reviews": len([r for r in self.review_metrics.values() if not r.end_time]),
                "active_learning": len([l for l in self.learning_metrics.values() if not l.end_time]),
            }

    def get_recent_activity(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent activity"""
        with self._lock:
            # Get recent reviews
            recent_reviews = sorted(
                [r for r in self.review_metrics.values() if r.end_time],
                key=lambda x: x.end_time or 0,
                reverse=True
            )[:limit]

            # Get recent learning operations
            recent_learning = sorted(
                [l for l in self.learning_metrics.values() if l.end_time],
                key=lambda x: x.end_time or 0,
                reverse=True
            )[:limit]

            return {
                "recent_reviews": [
                    {
                        "project_id": r.project_id,
                        "mr_iid": r.mr_iid,
                        "processing_time": r.processing_time,
                        "files_reviewed": r.files_reviewed,
                        "issues_found": r.issues_found,
                        "status": r.status,
                        "completed_at": datetime.fromtimestamp(r.end_time).isoformat() if r.end_time else None,
                    }
                    for r in recent_reviews
                ],
                "recent_learning": [
                    {
                        "project_name": l.project_name,
                        "processing_time": l.processing_time,
                        "files_processed": l.files_processed,
                        "chunks_created": l.chunks_created,
                        "status": l.status,
                        "completed_at": datetime.fromtimestamp(l.end_time).isoformat() if l.end_time else None,
                    }
                    for l in recent_learning
                ],
            }

    def cleanup_old_metrics(self, max_age_days: int = 30):
        """Clean up old metrics to prevent memory bloat"""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

        with self._lock:
            # Remove old completed reviews
            self.review_metrics = {
                k: v for k, v in self.review_metrics.items()
                if not v.end_time or v.end_time > cutoff_time
            }

            # Remove old completed learning operations
            self.learning_metrics = {
                k: v for k, v in self.learning_metrics.items()
                if not v.end_time or v.end_time > cutoff_time
            }

            logger.info(f"Cleaned up old metrics, kept {len(self.review_metrics)} reviews and {len(self.learning_metrics)} learning operations")


# Global metrics collector instance
metrics_collector = MetricsCollector()