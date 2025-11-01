import os
import logging
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import uvicorn
from contextlib import asynccontextmanager

from src.config_manager import ConfigManager, ProjectConfig
from src.gitlab_client import GitLabClient
from src.openai_analyzer import OpenAIAnalyzer
from src.vector_store import CodeMemoryManager
from src.merge_request_handler import MergeRequestHandler
from src.metrics import metrics_collector
from src.circuit_breaker import circuit_breaker_registry

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# Pydantic models for API validation
class WebhookResponse(BaseModel):
    """Response model for webhook processing"""
    status: str = Field(..., description="Processing status")
    project_id: Optional[int] = Field(None, description="GitLab project ID")
    mr_iid: Optional[int] = Field(None, description="Merge request IID")
    message: Optional[str] = Field(None, description="Additional message")
    queued_at: Optional[str] = Field(None, description="Timestamp when queued")


class ReviewRequest(BaseModel):
    """Request model for manual review trigger"""
    project_id: int = Field(..., description="GitLab project ID")
    mr_iid: int = Field(..., description="Merge request IID")


class ReviewResponse(BaseModel):
    """Response model for review operations"""
    status: str = Field(..., description="Review status")
    project_id: int = Field(..., description="GitLab project ID")
    mr_iid: int = Field(..., description="Merge request IID")
    review_id: Optional[str] = Field(None, description="Unique review identifier")
    started_at: str = Field(..., description="Review start timestamp")
    estimated_duration: Optional[int] = Field(None, description="Estimated completion time in seconds")


class LearningRequest(BaseModel):
    """Request model for learning operations"""
    project_id: int = Field(..., description="GitLab project ID")
    branch: Optional[str] = Field("main", description="Branch to learn from")


class LearningResponse(BaseModel):
    """Response model for learning operations"""
    status: str = Field(..., description="Learning status")
    project_id: int = Field(..., description="GitLab project ID")
    branch: str = Field(..., description="Branch learned from")
    files_processed: int = Field(..., description="Number of files processed")
    files_skipped: int = Field(..., description="Number of files skipped")
    total_chunks: int = Field(..., description="Total code chunks created")
    duration_seconds: float = Field(..., description="Learning duration")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


class HealthCheckResponse(BaseModel):
    """Response model for basic health check"""
    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    timestamp: str = Field(..., description="Health check timestamp")


class DetailedHealthCheckResponse(BaseModel):
    """Response model for detailed health check"""
    status: str = Field(..., description="Overall health status")
    service: str = Field(..., description="Service name")
    timestamp: str = Field(..., description="Health check timestamp")
    checks: Dict[str, Dict[str, Any]] = Field(..., description="Individual component health checks")


class StatsResponse(BaseModel):
    """Response model for service statistics"""
    projects_configured: int = Field(..., description="Number of configured projects")
    active_reviews: int = Field(..., description="Number of currently active reviews")
    learning_stats: Dict[str, Any] = Field(default_factory=dict, description="Learning operation statistics")
    cache_size: int = Field(..., description="Size of review cache")


class ReloadResponse(BaseModel):
    """Response model for configuration reload"""
    success: bool = Field(..., description="Whether reload was successful")
    message: str = Field(..., description="Reload status message")
    changes: Dict[str, Any] = Field(default_factory=dict, description="Configuration changes detected")


class ProjectResponse(BaseModel):
    """Response model for project data"""
    project_id: int = Field(..., description="GitLab project ID")
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    review_enabled: bool = Field(True, description="Whether reviews are enabled")
    review_drafts: bool = Field(False, description="Whether to review draft MRs")
    min_lines_changed: int = Field(10, description="Minimum lines changed to trigger review")
    max_files_per_review: int = Field(50, description="Maximum files to review per MR")
    excluded_paths: List[str] = Field(default_factory=lambda: ["vendor/", "node_modules/", "dist/", "build/"], description="Paths to exclude from review")
    included_extensions: List[str] = Field(default_factory=lambda: [".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c"], description="File extensions to include")
    custom_prompts: Optional[Dict[str, str]] = Field(default_factory=dict, description="Custom review prompts")
    review_model: Optional[str] = Field(None, description="AI model for reviews")
    team_preferences: Optional[List[str]] = Field(default_factory=list, description="Team-specific preferences")


class ProjectsResponse(BaseModel):
    """Response model for projects list"""
    projects: List[ProjectResponse] = Field(default_factory=list, description="List of configured projects")


class ReviewHistoryItem(BaseModel):
    """Model for individual review history item"""
    id: str = Field(..., description="Unique review identifier")
    project_id: int = Field(..., description="GitLab project ID")
    mr_iid: int = Field(..., description="Merge request IID")
    status: str = Field(..., description="Review status")
    created_at: str = Field(..., description="Review creation timestamp")
    completed_at: Optional[str] = Field(None, description="Review completion timestamp")
    results_count: int = Field(0, description="Number of review results")


class ReviewHistoryResponse(BaseModel):
    """Response model for review history"""
    reviews: List[ReviewHistoryItem] = Field(default_factory=list, description="List of review history items")


class ActiveReviewItem(BaseModel):
    """Model for active review item"""
    id: str = Field(..., description="Unique review identifier")
    project_id: int = Field(..., description="GitLab project ID")
    mr_iid: int = Field(..., description="Merge request IID")
    status: str = Field(..., description="Review status")
    started_at: str = Field(..., description="Review start timestamp")


class ActiveReviewsResponse(BaseModel):
    """Response model for active reviews"""
    active_reviews: List[ActiveReviewItem] = Field(default_factory=list, description="List of active reviews")


class ProjectLearningStats(BaseModel):
    """Learning statistics for a single project"""
    project_id: int = Field(..., description="Project ID")
    project_name: str = Field(..., description="Project name")
    collection_name: str = Field(..., description="Vector store collection name")
    document_count: int = Field(0, description="Number of documents in vector store")
    last_learning: Optional[Dict[str, Any]] = Field(None, description="Last learning operation details")
    total_learning_operations: int = Field(0, description="Total learning operations")


class AllLearningStatsResponse(BaseModel):
    """Response model for all learning statistics"""
    projects: List[ProjectLearningStats] = Field(default_factory=list, description="Learning stats per project")
    total_documents: int = Field(0, description="Total documents across all projects")
    total_projects_learned: int = Field(0, description="Number of projects with learned data")


class GitLabReviewerApp:
    """Main application class"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.gitlab_client = None
        self.analyzer = None
        self.vector_store = None
        self.memory_manager = None
        self.mr_handler = None
        self.app = None

    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing GitLab AI Reviewer...")
        start_time = datetime.now()

        try:
            # Load configuration
            logger.debug("Loading configuration...")
            self.config_manager = ConfigManager()
            logger.info(f"Configuration loaded from {self.config_manager.config_dir}")

            # Initialize GitLab client
            logger.debug("Initializing GitLab client...")
            gitlab_url = self.config_manager.get_global_setting("gitlab", "url")
            gitlab_token = self.config_manager.get_global_setting("gitlab", "token")

            if not gitlab_url or not gitlab_token:
                logger.error("GitLab URL and token must be configured")
                raise ValueError("GitLab URL and token must be configured")

            self.gitlab_client = GitLabClient(gitlab_url, gitlab_token)
            logger.info(f"GitLab client initialized for {gitlab_url}")

            # Initialize vector store
            logger.debug("Initializing vector store...")
            vector_config = self.config_manager.get_global_setting("vector_store")
            openai_key = self.config_manager.get_global_setting("openai", "api_key")
            store_type = vector_config.get("type", "qdrant")

            if store_type == "qdrant":
                from src.vector_store import QdrantStore
                qdrant_config = vector_config.get("qdrant", {})
                self.vector_store = QdrantStore(
                    host=qdrant_config.get("host", "qdrant"),
                    port=qdrant_config.get("port", 6333),
                    openai_api_key=openai_key,
                )
                logger.info(f"Qdrant vector store initialized at {qdrant_config.get('host', 'qdrant')}:{qdrant_config.get('port', 6333)}")
            elif store_type == "chromadb":
                from src.vector_store import ChromaDBStore
                self.vector_store = ChromaDBStore(
                    path=vector_config.get("path", "./storage/vectordb"),
                    openai_api_key=openai_key,
                )
                logger.info(f"ChromaDB vector store initialized at {vector_config.get('path', './storage/vectordb')}")
            else:
                logger.error(f"Unsupported vector store type: {store_type}")
                raise ValueError(f"Unsupported vector store type: {store_type}")

            self.memory_manager = CodeMemoryManager(self.vector_store, self.gitlab_client)
            logger.debug("Code memory manager initialized")

            # Initialize OpenAI analyzer
            logger.debug("Initializing OpenAI analyzer...")
            self.analyzer = OpenAIAnalyzer(
                api_key=openai_key,
                model=self.config_manager.get_global_setting("openai", "model"),
                vector_store=self.vector_store,
                max_tokens=self.config_manager.get_global_setting("openai", "max_tokens"),
            )
            logger.info(f"OpenAI analyzer initialized with model {self.config_manager.get_global_setting('openai', 'model')}")

            # Initialize MR handler
            logger.debug("Initializing merge request handler...")
            self.mr_handler = MergeRequestHandler(
                self.gitlab_client, self.analyzer, self.memory_manager, self.config_manager
            )
            logger.debug("Merge request handler initialized")

            init_duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Initialization complete in {init_duration:.2f} seconds")

        except Exception as e:
            logger.error(f"Failed to initialize GitLab AI Reviewer: {e}", exc_info=True)
            raise

    async def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down GitLab AI Reviewer...")
        # Add any cleanup code here

    def create_app(self) -> FastAPI:
        """Create FastAPI application"""

        # Initialize config manager early for app setup
        if self.config_manager is None:
            self.config_manager = ConfigManager()

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self.initialize()
            yield
            await self.shutdown()

        app = FastAPI(
            title="Merge Mind", version="1.2.0", lifespan=lifespan
        )

        @app.post("/webhook", response_model=WebhookResponse)
        async def gitlab_webhook(request: Request, background_tasks: BackgroundTasks):
            """Handle GitLab webhook events"""

            # Verify webhook secret if configured
            webhook_secret = self.config_manager.get_global_setting(
                "gitlab", "webhook_secret"
            )
            if webhook_secret:
                provided_token = request.headers.get("X-Gitlab-Token")
                if provided_token != webhook_secret:
                    logger.warning("Invalid webhook token provided")
                    raise HTTPException(status_code=401, detail="Invalid webhook token")

            # Parse webhook data
            try:
                webhook_data = await request.json()
                logger.info(f"Received webhook event: {webhook_data.get('object_kind', 'unknown')}")
            except Exception as e:
                logger.error(f"Failed to parse webhook data: {e}")
                raise HTTPException(status_code=400, detail="Invalid JSON payload")

            # Process webhook
            try:
                result = await self.mr_handler.process_webhook(webhook_data)
                logger.info(f"Webhook processed successfully: {result}")
                return WebhookResponse(**result)
            except Exception as e:
                logger.error(f"Failed to process webhook: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Webhook processing failed")

        @app.post("/review/{project_id}/{mr_iid}", response_model=ReviewResponse)
        async def trigger_review(project_id: int, mr_iid: int):
            """Manually trigger a review"""
            logger.info(f"Manual review triggered for project {project_id}, MR {mr_iid}")

            # Start metrics tracking
            review_id = metrics_collector.start_review(project_id, mr_iid)

            try:
                result = await self.mr_handler.review_merge_request(project_id, mr_iid)

                # Complete metrics tracking
                files_reviewed = result.get("files_reviewed", 0)
                issues_found = result.get("issues_found", 0)
                metrics_collector.complete_review(review_id, files_reviewed, issues_found)

                # Ensure required fields are present for ReviewResponse
                result.setdefault("project_id", project_id)
                result.setdefault("mr_iid", mr_iid)
                result.setdefault("started_at", datetime.now().isoformat())

                logger.info(f"Review completed for project {project_id}, MR {mr_iid}")
                return ReviewResponse(**result)
            except Exception as e:
                # Fail metrics tracking
                metrics_collector.fail_review(review_id, str(e))

                logger.error(f"Failed to trigger review for project {project_id}, MR {mr_iid}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Review trigger failed")

        @app.post("/learn/{project_id}", response_model=LearningResponse)
        async def learn_repository(project_id: int, branch: str = "main"):
            """Learn from a repository

            Args:
                project_id: GitLab project ID
                branch: Branch to learn from (query parameter, defaults to 'main')
            """
            logger.info(f"Learning triggered for project {project_id}, branch {branch}")

            project_config = self.config_manager.get_project_config(project_id)
            if not project_config:
                logger.warning(f"Project {project_id} not configured")
                raise HTTPException(status_code=404, detail="Project not configured")

            # Start metrics tracking
            learning_id = metrics_collector.start_learning(project_config.name)

            try:
                stats = await self.memory_manager.learn_from_repository(
                    project_id,
                    project_config.name,
                    branch,
                    included_extensions=project_config.included_extensions,
                    excluded_paths=project_config.excluded_paths
                )

                # Complete metrics tracking
                files_processed = stats.get("files_processed", 0)
                chunks_created = stats.get("total_chunks", 0)
                metrics_collector.complete_learning(learning_id, files_processed, chunks_created)

                # Add additional metadata
                stats["project_id"] = project_id
                stats["branch"] = branch
                stats["status"] = "success"
                stats["duration_seconds"] = self.memory_manager.learning_stats.get(
                    project_config.name, {}
                ).get("duration_seconds", 0)

                logger.info(f"Learning completed for project {project_id}: {stats}")
                return LearningResponse(**stats)

            except Exception as e:
                # Fail metrics tracking
                metrics_collector.fail_learning(learning_id, str(e))

                logger.error(f"Learning failed for project {project_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Learning operation failed")

        @app.get("/health", response_model=HealthCheckResponse)
        async def health_check():
            """Basic health check endpoint"""
            return HealthCheckResponse(
                status="healthy",
                service="merge-mind",
                timestamp=datetime.now().isoformat()
            )

        @app.get("/health/detailed", response_model=DetailedHealthCheckResponse)
        async def detailed_health_check():
            """Detailed health check for all dependencies"""
            health_status = {
                "status": "healthy",
                "service": "merge-mind",
                "timestamp": datetime.now().isoformat(),
                "checks": {}
            }

            # Check configuration
            try:
                config_healthy = bool(
                    self.config_manager.get_global_setting("gitlab", "url") and
                    self.config_manager.get_global_setting("gitlab", "token") and
                    self.config_manager.get_global_setting("openai", "api_key")
                )
                health_status["checks"]["configuration"] = {
                    "status": "healthy" if config_healthy else "unhealthy",
                    "details": f"{len(self.config_manager.projects)} projects configured"
                }
            except Exception as e:
                health_status["checks"]["configuration"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }

            # Check GitLab connectivity
            try:
                if self.gitlab_client and hasattr(self.gitlab_client, 'current_user'):
                    health_status["checks"]["gitlab"] = {
                        "status": "healthy",
                        "details": f"Connected as {self.gitlab_client.current_user.username}"
                    }
                else:
                    health_status["checks"]["gitlab"] = {
                        "status": "unhealthy",
                        "error": "GitLab client not initialized"
                    }
            except Exception as e:
                health_status["checks"]["gitlab"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }

            # Check vector store
            try:
                if self.vector_store:
                    # Try a simple operation to test connectivity
                    health_status["checks"]["vector_store"] = {
                        "status": "healthy",
                        "details": f"Vector store initialized ({type(self.vector_store).__name__})"
                    }
                else:
                    health_status["checks"]["vector_store"] = {
                        "status": "unhealthy",
                        "error": "Vector store not initialized"
                    }
            except Exception as e:
                health_status["checks"]["vector_store"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }

            # Check OpenAI connectivity
            try:
                if self.analyzer:
                    health_status["checks"]["openai"] = {
                        "status": "healthy",
                        "details": f"Analyzer initialized with model {self.analyzer.model}"
                    }
                else:
                    health_status["checks"]["openai"] = {
                        "status": "unhealthy",
                        "error": "OpenAI analyzer not initialized"
                    }
            except Exception as e:
                health_status["checks"]["openai"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }

            # Determine overall status
            unhealthy_checks = [check for check in health_status["checks"].values() if check["status"] == "unhealthy"]
            if unhealthy_checks:
                health_status["status"] = "unhealthy"
                logger.warning(f"Health check failed: {len(unhealthy_checks)} unhealthy components")

            return DetailedHealthCheckResponse(**health_status)

        @app.get("/stats", response_model=StatsResponse)
        async def get_stats():
            """Get service statistics"""
            stats = StatsResponse(
                projects_configured=len(self.config_manager.projects),
                active_reviews=len(self.mr_handler.active_reviews) if self.mr_handler else 0,
                learning_stats=self.memory_manager.learning_stats if self.memory_manager else {},
                cache_size=len(self.analyzer.review_cache) if self.analyzer else 0,
            )
            return stats

        @app.get("/metrics")
        async def get_metrics():
            """Get comprehensive metrics including detailed learning statistics"""
            summary_stats = metrics_collector.get_summary_stats()
            recent_activity = metrics_collector.get_recent_activity(limit=10)
            circuit_breaker_stats = circuit_breaker_registry.get_all_stats()

            # Get vector store statistics for each project
            vector_store_stats = {}
            if self.vector_store:
                for project_id, project_config in self.config_manager.projects.items():
                    collection_name = f"gitlab_code_{project_config.name}"
                    try:
                        if hasattr(self.vector_store, 'get_collection_stats'):
                            stats = self.vector_store.get_collection_stats(collection_name)
                            if stats:
                                vector_store_stats[project_config.name] = stats
                    except Exception as e:
                        logger.debug(f"Could not get stats for {collection_name}: {e}")

            # Build detailed learning statistics
            learning_details = {
                "total_operations": summary_stats.get("total_learning_operations", 0),
                "successful_operations": summary_stats.get("successful_learning_operations", 0),
                "failed_operations": summary_stats.get("failed_learning_operations", 0),
                "average_time_seconds": summary_stats.get("average_learning_time_seconds", 0),
                "recent_learning": recent_activity.get("recent_learning", []),
                "vector_store_stats": vector_store_stats
            }

            return {
                "summary": summary_stats,
                "recent_activity": {
                    "reviews": recent_activity.get("recent_reviews", []),
                    "learning": recent_activity.get("recent_learning", [])
                },
                "learning": learning_details,
                "circuit_breakers": circuit_breaker_stats,
                "performance": {
                    "embedding_cache_size": len(getattr(self.analyzer, '_embedding_cache', {})) if self.analyzer else 0,
                    "review_cache_size": len(self.analyzer.review_cache) if self.analyzer else 0,
                    "vector_store_collections": len(vector_store_stats)
                },
                "projects": {
                    "configured": len(self.config_manager.projects),
                    "with_learned_data": len(vector_store_stats)
                },
                "timestamp": datetime.now().isoformat(),
            }

        @app.get("/learning/stats", response_model=AllLearningStatsResponse)
        async def get_learning_stats():
            """Get detailed learning statistics for all projects"""
            projects_stats = []
            total_documents = 0

            # Get recent learning activity from metrics
            recent_activity = metrics_collector.get_recent_activity(limit=50)
            learning_by_project = {}

            for learning_op in recent_activity.get("recent_learning", []):
                project_name = learning_op["project_name"]
                if project_name not in learning_by_project:
                    learning_by_project[project_name] = []
                learning_by_project[project_name].append(learning_op)

            # Get stats for each configured project
            for project_id, project_config in self.config_manager.projects.items():
                collection_name = f"gitlab_code_{project_config.name}"

                # Get vector store stats
                doc_count = 0
                if self.vector_store and hasattr(self.vector_store, 'get_collection_stats'):
                    try:
                        stats = self.vector_store.get_collection_stats(collection_name)
                        doc_count = stats.get("document_count", 0)
                    except Exception as e:
                        logger.debug(f"Could not get stats for {collection_name}: {e}")

                # Get last learning operation
                last_learning = None
                project_learning_ops = learning_by_project.get(project_config.name, [])
                if project_learning_ops:
                    last_learning = project_learning_ops[0]  # Most recent

                project_stat = ProjectLearningStats(
                    project_id=project_id,
                    project_name=project_config.name,
                    collection_name=collection_name,
                    document_count=doc_count,
                    last_learning=last_learning,
                    total_learning_operations=len(project_learning_ops)
                )

                projects_stats.append(project_stat)
                total_documents += doc_count

            return AllLearningStatsResponse(
                projects=projects_stats,
                total_documents=total_documents,
                total_projects_learned=len([p for p in projects_stats if p.document_count > 0])
            )

        @app.get("/learning/stats/{project_id}", response_model=ProjectLearningStats)
        async def get_project_learning_stats(project_id: int):
            """Get learning statistics for a specific project"""
            project_config = self.config_manager.get_project_config(project_id)
            if not project_config:
                raise HTTPException(status_code=404, detail="Project not found")

            collection_name = f"gitlab_code_{project_config.name}"

            # Get vector store stats
            doc_count = 0
            if self.vector_store and hasattr(self.vector_store, 'get_collection_stats'):
                try:
                    stats = self.vector_store.get_collection_stats(collection_name)
                    doc_count = stats.get("document_count", 0)
                except Exception as e:
                    logger.warning(f"Could not get stats for {collection_name}: {e}")

            # Get learning operations for this project
            recent_activity = metrics_collector.get_recent_activity(limit=50)
            project_learning_ops = [
                op for op in recent_activity.get("recent_learning", [])
                if op["project_name"] == project_config.name
            ]

            last_learning = project_learning_ops[0] if project_learning_ops else None

            return ProjectLearningStats(
                project_id=project_id,
                project_name=project_config.name,
                collection_name=collection_name,
                document_count=doc_count,
                last_learning=last_learning,
                total_learning_operations=len(project_learning_ops)
            )

        @app.post("/reload", response_model=ReloadResponse)
        async def reload_configuration():
            """Reload configuration from environment variables"""
            try:
                # Store old configuration for comparison
                old_config = {
                    "openai_model": self.config_manager.get_global_setting("openai", "model"),
                    "openai_api_key": self.config_manager.get_global_setting("openai", "api_key"),
                    "webhook_secret": self.config_manager.get_global_setting("gitlab", "webhook_secret"),
                }

                # Reload configuration
                config_changes = self.config_manager.reload()

                # Get new configuration
                new_config = {
                    "openai_model": self.config_manager.get_global_setting("openai", "model"),
                    "openai_api_key": self.config_manager.get_global_setting("openai", "api_key"),
                    "webhook_secret": self.config_manager.get_global_setting("gitlab", "webhook_secret"),
                }

                # Update analyzer for hot-reloadable changes
                changes = {}
                if old_config["openai_model"] != new_config["openai_model"]:
                    self.analyzer.update_model(new_config["openai_model"])
                    changes["openai_model"] = {"old": old_config["openai_model"], "new": new_config["openai_model"]}

                if old_config["openai_api_key"] != new_config["openai_api_key"]:
                    self.analyzer.update_api_key(new_config["openai_api_key"])
                    changes["openai_api_key"] = {"old": "***", "new": "***"}  # Don't log actual API keys

                if old_config["webhook_secret"] != new_config["webhook_secret"]:
                    changes["webhook_secret"] = {"old": "***", "new": "***"}  # Don't log secrets

                # Add any other detected changes
                for key, change in config_changes.items():
                    if key not in ["openai.model", "openai.api_key", "gitlab.webhook_secret"]:
                        changes[key] = change

                message = f"Configuration reloaded successfully. {len(changes)} changes applied."
                logger.info(message)

                return ReloadResponse(
                    success=True,
                    message=message,
                    changes=changes
                )

            except Exception as e:
                error_msg = f"Failed to reload configuration: {str(e)}"
                logger.error(error_msg)
                return ReloadResponse(
                    success=False,
                    message=error_msg,
                    changes={}
                )

        # Project Management Endpoints
        @app.get("/projects", response_model=ProjectsResponse)
        async def get_projects():
            """Get all configured projects"""
            projects = []
            for project_id, project_config in self.config_manager.projects.items():
                projects.append(ProjectResponse(
                    project_id=project_config.project_id,
                    name=project_config.name,
                    description=project_config.description,
                    review_enabled=project_config.review_enabled,
                    review_drafts=project_config.review_drafts,
                    min_lines_changed=project_config.min_lines_changed,
                    max_files_per_review=project_config.max_files_per_review,
                    excluded_paths=project_config.excluded_paths,
                    included_extensions=project_config.included_extensions,
                    custom_prompts=project_config.custom_prompts,
                    review_model=project_config.review_model,
                    team_preferences=project_config.team_preferences,
                ))
            return ProjectsResponse(projects=projects)

        @app.get("/projects/{project_id}", response_model=ProjectResponse)
        async def get_project(project_id: int):
            """Get a specific project by ID"""
            project_config = self.config_manager.get_project_config(project_id)
            if not project_config:
                raise HTTPException(status_code=404, detail="Project not found")

            return ProjectResponse(
                project_id=project_config.project_id,
                name=project_config.name,
                description=project_config.description,
                review_enabled=project_config.review_enabled,
                review_drafts=project_config.review_drafts,
                min_lines_changed=project_config.min_lines_changed,
                max_files_per_review=project_config.max_files_per_review,
                excluded_paths=project_config.excluded_paths,
                included_extensions=project_config.included_extensions,
                custom_prompts=project_config.custom_prompts,
                review_model=project_config.review_model,
                team_preferences=project_config.team_preferences,
            )

        @app.post("/projects", response_model=ProjectResponse)
        async def create_project(project: ProjectResponse):
            """Create a new project configuration"""
            if project.project_id in self.config_manager.projects:
                raise HTTPException(status_code=409, detail="Project already exists")

            # Create ProjectConfig from ProjectResponse
            project_config = ProjectConfig(
                project_id=project.project_id,
                name=project.name,
                description=project.description or "",
                review_enabled=project.review_enabled,
                review_drafts=project.review_drafts,
                min_lines_changed=project.min_lines_changed,
                max_files_per_review=project.max_files_per_review,
                excluded_paths=project.excluded_paths,
                included_extensions=project.included_extensions,
                custom_prompts=project.custom_prompts or {},
                review_model=project.review_model or "gpt-4-turbo-preview",
                team_preferences=project.team_preferences or [],
            )

            self.config_manager.projects[project.project_id] = project_config
            logger.info(f"Created project {project.project_id}: {project.name}")

            return project

        @app.put("/projects/{project_id}", response_model=ProjectResponse)
        async def update_project(project_id: int, updates: ProjectResponse):
            """Update an existing project configuration"""
            if project_id != updates.project_id:
                raise HTTPException(status_code=400, detail="Project ID mismatch")

            if project_id not in self.config_manager.projects:
                raise HTTPException(status_code=404, detail="Project not found")

            # Update the project config
            project_config = self.config_manager.projects[project_id]
            project_config.name = updates.name
            project_config.description = updates.description or ""
            project_config.review_enabled = updates.review_enabled
            project_config.review_drafts = updates.review_drafts
            project_config.min_lines_changed = updates.min_lines_changed
            project_config.max_files_per_review = updates.max_files_per_review
            project_config.excluded_paths = updates.excluded_paths
            project_config.included_extensions = updates.included_extensions
            project_config.custom_prompts = updates.custom_prompts or {}
            project_config.review_model = updates.review_model or "gpt-4-turbo-preview"
            project_config.team_preferences = updates.team_preferences or []

            logger.info(f"Updated project {project_id}: {updates.name}")

            return updates

        @app.delete("/projects/{project_id}")
        async def delete_project(project_id: int):
            """Delete a project configuration"""
            if project_id not in self.config_manager.projects:
                raise HTTPException(status_code=404, detail="Project not found")

            del self.config_manager.projects[project_id]
            logger.info(f"Deleted project {project_id}")

            return {"message": f"Project {project_id} deleted successfully"}

        # Review Management Endpoints
        @app.get("/reviews/active", response_model=ActiveReviewsResponse)
        async def get_active_reviews():
            """Get all active reviews"""
            active_reviews = []
            if self.mr_handler:
                # Get active review keys and parse them
                for review_key in self.mr_handler.active_reviews.keys():
                    try:
                        # Assuming review_key format is "project_id_mr_iid"
                        project_id_str, mr_iid_str = review_key.split("_")
                        project_id = int(project_id_str)
                        mr_iid = int(mr_iid_str)

                        active_reviews.append(ActiveReviewItem(
                            id=review_key,
                            project_id=project_id,
                            mr_iid=mr_iid,
                            status="in_progress",
                            started_at=datetime.now().isoformat(),  # Approximate
                        ))
                    except (ValueError, AttributeError):
                        logger.warning(f"Invalid review key format: {review_key}")
                        continue

            return ActiveReviewsResponse(active_reviews=active_reviews)

        @app.get("/reviews/history", response_model=ReviewHistoryResponse)
        async def get_review_history(project_id: Optional[int] = None, limit: int = 10):
            """Get review history with optional filtering"""
            # For now, return empty list as we don't have persistent history storage
            # In a real implementation, this would query a database or metrics store
            reviews = []

            # If we had metrics data, we could populate this
            # For now, return empty to avoid errors

            return ReviewHistoryResponse(reviews=reviews)

        self.app = app
        return app


# Global app instance for uvicorn import
app_instance = GitLabReviewerApp()
app = app_instance.create_app()


# Entry point
def main():
    """Main entry point"""
    logger.info("Starting GitLab AI Reviewer main function")

    # Get server configuration
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8080"))
    workers = int(os.getenv("SERVER_WORKERS", "1"))  # Changed to 1 to avoid import string requirement

    # Run the server
    reload_mode = os.getenv("DEV_MODE", "false").lower() == "true"
    uvicorn.run("src.main:app", host=host, port=port, workers=workers, log_level="info", reload=reload_mode)


if __name__ == "__main__":
    main()
