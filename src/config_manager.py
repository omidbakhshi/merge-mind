"""
Configuration management for GitLab AI Reviewer
Handles both global settings and project-specific configurations
Location: src/config_manager.py
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProjectConfig:
    """Configuration for a specific GitLab project"""

    project_id: int
    name: str
    description: str = ""
    review_enabled: bool = True
    auto_review_on_open: bool = True
    review_drafts: bool = False
    min_lines_changed: int = 10
    max_files_per_review: int = 50
    excluded_paths: List[str] = field(default_factory=lambda: ["vendor/", "node_modules/", "dist/", "build/"])
    included_extensions: List[str] = field(default_factory=lambda: [".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c"])
    custom_prompts: Dict[str, str] = field(default_factory=dict)
    review_model: str = "gpt-4-turbo-preview"
    team_preferences: List[str] = field(default_factory=list)




class ConfigManager:
    """Manages all configuration for the AI reviewer"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.global_config: Dict[str, Any] = {}
        self.projects: Dict[int, ProjectConfig] = {}
        self.load_configurations()

    def load_configurations(self):
        """Load all configuration files"""
        logger.info(f"Loading configurations from {self.config_dir}")

        # Load global configuration
        global_config_path = self.config_dir / "config.yaml"
        if global_config_path.exists():
            try:
                with open(global_config_path, "r") as f:
                    self.global_config = yaml.safe_load(f)
                    self._substitute_env_vars(self.global_config)
                    logger.info(f"Loaded global config from {global_config_path}")
                    logger.debug(f"Global config keys: {list(self.global_config.keys())}")
            except Exception as e:
                logger.error(f"Failed to load global config from {global_config_path}: {e}")
                raise
        else:
            logger.warning(f"Global config not found at {global_config_path}, creating default config")
            self._create_default_config()

        # Load project configurations
        projects_config_path = self.config_dir / "projects.yaml"
        if projects_config_path.exists():
            try:
                with open(projects_config_path, "r") as f:
                    projects_data = yaml.safe_load(f)
                    self._substitute_env_vars(projects_data)
                    for project_data in projects_data.get("projects", []):
                        project = ProjectConfig(**project_data)
                        self.projects[project.project_id] = project
                    logger.info(f"Loaded {len(self.projects)} project configurations")
                    logger.debug(f"Project IDs: {list(self.projects.keys())}")
            except Exception as e:
                logger.error(f"Failed to load project config from {projects_config_path}: {e}")
                raise
        else:
            logger.warning(f"Projects config not found at {projects_config_path}")

    def _substitute_env_vars(self, config: Dict[str, Any]) -> None:
        """Recursively substitute environment variables in config"""
        for key, value in config.items():
            if isinstance(value, dict):
                self._substitute_env_vars(value)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                config[key] = os.getenv(env_var, value)

    def _create_default_config(self):
        """Create default configuration"""
        self.global_config = {
            "gitlab": {
                "url": os.getenv("GITLAB_URL", "https://gitlab.example.com"),
                "token": os.getenv("GITLAB_TOKEN", ""),
                "webhook_secret": os.getenv("GITLAB_WEBHOOK_SECRET", ""),
            },
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "model": "gpt-4-turbo-preview",
                "max_tokens": 2000,
                "temperature": 0.3,
            },
            "vector_store": {
                "type": "chromadb",  # or 'qdrant', 'pinecone'
                "path": "./storage/vectordb",
                "collection_prefix": "gitlab_code_",
            },
            "server": {"host": "0.0.0.0", "port": 8080, "workers": 4},
            "review": {
                "max_comment_length": 1000,
                "batch_size": 5,  # Number of files to review at once
                "context_lines": 10,  # Lines of context around changes
                "severity_levels": ["critical", "major", "minor", "suggestion"],
            },
            "learning": {
                "update_interval_hours": 24,
                "min_stars_for_learning": 0,  # Min stars on a file for it to be learned
                "max_file_size_kb": 500,
            },
        }

    def get_project_config(self, project_id: int) -> Optional[ProjectConfig]:
        """Get configuration for a specific project"""
        return self.projects.get(project_id)

    def get_global_setting(self, *keys: str, default: Any = None) -> Any:
        """Get a global setting using dot notation"""
        value = self.global_config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    def is_file_reviewable(
        self, project_id: int, file_path: str, lines_changed: int
    ) -> bool:
        """Check if a file should be reviewed based on project config"""
        project = self.get_project_config(project_id)
        if not project or not project.review_enabled:
            return False

        # Check minimum lines changed
        if lines_changed < project.min_lines_changed:
            return False

        # Check excluded paths
        for excluded in project.excluded_paths:
            if excluded in file_path:
                return False

        # Check included extensions
        file_ext = Path(file_path).suffix
        if file_ext not in project.included_extensions:
            return False

        return True

    def get_review_prompt_template(self, project_id: int, prompt_type: str) -> str:
        """Get custom prompt template for a project"""
        project = self.get_project_config(project_id)
        if project and prompt_type in project.custom_prompts:
            return project.custom_prompts[prompt_type]

        # Type assertion for mypy
        assert project is not None or True  # Always true, but helps mypy

        # Return default prompts
        default_prompts = {
            "code_review": """You are an expert code reviewer. Review the following code changes:

File: {file_path}
Language: {language}
Context: {context}

Code Diff:
{diff}

Provide feedback on:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Security concerns
5. Suggestions for improvement

Use the team's existing code patterns from the knowledge base when relevant.""",
            "security_review": """Focus on security implications of these code changes:
{diff}

Check for: SQL injection, XSS, authentication issues, data exposure, etc.""",
            "performance_review": """Analyze performance implications of these changes:
{diff}

Consider: Time complexity, memory usage, database queries, caching opportunities.""",
        }

        return default_prompts.get(prompt_type, default_prompts["code_review"])

    def reload(self):
        """Reload configuration files"""
        logger.info("Reloading configuration files...")
        self.load_configurations()


# Example configuration files content

EXAMPLE_GLOBAL_CONFIG = """# config/config.yaml
gitlab:
  url: ${GITLAB_URL}
  token: ${GITLAB_TOKEN}
  webhook_secret: ${GITLAB_WEBHOOK_SECRET}

openai:
  api_key: ${OPENAI_API_KEY}
  model: gpt-4-turbo-preview
  max_tokens: 2000
  temperature: 0.3

vector_store:
  type: chromadb
  path: ./storage/vectordb
  collection_prefix: gitlab_code_

server:
  host: 0.0.0.0
  port: 8080
  workers: 4

review:
  max_comment_length: 1000
  batch_size: 5
  context_lines: 10
  severity_levels:
    - critical
    - major
    - minor
    - suggestion

learning:
  update_interval_hours: 24
  min_stars_for_learning: 0
  max_file_size_kb: 500
"""

EXAMPLE_PROJECTS_CONFIG = """# config/projects.yaml
projects:
  - project_id: 1
    name: "Backend API"
    review_enabled: true
    auto_review_on_open: true
    review_drafts: false
    min_lines_changed: 10
    max_files_per_review: 50
    excluded_paths:
      - vendor/
      - node_modules/
      - migrations/
    included_extensions:
      - .py
      - .go
      - .sql
    review_model: gpt-4-turbo-preview
    custom_prompts:
      code_review: |
        Review this backend API code focusing on:
        - RESTful design principles
        - Database query optimization
        - API security best practices
        - Error handling patterns

        Code: {diff}

  - project_id: 2
    name: "Frontend App"
    review_enabled: true
    auto_review_on_open: true
    review_drafts: true
    min_lines_changed: 5
    excluded_paths:
      - dist/
      - build/
      - public/assets/
    included_extensions:
      - .tsx
      - .ts
      - .jsx
      - .js
      - .css
      - .scss
    review_model: gpt-4-turbo-preview

  - project_id: 3
    name: "Data Pipeline"
    review_enabled: true
    auto_review_on_open: false  # Manual trigger only
    min_lines_changed: 20
    included_extensions:
      - .py
      - .sql
      - .yaml
    custom_prompts:
      code_review: |
        Review this data pipeline code for:
        - Data validation and error handling
        - Scalability concerns
        - SQL query performance
        - ETL best practices

        Changes: {diff}
"""
