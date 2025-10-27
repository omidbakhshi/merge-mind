#!/usr/bin/env python

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_manager import ConfigManager


def fetch_all_projects() -> List[Dict[str, Any]]:
    """Fetch all projects from GitLab"""

    # Load configuration
    config = ConfigManager()

    # Initialize GitLab client
    try:
        from src.gitlab_client import GitLabClient
        gitlab_client = GitLabClient(
            config.get_global_setting('gitlab', 'url'),
            config.get_global_setting('gitlab', 'token')
        )
    except Exception as e:
        print(f"❌ Failed to initialize GitLab client: {e}")
        print("Make sure GITLAB_URL and GITLAB_TOKEN are set in your .env file")
        return []

    print("🔍 Fetching projects from GitLab...")

    try:
        # Get all projects (this user has access to)
        # Note: This gets projects the user is a member of
        projects = gitlab_client.gl.projects.list(all=True, membership=True)

        project_configs = []

        for project in projects:
            print(f"📋 Found project: {project.name} (ID: {project.id})")

            # Create basic project configuration
            project_config = {
                'project_id': project.id,
                'name': project.name,
                'description': project.description or f"Project: {project.name}",
                'review_enabled': True,
                'auto_review_on_open': True,
                'review_drafts': False,
                'min_lines_changed': 10,
                'max_files_per_review': 50,
                'excluded_paths': [
                    'vendor/',
                    'node_modules/',
                    'dist/',
                    'build/',
                    '.git/',
                    '__pycache__/',
                    '*.lock',
                    '*.log'
                ],
                'included_extensions': [
                    '.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c',
                    '.cs', '.rb', '.php', '.swift', '.kotlin', '.scala',
                    '.sql', '.yaml', '.yml', '.json', '.xml', '.html', '.css'
                ],
                'review_model': 'gpt-3.5-turbo',  # Use cheaper model by default
                'custom_prompts': {},
                'team_preferences': []
            }

            project_configs.append(project_config)

        print(f"\n✅ Found {len(project_configs)} projects")
        return project_configs

    except Exception as e:
        print(f"❌ Failed to fetch projects: {e}")
        return []


def update_projects_config(projects: List[Dict[str, Any]]) -> bool:
    """Update the projects.yaml file with fetched projects"""

    if not projects:
        print("❌ No projects to update")
        return False

    config_path = Path(__file__).parent.parent / 'config' / 'projects.yaml'

    # Create new config structure
    config_data = {
        'projects': projects,
        'default': {
            'review_enabled': True,
            'auto_review_on_open': True,
            'review_drafts': False,
            'min_lines_changed': 10,
            'max_files_per_review': 50,
            'excluded_paths': [
                'vendor/',
                'node_modules/',
                'dist/',
                'build/'
            ],
            'included_extensions': [
                '.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c',
                '.cs', '.rb', '.php', '.swift', '.kotlin', '.scala',
                '.sql', '.yaml', '.yml', '.json', '.xml', '.html', '.css', '.scss'
            ],
            'review_model': 'gpt-3.5-turbo'
        }
    }

    try:
        # Write to file
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        print(f"✅ Updated {config_path} with {len(projects)} projects")
        return True

    except Exception as e:
        print(f"❌ Failed to update config file: {e}")
        return False


def main():
    print("🚀 GitLab Project Auto-Configuration Tool")
    print("=" * 50)

    # Fetch projects
    projects = fetch_all_projects()

    if not projects:
        print("❌ No projects found. Please check your GitLab credentials.")
        sys.exit(1)

    # Show summary
    print(f"\n📊 Summary:")
    print(f"   Total projects found: {len(projects)}")
    print("\n📋 Projects:")
    for i, project in enumerate(projects[:10], 1):  # Show first 10
        print(f"   {i}. {project['name']} (ID: {project['project_id']})")

    if len(projects) > 10:
        print(f"   ... and {len(projects) - 10} more")

    # Automatically update configuration
    print(f"\n🔄 Updating projects.yaml with {len(projects)} projects...")
    if update_projects_config(projects):
        print("\n🎉 Configuration updated successfully!")
        print("🔄 Restart your containers to load the new configuration:")
        print("   docker compose -f docker/docker-compose.yml up --build -d")
    else:
        print("\n❌ Failed to update configuration")
        sys.exit(1)


if __name__ == '__main__':
    main()