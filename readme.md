# GitLab AI Code Reviewer

An intelligent, self-learning code review bot for self-hosted GitLab instances. Uses OpenAI's GPT models to provide context-aware code reviews while learning from your team's coding patterns and preferences.

## 🌟 Features

- **🤖 AI-Powered Reviews**: Leverages OpenAI GPT-4 for intelligent code analysis
- **📚 Learning Capability**: Learns from your codebase and adapts to team patterns
- **🎯 Multi-Project Support**: Configure different rules for different projects
- **💬 Smart Comments**: Posts inline comments on specific issues with severity levels
- **🔄 CI/CD Integration**: Runs as part of your GitLab CI pipeline
- **📊 Qdrant Vector Database**: High-performance vector storage for code embeddings
- **🔐 Self-Hosted**: Runs on your infrastructure with your GitLab instance
- **📈 Continuous Improvement**: Learns from merged code and team feedback
- **🏠 Local Training**: Train on local codebases without GitLab API dependencies
- **🔌 Circuit Breaker Protection**: Automatic failure detection and recovery for external services
- **⚡ Performance Optimized**: Smart caching, async operations, and batch processing
- **📊 Advanced Monitoring**: Comprehensive metrics, health checks, and performance tracking
- **🛡️ Enterprise Reliability**: Request validation, structured logging, and graceful degradation

## 🆕 Recent Updates

### v1.2.0 - Enterprise-Grade Reliability & Performance
- **🔌 Circuit Breaker Protection**: Automatic failure detection and recovery for external APIs (OpenAI, GitLab, Qdrant)
- **⚡ Performance Optimizations**: Smart caching, async operations, and optimized batch processing
- **📊 Advanced Monitoring**: Comprehensive metrics collection, health checks, and performance tracking
- **🛡️ Enhanced Reliability**: Request validation, structured logging, and graceful error handling
- **📈 CLI Progress Indicators**: Real-time progress bars for long-running operations
- **🔍 Health Check Endpoints**: Detailed service health monitoring with dependency checks

### v1.1.0 - Qdrant Integration & Local Training
- **🔄 Vector Store Migration**: Switched from ChromaDB to Qdrant for better performance and scalability
- **🏠 Local Codebase Training**: New capability to train on local codebases without GitLab API
- **🐳 Improved Docker Setup**: Enhanced container configuration with automatic Qdrant service
- **🔧 Configuration Enhancements**: Environment variable substitution and better config validation
- **🧪 Testing Tools**: Added scripts for verifying training results and search functionality

## 📋 Requirements

- Python 3.11+
- Self-hosted GitLab instance (12.0+)
- OpenAI API key
- Qdrant vector database (included in Docker setup)
- 4GB+ RAM recommended (8GB+ for large codebases)
- 10GB+ disk space for vector database
- Redis (optional, for enhanced caching)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourorg/gitlab-ai-reviewer.git
cd gitlab-ai-reviewer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your values
```

### 2. Configure Your Projects

> **🔒 Security Note:** Never commit sensitive configuration files to version control:
> - `config/projects.yaml` (contains project IDs and settings)
> - `.env` (contains API keys and secrets)
> - Use `config/projects.yaml.example` as a template
> - Store secrets in environment variables or `.env` files (add `.env` to `.gitignore`)

#### Option A: Auto-Configure from GitLab (Recommended)
Automatically fetch all your GitLab projects:

```bash
# Run the auto-configuration script
docker compose -f docker/docker-compose.yml exec ai-reviewer \
  python scripts/fetch_gitlab_projects.py

# Restart containers to load the new configuration
docker compose -f docker/docker-compose.yml up --build -d
```

This will:
- Fetch all projects you have access to from GitLab
- Create default configurations for each project
- Update `config/projects.yaml` automatically

#### Option B: Manual Configuration
Copy the example configuration and edit `config/projects.yaml`:

```bash
cp config/projects.yaml.example config/projects.yaml
```

Then edit `config/projects.yaml` with your project details:

```yaml
projects:
  - project_id: 123  # Your GitLab project ID
    name: "backend-api"
    review_enabled: true
    auto_review_on_open: true
    min_lines_changed: 10
    excluded_paths:
      - vendor/
      - node_modules/
    included_extensions:
      - .py
      - .go
      - .sql
```

### 3. Initial Learning (Optional but Recommended)

Train the bot on your existing codebase:

#### Option A: Learn from GitLab Repository
```bash
python scripts/learn_codebase.py 123 --branch main
```

#### Option B: Learn from Local Codebase (Recommended)
```bash
# Mount your local codebase in Docker
echo "- /path/to/your/codebase:/app/my_codebase:ro" >> docker/docker-compose.yml

# Restart containers
docker compose -f docker/docker-compose.yml up --build -d

# Train on local codebase
docker compose -f docker/docker-compose.yml exec ai-reviewer \
  python scripts/learn_local_codebase.py my_project /app/my_codebase
```

### 4. Run the Service

```bash
# Development
python -m src.main

# Production with Docker
docker compose -f docker/docker-compose.yml up -d
```

### 5. Configure GitLab Webhook

In your GitLab project:
1. Go to Settings → Webhooks
2. Add webhook URL: `http://your-server:8080/webhook`
3. Secret token: (same as `GITLAB_WEBHOOK_SECRET` in .env)
4. Select triggers:
   - Merge request events
   - Comments (for feedback processing)

## 🔧 Configuration

### Environment Variables

```bash
# Required
GITLAB_URL=https://gitlab.yourcompany.com
GITLAB_TOKEN=glpat-xxxxxxxxxxxxx  # Personal access token with api scope
OPENAI_API_KEY=sk-xxxxxxxxxxxxx

# Optional
GITLAB_WEBHOOK_SECRET=your-secret
SERVER_PORT=8080
LOG_LEVEL=INFO
```

### Personal Access Token Permissions

Create a GitLab personal access token with these scopes:
- `api` - Full API access
- `read_repository` - Read repository content
- `write_repository` - Post review comments

### Project Configuration Options

Each project in `config/projects.yaml` (copied from `config/projects.yaml.example`) can have:

| Option | Description | Default |
|--------|-------------|---------|
| `review_enabled` | Enable/disable reviews | `true` |
| `auto_review_on_open` | Review on MR open | `true` |
| `review_drafts` | Review draft MRs | `false` |
| `min_lines_changed` | Minimum lines to trigger review | `10` |
| `max_files_per_review` | Max files to review at once | `50` |
| `excluded_paths` | Paths to ignore | `['vendor/', 'node_modules/']` |
| `included_extensions` | File types to review | `['.py', '.js', '.ts', ...]` |
| `review_model` | OpenAI model to use | `gpt-4-turbo-preview` |

## 🔄 CI/CD Integration

### Option 1: Include in Project CI

Add to your project's `.gitlab-ci.yml`:

```yaml
include:
  - remote: 'http://your-server:8080/ci/reviewer.yml'

variables:
  AI_REVIEW_ENABLED: "true"
  AI_REVIEWER_TOKEN: $CI_AI_REVIEWER_TOKEN  # Set in CI/CD variables
```

### Option 2: Direct Integration

```yaml
stages:
  - review

ai_code_review:
  stage: review
  script:
    - |
      curl -X POST \
        -H "Authorization: Bearer $AI_REVIEWER_TOKEN" \
        "http://your-server:8080/review/$CI_PROJECT_ID/$CI_MERGE_REQUEST_IID"
  only:
    - merge_requests
  allow_failure: true
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/webhook` | POST | GitLab webhook receiver |
| `/review/{project_id}/{mr_iid}` | POST | Manually trigger review |
| `/learn/{project_id}?branch=main` | POST | Learn from GitLab repository (specify branch) |
| `/health` | GET | Basic health check |
| `/health/detailed` | GET | Detailed health check with dependency status |
| `/stats` | GET | Service statistics |
| `/metrics` | GET | Comprehensive performance metrics and circuit breaker status |

## 🔌 Circuit Breaker Protection

The service includes automatic circuit breaker protection for all external API calls to prevent cascading failures:

### Circuit Breaker States
- **CLOSED**: Normal operation, requests flow through
- **OPEN**: Service detected as failing, requests blocked
- **HALF_OPEN**: Testing if service has recovered

### Protected Services
- **OpenAI API**: 3 failures → OPEN, 30s recovery timeout
- **GitLab API**: 5 failures → OPEN, 60s recovery timeout
- **Qdrant Vector DB**: 3 failures → OPEN, 15s recovery timeout

### Benefits
- **Zero Downtime**: Automatic failure detection and recovery
- **Cost Protection**: Prevents wasted API calls to failing services
- **User Experience**: Graceful error messages instead of timeouts

## ⚡ Performance Optimizations

### Smart Caching
- **Embedding Cache**: 500-item LRU cache for OpenAI embeddings (50-70% faster repeated queries)
- **Review Cache**: Prevents duplicate analysis of identical code changes
- **Connection Pooling**: Optimized HTTP connections for external APIs

### Async Operations
- **Concurrent Processing**: Up to 3 simultaneous OpenAI requests
- **Non-blocking I/O**: Async file operations and API calls
- **Batch Processing**: Optimized document insertion in configurable batches

### Resource Management
- **Memory Limits**: Automatic cleanup of old caches and metrics
- **Connection Limits**: Configurable connection pool sizes
- **Timeout Controls**: Intelligent timeouts for different operation types

## 📊 Monitoring & Observability

### Health Checks
```bash
# Basic health
curl http://localhost:8080/health

# Detailed health with dependency status
curl http://localhost:8080/health/detailed
```

### Performance Metrics
```bash
# Comprehensive metrics
curl http://localhost:8080/metrics
```

### Metrics Tracked
- **Review Performance**: Processing times, success rates, issue detection
- **Learning Operations**: Training duration, files processed, chunks created
- **Circuit Breaker Status**: Service health, failure rates, recovery attempts
- **Cache Performance**: Hit rates, memory usage, eviction counts

## 🛠️ Configuration & Training Scripts

### Auto-Configure Projects from GitLab
```bash
# Automatically fetch all your GitLab projects and create configurations
docker compose -f docker/docker-compose.yml exec ai-reviewer \
  python scripts/fetch_gitlab_projects.py

# This will update config/projects.yaml with all accessible projects
```

### Local Codebase Training
```bash
# Train on local codebase
docker compose -f docker/docker-compose.yml exec ai-reviewer \
  python scripts/learn_local_codebase.py <project_name> <local_path> [options]

# Options:
# --extensions .py .js .ts    # File extensions to include
# --help                     # Show help
```

### GitLab Repository Training
```bash
# Train on GitLab repository (command line)
python scripts/learn_codebase.py <project_id> --branch <branch>

# Train via API endpoint
curl -X POST "http://localhost:8080/learn/123?branch=develop"
```

### Testing Training Results
```bash
# Test search functionality
docker compose -f docker/docker-compose.yml exec ai-reviewer \
  python scripts/test_search.py
```

## 🧠 Learning and Adaptation

The bot learns in three ways:

1. **Initial Learning**: Analyzes your existing codebase
    ```bash
    # From GitLab repository (specify branch)
    python scripts/learn_codebase.py <project_id> --branch main
    # Or via API: curl -X POST "http://localhost:8080/learn/123?branch=main"

    # From local codebase (recommended)
    docker compose -f docker/docker-compose.yml exec ai-reviewer \
      python scripts/learn_local_codebase.py my_project /path/to/codebase
    ```

2. **Continuous Learning**: Automatically learns from merged MRs
    - Extracts patterns from successful merges
    - Updates knowledge base with new code

3. **Feedback Learning**: Adapts based on team feedback
    - Learns from resolved comments
    - Adjusts severity based on team responses

### Local Codebase Training

For the best results, train the bot on your local codebase:

```bash
# 1. Mount your codebase
echo "- /home/user/my-project:/app/my_codebase:ro" >> docker/docker-compose.yml

# 2. Restart containers
docker compose -f docker/docker-compose.yml up --build -d

# 3. Train the bot
docker compose -f docker/docker-compose.yml exec ai-reviewer \
  python scripts/learn_local_codebase.py my_project /app/my_codebase \
  --extensions .py .js .ts .java

# 4. Verify training worked
docker compose -f docker/docker-compose.yml exec ai-reviewer \
  python scripts/test_search.py
```

**Benefits of local training:**
- No GitLab API rate limits
- Access to private/local repositories
- Faster training process
- Better code context understanding

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start
docker compose -f docker/docker-compose.yml up -d

# View logs
docker compose -f docker/docker-compose.yml logs -f ai-reviewer

# Stop
docker compose -f docker/docker-compose.yml down
```

### Using Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gitlab-ai-reviewer
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ai-reviewer
  template:
    metadata:
      labels:
        app: ai-reviewer
    spec:
      containers:
      - name: reviewer
        image: gitlab-ai-reviewer:latest
        env:
        - name: GITLAB_URL
          valueFrom:
            secretKeyRef:
              name: reviewer-secrets
              key: gitlab-url
        - name: GITLAB_TOKEN
          valueFrom:
            secretKeyRef:
              name: reviewer-secrets
              key: gitlab-token
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: reviewer-secrets
              key: openai-key
        ports:
        - containerPort: 8080
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
          requests:
            memory: "2Gi"
            cpu: "1"
```

## 🔍 Monitoring

### Prometheus Metrics

The service exposes metrics at `/metrics`:
- `reviews_total` - Total reviews performed
- `review_duration_seconds` - Review processing time
- `openai_requests_total` - OpenAI API calls
- `learning_cycles_total` - Learning cycles completed

### Logging

Logs are written to `logs/` directory and stdout. Configure log level via `LOG_LEVEL` env var.

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_gitlab_client.py
```

### Code Quality

```bash
# Format code
black src/

# Lint
flake8 src/

# Type checking
mypy src/
```

### Adding New Language Support

1. Add language detection in `src/gitlab_client.py`
2. Add pattern extraction in `src/learning_engine.py`
3. Add language-specific prompts in `src/openai_analyzer.py`

## 📈 Performance Tuning

### Vector Store Configuration

The system uses **Qdrant** as the vector database backend:

- **Qdrant**: High-performance vector database, included in Docker setup
- **Automatic Setup**: Qdrant service is automatically started with Docker Compose
- **Persistent Storage**: Vector data persists across container restarts

#### Qdrant Configuration
```yaml
vector_store:
  type: qdrant
  qdrant:
    host: qdrant
    port: 6333
    collection_size: 1536  # OpenAI Ada-002 embedding dimension
```

### OpenAI Model Selection

| Model | Cost | Speed | Quality | Max Context |
|-------|------|-------|---------|-------------|
| gpt-4-turbo-preview | $$$ | Medium | Best | 128k tokens |
| gpt-4 | $$ | Slow | Excellent | 8k tokens |
| gpt-3.5-turbo | $ | Fast | Good | 16k tokens |

### Scaling Recommendations

- **Small Teams (<50 devs)**: Single instance, Qdrant (Docker), 4GB RAM
- **Medium Teams (50-200 devs)**: 2-4 instances, Qdrant cluster, 8GB RAM
- **Large Teams (200+ devs)**: Kubernetes, Qdrant cluster, 16GB+ RAM

## 🔒 Security Considerations

1. **API Keys**: Never commit keys to version control
2. **Network**: Use HTTPS for webhooks in production
3. **Access Control**: Limit GitLab token permissions
4. **Data Privacy**: Vector database contains code snippets
5. **Rate Limiting**: Implement rate limits for API endpoints

## 🐛 Troubleshooting

### Common Issues

**Bot not commenting on MRs:**
- Check webhook configuration
- Verify GitLab token has correct permissions
- Check logs: `docker compose -f docker/docker-compose.yml logs ai-reviewer`

**OpenAI rate limits:**
- Reduce `max_files_per_review` in config
- Increase retry delays
- Consider using multiple API keys

**High memory usage:**
- Reduce vector store cache size
- Limit concurrent reviews
- Use external vector database

**Learning not working:**
- For GitLab learning: Ensure GitLab token can read repository
- For local learning: Ensure codebase is properly mounted in Docker
- Check file size limits in config (500KB default)
- Verify OpenAI API key has embedding access
- Check Qdrant service is running: `docker compose -f docker/docker-compose.yml logs qdrant`

## 📝 License

MIT License - see LICENSE file

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 💬 Support

- Issues: [GitHub Issues](https://github.com/yourorg/gitlab-ai-reviewer/issues)
- Documentation: [Wiki](https://github.com/yourorg/gitlab-ai-reviewer/wiki)
- Email: ai-reviewer@yourcompany.com

## 🙏 Acknowledgments

- OpenAI for GPT models
- GitLab for the excellent API
- Qdrant for high-performance vector storage
- The open source community

---

# Testing Documentation

## tests/test_gitlab_client.py

```python
"""
Test suite for GitLab client
Location: tests/test_gitlab_client.py
"""

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
        mock_mr.created_at = "2024-01-01T00:00:00Z"
        mock_mr.updated_at = "2024-01-02T00:00:00Z"
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
```

## tests/test_analyzer.py

```python
"""
Test suite for OpenAI analyzer
Location: tests/test_analyzer.py
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import json
from src.openai_analyzer import OpenAIAnalyzer, CodeReviewResult, AnalysisContext


class TestOpenAIAnalyzer:
    """Test OpenAI analyzer functionality"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        with patch('src.openai_analyzer.openai'):
            return OpenAIAnalyzer(
                api_key="test_key",
                model="gpt-4-turbo-preview",
                max_tokens=2000
            )
    
    def test_token_counting(self, analyzer):
        """Test token counting functionality"""
        text = "This is a test string for counting tokens."
        token_count = analyzer.count_tokens(text)
        assert token_count > 0
        assert isinstance(token_count, int)
    
    def test_cache_key_generation(self, analyzer):
        """Test cache key generation"""
        file_path = "test.py"
        diff = "def test():\n    pass"
        
        key1 = analyzer._get_cache_key(file_path, diff)
        key2 = analyzer._get_cache_key(file_path, diff)
        key3 = analyzer._get_cache_key("other.py", diff)
        
        assert key1 == key2  # Same input = same key
        assert key1 != key3  # Different input = different key
    
    @pytest.mark.asyncio
    async def test_analyze_file_diff(self, analyzer):
        """Test file diff analysis"""
        # Mock OpenAI response
        mock_response = json.dumps({
            "reviews": [
                {
                    "severity": "minor",
                    "line_number": 10,
                    "message": "Consider using more descriptive variable names",
                    "suggestion": "Use 'user_count' instead of 'uc'",
                    "confidence": 0.8
                }
            ],
            "summary": "Code looks good with minor suggestions"
        })
        
        with patch.object(analyzer, '_call_openai', 
                         new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            context = AnalysisContext(
                project_name="test_project",
                merge_request_title="Add feature",
                merge_request_description="New feature",
                target_branch="main",
                similar_code_examples=[],
                project_patterns=[],
                recent_reviews=[]
            )
            
            results = await analyzer.analyze_file_diff(
                "test.py",
                "+def test():\n+    uc = 0\n+    return uc",
                context,
                "python"
            )
            
            assert len(results) == 1
            assert results[0].severity == "minor"
            assert results[0].line_number == 10
            assert "variable names" in results[0].message
    
    def test_parse_review_response(self, analyzer):
        """Test parsing of OpenAI responses"""
        response = json.dumps({
            "reviews": [
                {
                    "severity": "critical",
                    "line_number": 5,
                    "message": "SQL injection vulnerability",
                    "suggestion": "Use parameterized queries",
                    "confidence": 0.95
                }
            ]
        })
        
        results = analyzer._parse_review_response(response, "db.py")
        
        assert len(results) == 1
        assert results[0].severity == "critical"
        assert results[0].file_path == "db.py"
        assert "SQL injection" in results[0].message
    
    def test_extract_code_chunks(self, analyzer):
        """Test code chunk extraction from diff"""
        diff = """
@@ -1,5 +1,8 @@
 def existing():
     pass

+def new_function():
+    x = 1
+    y = 2
+    return x + y
+
@@ -10,3 +13,6 @@
 class Existing:
     pass
+
+class NewClass:
+    def __init__(self):
+        self.value = 0
"""
        
        chunks = analyzer._extract_code_chunks(diff)
        
        assert len(chunks) > 0
        assert "new_function" in chunks[0]
        assert any("NewClass" in chunk for chunk in chunks)
```

---

# setup.py

```python
"""
Setup configuration for GitLab AI Code Reviewer
Location: setup.py
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gitlab-ai-reviewer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@company.com",
    description="AI-powered code review bot for GitLab",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/gitlab-ai-reviewer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Quality Assurance",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gitlab-reviewer=src.main:main",
            "learn-codebase=scripts.learn_codebase:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
```