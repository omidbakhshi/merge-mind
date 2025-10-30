# Merge Mind

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
- **🔄 Hot Reload Support**: Update AI model and API keys without restarting the service
- **🎨 Web Dashboard**: Modern React-based dashboard for monitoring and management

## 📋 Requirements

- Python 3.11+
- Self-hosted GitLab instance (12.0+)
- OpenAI API key
- Qdrant vector database (included in Docker setup)
- Docker & Docker Compose (recommended)
- 4GB+ RAM recommended (8GB+ for large codebases)
- 10GB+ disk space for vector database
- Redis (optional, for enhanced caching)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/omidbakhshi/merge-mind.git
cd merge-mind

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your values (GITLAB_URL, GITLAB_TOKEN, OPENAI_API_KEY)
```

### 2. Configure Your Projects

> **🔒 Security Note:** Never commit sensitive configuration files to version control:
> - `config/projects.yaml` (contains project IDs and settings)
> - `.env` (contains API keys and secrets)
> - Use `config/projects.yaml.example` as a template
> - Store secrets in environment variables or `.env` files

#### Option A: Auto-Configure from GitLab (Recommended)

Automatically fetch all your GitLab projects:

```bash
# Start containers first
docker compose up -d

# Run the auto-configuration script
docker compose exec ai-reviewer python scripts/fetch_gitlab_projects.py

# Restart containers to load the new configuration
docker compose up -d --build
```

This will:
- Fetch all projects you have access to from GitLab
- Create default configurations for each project
- Update `config/projects.yaml` automatically

#### Option B: Manual Configuration

Copy the example configuration and edit:

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
docker compose exec ai-reviewer python scripts/learn_codebase.py 123 --branch main
```

#### Option B: Learn from Local Codebase (Recommended)

```bash
# Mount your local codebase
echo "- /path/to/your/codebase:/app/my_codebase:ro" >> docker-compose.yml

# Restart containers
docker compose up -d --build

# Train on local codebase
docker compose exec ai-reviewer \
  python scripts/learn_local_codebase.py my_project /app/my_codebase \
  --extensions .py .js .ts .java
```

### 4. Run the Service

```bash
# Using Docker Compose (Recommended)
docker compose up -d

# View logs
docker compose logs -f ai-reviewer

# Check health
curl http://localhost:8080/health
```

### 5. Access the Dashboard

Open your browser and navigate to:
- **Dashboard**: `http://localhost:8000` (via Nginx proxy)
- **API**: `http://localhost:8000/api`
- **Direct API**: `http://localhost:8080` (if not using proxy)

### 6. Configure GitLab Webhook

In your GitLab project:

1. Go to **Settings → Webhooks**
2. Add webhook URL: `http://your-server:8080/webhook`
3. Secret token: (same as `GITLAB_WEBHOOK_SECRET` in .env)
4. Select triggers:
   - ✅ Merge request events
   - ✅ Comments (for feedback processing)
5. Click **Add webhook**

## 🔧 Configuration

### Environment Variables

Required settings in `.env`:

```bash
# GitLab Configuration
GITLAB_URL=https://gitlab.yourcompany.com
GITLAB_TOKEN=glpat-xxxxxxxxxxxxx
GITLAB_WEBHOOK_SECRET=your-secret-token

# OpenAI Configuration
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
OPENAI_MODEL=gpt-4-turbo-preview

# Server Configuration
SERVER_PORT=8080
SERVER_WORKERS=4
LOG_LEVEL=INFO

# Dashboard
DASHBOARD_PORT=3000
NGINX_PORT=8000
```

### Hot Reload Configuration

The service supports **hot reloading** of certain configurations without restart:

```bash
# Change OpenAI model dynamically
export OPENAI_MODEL=gpt-4
curl -X POST http://localhost:8080/reload

# Response shows configuration changes
{
  "success": true,
  "message": "Configuration reloaded successfully. 1 changes applied.",
  "changes": {
    "openai_model": {
      "old": "gpt-4-turbo-preview",
      "new": "gpt-4"
    }
  }
}
```

**Supported hot reload settings:**
- `OPENAI_MODEL` - Change AI model for code reviews
- `OPENAI_API_KEY` - Update OpenAI API key
- `GITLAB_WEBHOOK_SECRET` - Update webhook validation secret

**Settings requiring service restart:**
- `GITLAB_URL`, `GITLAB_TOKEN` - GitLab connection settings
- `SERVER_HOST`, `SERVER_PORT`, `SERVER_WORKERS` - Server configuration
- Vector store configuration

### Personal Access Token Permissions

Create a GitLab personal access token with these scopes:
- `api` - Full API access
- `read_repository` - Read repository content
- `write_repository` - Post review comments

### Project Configuration Options

Each project in `config/projects.yaml` can have:

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
| `custom_prompts` | Custom review prompts | `{}` |
| `team_preferences` | Team-specific preferences | `[]` |

## 🔄 CI/CD Integration

### Option 1: Include in Project CI

Add to your project's `.gitlab-ci.yml`:

```yaml
include:
  - remote: 'http://your-server:8080/ci/reviewer.yml'

variables:
  AI_REVIEW_ENABLED: "true"
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

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/webhook` | POST | GitLab webhook receiver |
| `/review/{project_id}/{mr_iid}` | POST | Manually trigger review |
| `/learn/{project_id}?branch=main` | POST | Learn from repository |
| `/reload` | POST | Reload configuration (hot reload) |

### Monitoring Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic health check |
| `/health/detailed` | GET | Detailed health with dependencies |
| `/stats` | GET | Service statistics |
| `/metrics` | GET | Comprehensive metrics & circuit breakers |

### Project Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/projects` | GET | List all configured projects |
| `/projects/{id}` | GET | Get specific project |
| `/projects` | POST | Create new project |
| `/projects/{id}` | PUT | Update project |
| `/projects/{id}` | DELETE | Delete project |

### Review Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reviews/active` | GET | Get active reviews |
| `/reviews/history` | GET | Get review history |

## 🔌 Circuit Breaker Protection

Automatic circuit breaker protection for all external API calls prevents cascading failures:

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

- **Embedding Cache**: 500-item LRU cache for OpenAI embeddings (50-70% faster)
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

## 🛠️ Training Scripts

### Auto-Configure Projects from GitLab

```bash
# Automatically fetch all your GitLab projects
docker compose exec ai-reviewer python scripts/fetch_gitlab_projects.py

# This updates config/projects.yaml with all accessible projects
```

### Local Codebase Training

```bash
# Train on local codebase
docker compose exec ai-reviewer \
  python scripts/learn_local_codebase.py <project_name> <local_path> [options]

# Options:
# --extensions .py .js .ts    # File extensions to include

# Example:
docker compose exec ai-reviewer \
  python scripts/learn_local_codebase.py my_api /app/my_codebase \
  --extensions .py .go .sql
```

### GitLab Repository Training

```bash
# Train on GitLab repository
docker compose exec ai-reviewer \
  python scripts/learn_codebase.py <project_id> --branch <branch>

# Example:
docker compose exec ai-reviewer \
  python scripts/learn_codebase.py 123 --branch develop

# Or via API:
curl -X POST "http://localhost:8080/learn/123?branch=main"
```

### Testing Training Results

```bash
# Test search functionality
docker compose exec ai-reviewer python scripts/test_search.py
```

### Local Training Benefits

- No GitLab API rate limits
- Access to private/local repositories
- Faster training process
- Better code context understanding

## 🧠 Learning and Adaptation

The bot learns in three ways:

### 1. Initial Learning

Analyzes your existing codebase to understand patterns:

```bash
# From GitLab repository
docker compose exec ai-reviewer \
  python scripts/learn_codebase.py <project_id> --branch main

# From local codebase (recommended)
docker compose exec ai-reviewer \
  python scripts/learn_local_codebase.py my_project /path/to/codebase
```

### 2. Continuous Learning

Automatically learns from merged MRs:
- Extracts patterns from successful merges
- Updates knowledge base with new code
- Adapts to evolving codebase

### 3. Feedback Learning

Adapts based on team feedback:
- Learns from resolved comments
- Adjusts severity based on team responses
- Improves accuracy over time

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start all services
docker compose up -d

# View logs
docker compose logs -f ai-reviewer

# View all service logs
docker compose logs -f

# Stop services
docker compose down

# Rebuild after changes
docker compose up -d --build
```

### Services Included

- **ai-reviewer**: Main application service
- **qdrant**: Vector database for embeddings
- **dashboard**: React-based web interface
- **nginx**: Reverse proxy for API and dashboard
- **prometheus** (optional): Metrics collection
- **grafana** (optional): Metrics visualization

### Using Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: merge-mind
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
        image: merge-mind:latest
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

## 📈 Performance Tuning

### Vector Store Configuration

The system uses **Qdrant** as the vector database backend:

```yaml
vector_store:
  type: qdrant
  qdrant:
    host: qdrant
    port: 6333
    collection_size: 1536  # OpenAI Ada-002 embedding dimension
```

**Features:**
- High-performance vector operations
- Persistent storage across container restarts
- Automatic setup with Docker Compose
- Optimized for similarity search

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
   - Use `.env` files (add to `.gitignore`)
   - Store in environment variables or secret managers

2. **Network**: Use HTTPS for webhooks in production
   - Configure SSL/TLS certificates
   - Use secure webhook secrets

3. **Access Control**: Limit GitLab token permissions
   - Only grant required scopes
   - Rotate tokens regularly

4. **Data Privacy**: Vector database contains code snippets
   - Secure Qdrant with authentication
   - Limit network access to trusted sources

5. **Rate Limiting**: Implement rate limits for API endpoints
   - Configured in `config/config.yaml`
   - Adjust based on your needs

## 🐛 Troubleshooting

### Common Issues

**Bot not commenting on MRs:**
- Check webhook configuration in GitLab
- Verify GitLab token has correct permissions
- Check logs: `docker compose logs ai-reviewer`
- Test webhook: `curl -X POST http://your-server:8080/health`

**OpenAI rate limits:**
- Reduce `max_files_per_review` in config
- Use `gpt-3.5-turbo` for less critical projects
- Implement request queuing
- Consider multiple API keys

**High memory usage:**
- Reduce vector store cache size
- Limit concurrent reviews
- Use external vector database
- Increase Docker memory limits

**Learning not working:**
- **For GitLab learning**: Ensure token can read repository
- **For local learning**: Ensure codebase is properly mounted
- Check file size limits (500KB default)
- Verify OpenAI API key has embedding access
- Check Qdrant: `docker compose logs qdrant`

**Circuit breaker issues:**
- Check metrics: `curl http://localhost:8080/metrics`
- Review circuit breaker states
- Adjust thresholds in configuration
- Monitor external service health

**Dashboard not loading:**
- Check nginx logs: `docker compose logs nginx`
- Verify CORS settings in `config/config.yaml`
- Ensure all services are running: `docker compose ps`
- Check network connectivity between services

## 🧪 Testing

### Running Tests

```bash
# Install development dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_gitlab_client.py

# Run integration tests
pytest tests/test_integration.py
```

### Test Structure

```
tests/
├── test_gitlab_client.py      # GitLab API client tests
├── test_analyzer.py            # OpenAI analyzer tests
├── test_circuit_breaker.py    # Circuit breaker tests
├── test_performance.py         # Performance optimization tests
└── test_integration.py         # End-to-end integration tests
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Development Setup

```bash
# Clone the repository
git clone https://github.com/omidbakhshi/merge-mind.git
cd merge-mind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up development environment
cp .env.example .env
# Edit .env with your development settings
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

### Submitting Changes

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure tests pass: `pytest`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Adding New Language Support

1. Add language detection in `src/gitlab_client.py`
2. Add pattern extraction in `src/learning_engine.py`
3. Add language-specific prompts in `src/openai_analyzer.py`
4. Add tests for the new language


### Why Support?

Your contributions help:
- ⚡ **Faster Development**: More time dedicated to new features
- 🐛 **Bug Fixes**: Quicker response to issues and bugs
- 📚 **Better Documentation**: Comprehensive guides and tutorials
- 🎯 **Community Support**: Help other users succeed
- 🚀 **New Features**: Bring your feature requests to life

### Other Ways to Help

- ⭐ **Star the Repository**: Show your appreciation on GitHub
- 🐛 **Report Bugs**: Help us identify and fix issues
- 💡 **Share Ideas**: Suggest new features and improvements
- 📖 **Improve Docs**: Contribute to documentation
- 🗣️ **Spread the Word**: Tell others about Merge Mind

Every contribution, no matter how small, makes a difference! Thank you for your support! 🙏

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for providing the GPT models that power the AI analysis
- **GitLab** for their excellent API and platform
- **Qdrant** for their high-performance vector database
- **FastAPI** for the modern, fast web framework
- **All contributors** who help improve Merge Mind

## 📞 Support & Contact

### Community
- [GitHub Issues](https://github.com/omidbakhshi/merge-mind/issues) - Bug reports and feature requests
- [Discussions](https://github.com/omidbakhshi/merge-mind/discussions) - Questions and community chat
- [Email](mailto:omid.bakhshi.dev@gmail.com) - Direct support

### Stay Updated
- Follow on GitHub for updates
- Star the repository to show your support
- Watch for release notifications

---

**Made with ❤️ by the Merge Mind community**

*Happy Reviewing! 🚀*