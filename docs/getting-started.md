---
layout: default
title: Getting Started
nav_order: 2
has_children: false
permalink: /docs/getting-started
---

# Getting Started with Merge Mind
{: .no_toc }

Get your AI-powered code review bot up and running in less than 10 minutes.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Prerequisites

Before you begin, ensure you have:

- **Self-hosted GitLab instance** (version 12.0+)
- **OpenAI API key** ([Get one here](https://platform.openai.com/api-keys))
- **Docker & Docker Compose** installed
- **4GB+ RAM** (8GB+ recommended for large codebases)
- **10GB+ disk space** for vector database

{: .warning }
> **Security Note:** Never commit API keys or secrets to version control. Always use environment variables or `.env` files.

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/omidbakhshi/merge-mind.git
cd merge-mind
```

---

## Step 2: Configure Environment Variables

Create your environment configuration:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# GitLab Configuration
GITLAB_URL=https://gitlab.yourcompany.com
GITLAB_TOKEN=glpat-xxxxxxxxxxxxx
GITLAB_WEBHOOK_SECRET=your-secret-token

# OpenAI Configuration
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
OPENAI_MODEL=gpt-4-turbo-preview

# Server Configuration (optional)
SERVER_PORT=8080
SERVER_WORKERS=4
```

### Getting Your GitLab Token

1. Go to **GitLab → User Settings → Access Tokens**
2. Create a new token with these scopes:
   - ✅ `api` - Full API access
   - ✅ `read_repository` - Read repository content
   - ✅ `write_repository` - Post review comments
3. Copy the token immediately (you won't see it again)

{: .tip }
> Store your token securely and rotate it regularly for security best practices.

---

## Step 3: Configure Your Projects

You have two options for configuring projects:

### Option A: Auto-Configure (Recommended)

Automatically fetch all your GitLab projects:

```bash
# Start containers
docker compose up -d

# Run auto-configuration
docker compose exec ai-reviewer python scripts/fetch_gitlab_projects.py

# Restart to apply configuration
docker compose up -d --build
```

This will:
- ✅ Fetch all accessible projects from GitLab
- ✅ Create default configurations
- ✅ Update `config/projects.yaml` automatically

### Option B: Manual Configuration

Copy and edit the configuration file:

```bash
cp config/projects.yaml.example config/projects.yaml
```

Example configuration for a Laravel project:

```yaml
projects:
  - project_id: 123
    name: "backend-api"
    review_enabled: true
    auto_review_on_open: true
    min_lines_changed: 10
    max_files_per_review: 50
    excluded_paths:
      - vendor/
      - node_modules/
      - storage/
    included_extensions:
      - .php
      - .blade.php
      - .sql
    review_model: gpt-4-turbo-preview
```

[See full configuration options →](/docs/configuration)

---

## Step 4: Start the Services

Launch all services with Docker Compose:

```bash
docker compose up -d
```

This starts:
- **ai-reviewer** - Main application service (port 8080)
- **qdrant** - Vector database (ports 6333, 6334)
- **dashboard** - Web interface (port 3000)
- **nginx** - Reverse proxy (port 8000)

### Verify Services

Check that all services are running:

```bash
docker compose ps
```

Expected output:
```
NAME                 STATUS
merge-mind           Up (healthy)
qdrant               Up
merge-mind-dashboard Up
merge-mind-proxy     Up
```

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f ai-reviewer
```

---

## Step 5: Access the Dashboard

Open your browser and navigate to:

- **Dashboard**: [http://localhost:8000](http://localhost:8000)
- **API Docs**: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)
- **Health Check**: [http://localhost:8080/health](http://localhost:8080/health)

---

## Step 6: Configure GitLab Webhook

Connect Merge Mind to your GitLab projects:

1. Go to your GitLab project
2. Navigate to **Settings → Webhooks**
3. Add a new webhook:
   - **URL**: `http://your-server:8080/webhook`
   - **Secret Token**: (same as `GITLAB_WEBHOOK_SECRET` in .env)
   - **Trigger**: ✅ Merge request events
   - **SSL verification**: Enable if using HTTPS
4. Click **Add webhook**

{: .important }
> For production, use HTTPS and ensure your server is accessible from GitLab. Consider using a reverse proxy like nginx with SSL certificates.

### Test the Webhook

After adding the webhook:

1. Click **Test → Merge request events**
2. Check the webhook logs in GitLab
3. Verify in Merge Mind logs: `docker compose logs ai-reviewer`

You should see:
```
INFO - Received webhook event: merge_request
INFO - Webhook processed successfully
```

---

## Step 7: Train on Your Codebase (Optional)

Train Merge Mind on your existing code for better reviews:

### From Local Codebase (Recommended)

```bash
docker compose exec ai-reviewer \
  python scripts/learn_local_codebase.py my_project /path/to/codebase \
  --extensions .php .blade.php .vue .js .ts
```

### From GitLab Repository

```bash
docker compose exec ai-reviewer \
  python scripts/learn_codebase.py 123 --branch main
```

Training typically takes 5-15 minutes depending on codebase size.

[Learn more about training →](/docs/training)

---

## Step 8: Test Your First Review

Create a test merge request:

1. Create a new branch with some code changes
2. Open a merge request in GitLab
3. Watch as Merge Mind analyzes the code
4. Review the AI-generated comments

The bot will post:
- Inline comments on specific lines
- A comprehensive summary comment
- Severity ratings (critical, major, minor, suggestion)

---

## Quick Tips

{: .tip }
> **Start Small**: Enable reviews on 1-2 projects first to fine-tune settings before rolling out team-wide.

{: .tip }
> **Use Hot Reload**: Change OpenAI model without restarting:
> ```bash
> export OPENAI_MODEL=gpt-4
> curl -X POST http://localhost:8080/reload
> ```

{: .tip }
> **Monitor Performance**: Check metrics regularly:
> ```bash
> curl http://localhost:8080/metrics | jq
> ```

---

## Need Help?

- 🐛 [Report Issues](https://github.com/omidbakhshi/merge-mind/issues)
- 📧 [Email Support](mailto:omid.bakhshi.dev@gmail.com)

---
