# Docker Deployment

This directory contains Docker configuration for deploying the GitLab AI Code Reviewer.

## Quick Start

1. **Basic deployment** (AI Reviewer only):
   ```bash
   docker-compose up -d
   ```

2. **Full monitoring stack** (with Prometheus, Grafana):
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
   ```

## Services

### AI Reviewer (`ai-reviewer`)
- **Port**: 8080
- **Health Check**: `http://localhost:8080/health`
- **Metrics**: `http://localhost:8080/metrics`
- **Environment**: Requires GitLab and OpenAI API credentials

### Qdrant (`qdrant`)
- **Ports**: 6333 (HTTP), 6334 (gRPC)
- **Purpose**: Vector database for code embeddings

### Monitoring Stack (Development)

#### Prometheus (`prometheus`)
- **Port**: 9090
- **Purpose**: Metrics collection and storage
- **Config**: `monitoring/prometheus.yml`

#### Grafana (`grafana`)
- **Port**: 3000
- **Credentials**: admin/admin
- **Purpose**: Metrics visualization
- **Dashboards**: Pre-configured AI Reviewer dashboard

#### Redis (`redis`)
- **Port**: 6379
- **Purpose**: Optional caching layer

## Configuration

### Environment Variables

Set these in your `.env` file or docker-compose environment:

```bash
# GitLab
GITLAB_URL=https://your-gitlab-instance.com
GITLAB_TOKEN=your-gitlab-token
GITLAB_WEBHOOK_SECRET=your-webhook-secret

# OpenAI
OPENAI_API_KEY=your-openai-key

# Monitoring
MONITORING_METRICS_ENABLED=true
MONITORING_HEALTH_CHECK_ENABLED=true

# Circuit Breakers
CIRCUIT_BREAKER_OPENAI_FAILURE_THRESHOLD=3
CIRCUIT_BREAKER_GITLAB_FAILURE_THRESHOLD=5
```

### Volumes

- `vectordb_data`: Persistent storage for vector database
- `qdrant_data`: Qdrant database files
- `prometheus_data`: Prometheus metrics data
- `grafana_data`: Grafana dashboards and settings

## Health Checks

The container includes health checks that verify:
- Basic health endpoint (`/health`)
- Metrics endpoint (`/metrics`)

## Scaling

For production deployments:

1. **Increase resources** in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '4'
         memory: 8G
       reservations:
         cpus: '2'
         memory: 4G
   ```

2. **Use external databases** instead of container volumes

3. **Configure load balancing** for multiple instances

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in docker-compose.yml
2. **Memory issues**: Increase Docker memory limit
3. **Network issues**: Check `gitlab-network` connectivity

### Logs

View logs:
```bash
docker-compose logs ai-reviewer
docker-compose logs qdrant
```

### Debugging

Access container shell:
```bash
docker-compose exec ai-reviewer bash
```

## Security

- Change default Grafana password in production
- Use secrets management for API keys
- Configure firewall rules for exposed ports
- Regularly update base images