## Production Considerations

### Observability
- Structured logging with request IDs
- Latency metrics per component
- Error rate tracking
- Embedding cache hit rate

### API Design
- Versioned routes (`/v1/query`)
- Rate limiting (100 req/min default)
- Health endpoint with dependency checks
- Graceful degradation on LLM timeout

### Deployment
- Containerized with Docker
- Horizontal scaling via load balancer
- Secrets via environment variables
- Zero-downtime deployments

### Monitoring Alerts
- P95 latency > 5s
- Error rate > 1%
- Embedding API failures
- Vector store disk usage > 80%
