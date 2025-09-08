# DocFoundry Installation and Deployment Guide

This guide covers the complete installation and deployment process for DocFoundry, including all security, observability, and data lineage features.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Development Setup](#development-setup)
4. [Production Deployment](#production-deployment)
5. [Database Configuration](#database-configuration)
6. [Security Configuration](#security-configuration)
7. [Observability Setup](#observability-setup)
8. [Data Lineage Configuration](#data-lineage-configuration)
9. [Environment Variables](#environment-variables)
10. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 10GB available space
- **OS**: Linux, macOS, or Windows with WSL2
- **Python**: 3.9 or higher
- **Docker**: 20.10 or higher (for containerized deployment)
- **Docker Compose**: 2.0 or higher

### Recommended Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 50GB+ SSD
- **Network**: Stable internet connection for initial setup

### Database Requirements
- **SQLite**: Built-in (development)
- **PostgreSQL**: 12+ (production recommended)
- **MySQL**: 8.0+ (alternative production option)

## Quick Start

Get DocFoundry running in under 5 minutes:

```bash
# Clone the repository
git clone https://github.com/your-org/docfoundry.git
cd docfoundry

# Start with Docker Compose (recommended)
docker-compose up -d

# Or start development server
make dev
```

Access the application at `http://localhost:8000`

## Development Setup

### 1. Clone and Setup Environment

```bash
# Clone repository
git clone https://github.com/your-org/docfoundry.git
cd docfoundry

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
vim .env  # or your preferred editor
```

### 3. Database Setup

```bash
# Initialize database
make db-init

# Run migrations
make db-migrate

# Seed test data (optional)
make db-seed
```

### 4. Start Development Server

```bash
# Start all services
make dev

# Or start individual components
make api      # API server only
make worker   # Background workers only
make ui       # Frontend only
```

### 5. Verify Installation

```bash
# Run health checks
make health-check

# Run tests
make test

# Check API endpoints
curl http://localhost:8000/health
```

## Production Deployment

### Option 1: Docker Compose (Recommended)

#### 1. Prepare Production Environment

```bash
# Clone repository on production server
git clone https://github.com/your-org/docfoundry.git
cd docfoundry

# Copy production environment template
cp ops/docker/.env.prod.template .env.prod

# Edit production configuration
vim .env.prod
```

#### 2. Configure Production Settings

```bash
# Set production environment variables
export ENVIRONMENT=production
export DATABASE_URL=postgresql://user:pass@db:5432/docfoundry
export REDIS_URL=redis://redis:6379/0
export SECRET_KEY=your-super-secret-key
```

#### 3. Deploy with Docker Compose

```bash
# Start production stack
docker-compose -f ops/docker/docker-compose.prod.yml up -d

# Verify deployment
docker-compose -f ops/docker/docker-compose.prod.yml ps

# Check logs
docker-compose -f ops/docker/docker-compose.prod.yml logs -f
```

#### 4. Initialize Production Database

```bash
# Run database migrations
docker-compose -f ops/docker/docker-compose.prod.yml exec api python -m alembic upgrade head

# Create admin user (optional)
docker-compose -f ops/docker/docker-compose.prod.yml exec api python scripts/create_admin.py
```

### Option 2: Manual Deployment

#### 1. System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3-pip postgresql-client redis-tools nginx

# CentOS/RHEL
sudo yum install -y python39 python39-pip postgresql redis nginx

# macOS
brew install python@3.9 postgresql redis nginx
```

#### 2. Application Setup

```bash
# Create application user
sudo useradd -m -s /bin/bash docfoundry
sudo su - docfoundry

# Clone and setup application
git clone https://github.com/your-org/docfoundry.git
cd docfoundry
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 3. Database Setup

```bash
# Create PostgreSQL database
sudo -u postgres createdb docfoundry
sudo -u postgres createuser docfoundry
sudo -u postgres psql -c "ALTER USER docfoundry WITH PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE docfoundry TO docfoundry;"

# Run migrations
DATABASE_URL=postgresql://docfoundry:secure_password@localhost/docfoundry python -m alembic upgrade head
```

#### 4. Process Management

```bash
# Install and configure systemd services
sudo cp ops/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable docfoundry-api docfoundry-worker
sudo systemctl start docfoundry-api docfoundry-worker
```

#### 5. Reverse Proxy Setup

```bash
# Configure Nginx
sudo cp ops/nginx/docfoundry.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/docfoundry.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Database Configuration

### SQLite (Development)

```bash
# Default configuration - no additional setup required
DATABASE_URL=sqlite:///./data/docfoundry.db
```

### PostgreSQL (Production)

#### 1. Installation

```bash
# Ubuntu/Debian
sudo apt install -y postgresql postgresql-contrib

# CentOS/RHEL
sudo yum install -y postgresql-server postgresql-contrib
sudo postgresql-setup initdb

# macOS
brew install postgresql
brew services start postgresql
```

#### 2. Database Setup

```sql
-- Connect as postgres user
sudo -u postgres psql

-- Create database and user
CREATE DATABASE docfoundry;
CREATE USER docfoundry WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE docfoundry TO docfoundry;

-- Enable required extensions
\c docfoundry
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
```

#### 3. Configuration

```bash
# Set database URL
DATABASE_URL=postgresql://docfoundry:your_secure_password@localhost:5432/docfoundry

# Connection pool settings
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
DB_POOL_TIMEOUT=30
```

### Database Migrations

```bash
# Check current migration status
python -m alembic current

# Run pending migrations
python -m alembic upgrade head

# Create new migration (development)
python -m alembic revision --autogenerate -m "Description of changes"

# Rollback migration
python -m alembic downgrade -1
```

## Security Configuration

### 1. API Security

```bash
# Generate secure secret key
SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Configure API security
API_KEY_HEADER=X-API-Key
API_RATE_LIMIT=100/minute
API_BURST_LIMIT=200/10seconds

# Enable CORS (adjust origins as needed)
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_ALLOW_HEADERS=Content-Type,Authorization,X-API-Key
```

### 2. Rate Limiting

```bash
# Redis for rate limiting (recommended)
REDIS_URL=redis://localhost:6379/1

# Rate limiting configuration
RATE_LIMIT_STORAGE=redis
RATE_LIMIT_DEFAULT=100/hour
RATE_LIMIT_LOGIN=5/minute
RATE_LIMIT_API=1000/hour
```

### 3. Input Validation

```bash
# Enable strict input validation
VALIDATION_STRICT=true
VALIDATION_MAX_FILE_SIZE=100MB
VALIDATION_ALLOWED_EXTENSIONS=pdf,doc,docx,txt,md

# Content security
CSP_ENABLED=true
CSP_POLICY="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
```

### 4. HTTPS Configuration

```bash
# SSL/TLS settings
SSL_ENABLED=true
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem

# Force HTTPS redirect
FORCE_HTTPS=true
HSTS_MAX_AGE=31536000
```

## Observability Setup

### 1. Logging Configuration

```bash
# Logging settings
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/docfoundry/app.log
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=5

# Structured logging
LOG_STRUCTURED=true
LOG_INCLUDE_TRACE_ID=true
```

### 2. Metrics and Monitoring

```bash
# Prometheus metrics
METRICS_ENABLED=true
METRICS_PORT=9090
METRICS_PATH=/metrics

# Performance monitoring
PERF_MONITORING_ENABLED=true
PERF_RESPONSE_TIME_THRESHOLD=2000
PERF_ERROR_RATE_THRESHOLD=0.05
```

### 3. OpenTelemetry Setup

```bash
# OpenTelemetry configuration
OTEL_ENABLED=true
OTEL_SERVICE_NAME=docfoundry
OTEL_SERVICE_VERSION=1.0.0
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:14268/api/traces

# Tracing settings
OTEL_TRACE_SAMPLING_RATE=0.1
OTEL_TRACE_INCLUDE_DB=true
OTEL_TRACE_INCLUDE_HTTP=true
```

### 4. Health Checks

```bash
# Health check configuration
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL=30
HEALTH_CHECK_TIMEOUT=10
HEALTH_CHECK_ENDPOINTS=/health,/ready,/live
```

## Data Lineage Configuration

### 1. Lineage Tracking

```bash
# Enable data lineage
LINEAGE_ENABLED=true
LINEAGE_RETENTION_DAYS=90
LINEAGE_CLEANUP_INTERVAL=24h

# Content tracking
LINEAGE_TRACK_CONTENT_CHANGES=true
LINEAGE_CONTENT_HASH_ALGORITHM=sha256
LINEAGE_TRACK_DEPENDENCIES=true
```

### 2. Incremental Processing

```bash
# Incremental processing settings
INCREMENTAL_PROCESSING_ENABLED=true
INCREMENTAL_BATCH_SIZE=100
INCREMENTAL_PROCESSING_INTERVAL=1h

# Change detection
INCREMENTAL_CHANGE_DETECTION=content_hash
INCREMENTAL_FORCE_REPROCESS_DAYS=30
```

### 3. Reprocessing Configuration

```bash
# Reprocessing settings
REPROCESSING_ENABLED=true
REPROCESSING_QUEUE_SIZE=1000
REPROCESSING_WORKER_COUNT=4
REPROCESSING_RETRY_ATTEMPTS=3
REPROCESSING_RETRY_DELAY=300
```

## Environment Variables

### Core Application

```bash
# Application settings
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-super-secret-key
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

# Server configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4
WORKER_CLASS=uvicorn.workers.UvicornWorker
```

### Database

```bash
# Database connection
DATABASE_URL=postgresql://user:pass@host:5432/dbname
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
DB_POOL_TIMEOUT=30
DB_ECHO=false
```

### Redis/Caching

```bash
# Redis configuration
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600
CACHE_MAX_SIZE=1000
```

### Security

```bash
# API Security
API_KEY_HEADER=X-API-Key
API_RATE_LIMIT=100/minute
JWT_SECRET_KEY=jwt-secret-key
JWT_EXPIRATION=3600

# CORS
CORS_ORIGINS=https://yourdomain.com
CORS_ALLOW_CREDENTIALS=true

# SSL/TLS
SSL_ENABLED=true
FORCE_HTTPS=true
```

### Observability

```bash
# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/docfoundry/app.log

# Metrics
METRICS_ENABLED=true
METRICS_PORT=9090

# Tracing
OTEL_ENABLED=true
OTEL_SERVICE_NAME=docfoundry
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:14268/api/traces
```

### Data Lineage

```bash
# Lineage tracking
LINEAGE_ENABLED=true
LINEAGE_RETENTION_DAYS=90
LINEAGE_CONTENT_HASH_ALGORITHM=sha256

# Incremental processing
INCREMENTAL_PROCESSING_ENABLED=true
INCREMENTAL_BATCH_SIZE=100
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Issues

```bash
# Check database connectivity
psql $DATABASE_URL -c "SELECT 1;"

# Verify database exists
psql $DATABASE_URL -c "\l"

# Check migrations status
python -m alembic current
```

#### 2. Permission Issues

```bash
# Fix file permissions
sudo chown -R docfoundry:docfoundry /path/to/docfoundry
sudo chmod -R 755 /path/to/docfoundry

# Fix log directory permissions
sudo mkdir -p /var/log/docfoundry
sudo chown docfoundry:docfoundry /var/log/docfoundry
```

#### 3. Memory Issues

```bash
# Check memory usage
free -h
ps aux | grep docfoundry

# Adjust worker count
export WORKERS=2  # Reduce if low memory

# Configure database pool
export DB_POOL_SIZE=10
export DB_MAX_OVERFLOW=20
```

#### 4. Network Issues

```bash
# Check port availability
sudo netstat -tlnp | grep :8000

# Test API connectivity
curl -I http://localhost:8000/health

# Check firewall settings
sudo ufw status
sudo iptables -L
```

### Performance Tuning

#### 1. Database Optimization

```sql
-- PostgreSQL performance tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
SELECT pg_reload_conf();
```

#### 2. Application Tuning

```bash
# Increase worker count for high load
export WORKERS=8

# Enable connection pooling
export DB_POOL_SIZE=50
export DB_MAX_OVERFLOW=100

# Configure caching
export CACHE_TTL=7200
export CACHE_MAX_SIZE=10000
```

#### 3. System Optimization

```bash
# Increase file descriptor limits
echo "docfoundry soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "docfoundry hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize kernel parameters
echo "net.core.somaxconn = 65535" | sudo tee -a /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65535" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### Monitoring and Alerts

#### 1. Health Check Endpoints

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health?detailed=true

# Readiness check
curl http://localhost:8000/ready

# Liveness check
curl http://localhost:8000/live
```

#### 2. Log Analysis

```bash
# View recent logs
tail -f /var/log/docfoundry/app.log

# Search for errors
grep -i error /var/log/docfoundry/app.log

# Analyze performance
grep "response_time" /var/log/docfoundry/app.log | tail -100
```

#### 3. Metrics Collection

```bash
# Check Prometheus metrics
curl http://localhost:9090/metrics

# Monitor key metrics
curl -s http://localhost:9090/metrics | grep -E "(http_requests_total|response_time_seconds)"
```

### Backup and Recovery

#### 1. Database Backup

```bash
# Create database backup
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Automated backup script
#!/bin/bash
BACKUP_DIR="/var/backups/docfoundry"
mkdir -p $BACKUP_DIR
pg_dump $DATABASE_URL | gzip > $BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).sql.gz
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
```

#### 2. Application Data Backup

```bash
# Backup application data
tar -czf docfoundry_data_$(date +%Y%m%d).tar.gz /path/to/docfoundry/data

# Backup configuration
cp .env .env.backup.$(date +%Y%m%d)
```

#### 3. Recovery Process

```bash
# Restore database
psql $DATABASE_URL < backup_20240115_120000.sql

# Restore application data
tar -xzf docfoundry_data_20240115.tar.gz -C /

# Restart services
sudo systemctl restart docfoundry-api docfoundry-worker
```

## Next Steps

After successful installation:

1. **Configure Security**: Review and customize security settings
2. **Set Up Monitoring**: Configure observability and alerting
3. **Test Data Lineage**: Verify lineage tracking functionality
4. **Performance Tuning**: Optimize for your specific workload
5. **Backup Strategy**: Implement regular backup procedures
6. **Documentation**: Customize documentation for your team

For additional help:
- Check the [API Documentation](api/lineage.md)
- Review [Security Configuration](security.md)
- See [Monitoring Setup Guide](monitoring.md)
- Visit our [GitHub Issues](https://github.com/your-org/docfoundry/issues)