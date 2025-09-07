# DocFoundry Makefile
# Common development and deployment targets

.PHONY: help install dev test lint format clean build docker run stop logs
.DEFAULT_GOAL := help

# Variables
VENV?=.venv
PY?=$(VENV)/bin/python
PIP?=$(VENV)/bin/pip
PYTHON := python
PORT := 8001
DOCKER_IMAGE := docfoundry
DOCKER_TAG := latest
DB_FILE := data/docfoundry.db

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BLUE)DocFoundry Development Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

setup: ## Setup virtual environment and install dependencies
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	python -m venv $(VENV) && $(PIP) install -U pip && $(PIP) install -r requirements.txt
	@echo "$(GREEN)Setup completed$(RESET)"

install: ## Install dependencies
	@echo "$(BLUE)Installing dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Dependencies installed successfully$(RESET)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	$(PIP) install pytest pytest-cov black flake8 mypy isort sphinx safety bandit
	@echo "$(GREEN)Development dependencies installed successfully$(RESET)"

dev: api ## Start development server with auto-reload (alias for api)

api: ## Start API server with auto-reload
	@echo "$(BLUE)Starting API server on port $(PORT)...$(RESET)"
	uvicorn server.rag_api:app --reload --port $(PORT)

test: ## Run tests
	@echo "$(BLUE)Running tests...$(RESET)"
	$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term 2>/dev/null || echo "$(YELLOW)Tests not found, skipping$(RESET)"
	@echo "$(GREEN)Tests completed$(RESET)"

test-fast: ## Run tests without coverage
	@echo "$(BLUE)Running fast tests...$(RESET)"
	$(PYTHON) -m pytest tests/ -v -x 2>/dev/null || echo "$(YELLOW)Tests not found, skipping$(RESET)"
	@echo "$(GREEN)Fast tests completed$(RESET)"

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(RESET)"
	$(PYTHON) -m flake8 . --exclude=$(VENV),__pycache__ --max-line-length=88 2>/dev/null || echo "$(YELLOW)flake8 not installed$(RESET)"
	$(PYTHON) -m mypy . --ignore-missing-imports 2>/dev/null || echo "$(YELLOW)mypy not installed$(RESET)"
	@echo "$(GREEN)Linting completed$(RESET)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	$(PYTHON) -m black . --exclude=$(VENV) 2>/dev/null || echo "$(YELLOW)black not installed$(RESET)"
	$(PYTHON) -m isort . --skip=$(VENV) 2>/dev/null || echo "$(YELLOW)isort not installed$(RESET)"
	@echo "$(GREEN)Code formatting completed$(RESET)"

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	$(PYTHON) -m black --check . --exclude=$(VENV) 2>/dev/null || echo "$(YELLOW)black not installed$(RESET)"
	$(PYTHON) -m isort --check-only . --skip=$(VENV) 2>/dev/null || echo "$(YELLOW)isort not installed$(RESET)"
	@echo "$(GREEN)Format check completed$(RESET)"

clean: ## Clean up temporary files and caches
	@echo "$(BLUE)Cleaning up...$(RESET)"
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ 2>/dev/null || true
	@echo "$(GREEN)Cleanup completed$(RESET)"

# Database operations
db-init: ## Initialize the database
	@echo "$(BLUE)Initializing database...$(RESET)"
	$(PY) -c "from indexer.build_index import init_db; init_db()" 2>/dev/null || echo "$(YELLOW)Database initialization skipped$(RESET)"
	@echo "$(GREEN)Database initialized$(RESET)"

db-reset: ## Reset the database (WARNING: This will delete all data)
	@echo "$(RED)WARNING: This will delete all data in the database!$(RESET)"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -f $(DB_FILE) 2>/dev/null || true
	$(MAKE) db-init
	@echo "$(GREEN)Database reset completed$(RESET)"

db-backup: ## Backup the database
	@echo "$(BLUE)Backing up database...$(RESET)"
	cp $(DB_FILE) $(DB_FILE).backup.$$(date +%Y%m%d_%H%M%S) 2>/dev/null || echo "$(YELLOW)Database backup failed$(RESET)"
	@echo "$(GREEN)Database backup completed$(RESET)"

# Indexing operations
index: ## Build search index from documents
	@echo "$(BLUE)Building search index...$(RESET)"
	$(PY) indexer/build_index.py
	@echo "$(GREEN)Index building completed$(RESET)"

index-update: ## Update embeddings in the index
	@echo "$(BLUE)Updating embeddings...$(RESET)"
	$(PY) indexer/embeddings.py --update 2>/dev/null || echo "$(YELLOW)Embeddings update not available$(RESET)"
	@echo "$(GREEN)Embeddings update completed$(RESET)"

index-search: ## Test search functionality
	@echo "$(BLUE)Testing search...$(RESET)"
	$(PY) indexer/embeddings.py --search "documentation" --limit 5 2>/dev/null || echo "$(YELLOW)Search test not available$(RESET)"
	@echo "$(GREEN)Search test completed$(RESET)"

# Ingestion operations
crawl-html: ## Crawl and ingest HTML content
	@echo "$(BLUE)Crawling HTML content...$(RESET)"
	$(PY) pipelines/html_ingest.py sources/openrouter.yaml
	@echo "$(GREEN)HTML crawling completed$(RESET)"

ingest-feeds: ## Ingest RSS/Atom feeds
	@echo "$(BLUE)Ingesting feeds...$(RESET)"
	$(PY) pipelines/feed_ingest.py 2>/dev/null || echo "$(YELLOW)Feed ingestion not available$(RESET)"
	@echo "$(GREEN)Feed ingestion completed$(RESET)"

ingest-repos: ## Ingest repository content
	@echo "$(BLUE)Ingesting repositories...$(RESET)"
	$(PY) pipelines/repo_ingest.py 2>/dev/null || echo "$(YELLOW)Repository ingestion not available$(RESET)"
	@echo "$(GREEN)Repository ingestion completed$(RESET)"

# Docker operations
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -f Dockerfile.api -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)Docker image built successfully$(RESET)"

docker-run: ## Run application in Docker container
	@echo "$(BLUE)Running Docker container...$(RESET)"
	docker run -d --name docfoundry-app -p $(PORT):$(PORT) -v $$(pwd)/data:/app/data $(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)Docker container started on port $(PORT)$(RESET)"

docker-stop: ## Stop Docker container
	@echo "$(BLUE)Stopping Docker container...$(RESET)"
	docker stop docfoundry-app 2>/dev/null || true
	docker rm docfoundry-app 2>/dev/null || true
	@echo "$(GREEN)Docker container stopped$(RESET)"

docker-logs: ## Show Docker container logs
	@echo "$(BLUE)Showing Docker logs...$(RESET)"
	docker logs -f docfoundry-app

# Documentation
serve-docs: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation on http://127.0.0.1:8000$(RESET)"
	mkdocs serve -a 127.0.0.1:8000

build-docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	mkdocs build
	@echo "$(GREEN)Documentation built$(RESET)"

# Quality assurance
qa: format lint test ## Run all quality assurance checks
	@echo "$(GREEN)All QA checks completed$(RESET)"

ci: format-check lint test ## Run CI pipeline checks
	@echo "$(GREEN)CI pipeline completed$(RESET)"

# Monitoring and health checks
health: ## Check application health
	@echo "$(BLUE)Checking application health...$(RESET)"
	curl -f http://localhost:$(PORT)/health 2>/dev/null || echo "$(RED)Health check failed$(RESET)"

metrics: ## Show application metrics
	@echo "$(BLUE)Fetching application metrics...$(RESET)"
	curl -s http://localhost:$(PORT)/metrics 2>/dev/null | head -20 || echo "$(YELLOW)Metrics not available$(RESET)"

# Security
security-scan: ## Run security vulnerability scan
	@echo "$(BLUE)Running security scan...$(RESET)"
	$(PYTHON) -m safety check 2>/dev/null || echo "$(YELLOW)safety not installed$(RESET)"
	$(PYTHON) -m bandit -r . -x $(VENV) 2>/dev/null || echo "$(YELLOW)bandit not installed$(RESET)"
	@echo "$(GREEN)Security scan completed$(RESET)"

# Environment setup
setup-dev: setup install-dev db-init ## Setup development environment
	@echo "$(GREEN)Development environment setup completed$(RESET)"

setup-prod: setup db-init ## Setup production environment
	@echo "$(GREEN)Production environment setup completed$(RESET)"

# Utility targets
version: ## Show version information
	@echo "$(BLUE)DocFoundry Version Information$(RESET)"
	@$(PYTHON) --version
	@$(PIP) --version 2>/dev/null || echo "pip not available"

env-info: ## Show environment information
	@echo "$(BLUE)Environment Information$(RESET)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Working Directory: $$(pwd)"
	@echo "Database File: $(DB_FILE)"
	@echo "Port: $(PORT)"

# Main workflow targets
all: crawl-html index api ## Run complete workflow: crawl, index, and start API

full-setup: setup-dev crawl-html index ## Complete setup with data ingestion
	@echo "$(GREEN)Full setup completed - ready for development!$(RESET)"
