# DocFoundry

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)

A comprehensive **Documentation Intelligence System** that discovers, ingests, normalizes, indexes, and serves documentation from multiple sources. DocFoundry provides intelligent search capabilities, seamless integrations with development tools, and advanced analytics for documentation management.

## 🚀 Features

### Core Capabilities
- **🔍 Intelligent Search**: Full-text search with FTS5, semantic search with embeddings, and hybrid search modes
- **📚 Multi-Source Ingestion**: HTML crawling, Markdown processing, feed ingestion, and repository scanning
- **🎯 Smart Chunking**: Advanced document chunking with context preservation and metadata extraction
- **📊 Analytics & Monitoring**: Comprehensive observability with OpenTelemetry, performance metrics, and search analytics
- **🔌 Developer Integrations**: VS Code extension, Chrome extension, and MCP server support

### Architecture Components
- **📖 Documentation Site**: MkDocs-powered human-readable documentation (`mkdocs.yml`, `docs/`)
- **⚙️ Source Registry**: Configurable crawl rules and source definitions (`sources/*.yaml`)
- **🔄 Processing Pipelines**: Modular ingestion pipelines for various content types (`pipelines/`)
- **🗃️ Indexing Engine**: Flexible indexing with SQLite FTS5 or PostgreSQL + pgvector (`indexer/`)
- **🌐 RAG API**: FastAPI-based REST API with search, document retrieval, and capture endpoints (`server/`)
- **🛠️ Development Tools**: VS Code extension for in-editor search and Chrome extension for content capture
- **📋 Automation**: Comprehensive Makefile with development and deployment tasks

### Database Support
- **SQLite + FTS5**: Lightweight local development with full-text search
- **PostgreSQL + pgvector**: Production-ready with vector similarity search and advanced analytics
- **Seamless Migration**: Built-in migration tools and unified adapter interface

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- Git
- Optional: Docker for containerized deployment

### Installation

```bash
# Clone the repository
git clone https://github.com/codessa-prime/docfoundry.git
cd docfoundry

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build the initial index from the docs/ folder
python indexer/build_index.py

# Start the API server
uvicorn server.rag_api:app --reload --port 8001

# Optional: Preview documentation site
mkdocs serve -a 127.0.0.1:8000
```

### Using Make Commands

DocFoundry includes a comprehensive Makefile for common tasks:

```bash
# Show all available commands
make help

# Complete setup with development dependencies
make setup

# Run the full workflow: crawl, index, and start API
make all

# Start the API server
make api

# Build documentation
make docs

# Run tests
make test
```

## 📋 Usage Workflow

### 1. Configure Sources
Create or edit source configuration files in `sources/`:

```yaml
# sources/example.yaml
id: example-docs
name: Example Documentation
base_url: https://example.com/docs
start_urls:
  - https://example.com/docs/getting-started
allow_patterns:
  - "/docs/.*"
deny_patterns:
  - "/docs/internal/.*"
max_depth: 3
delay: 1.0
```

### 2. Ingest Content
```bash
# Crawl HTML content to Markdown
python pipelines/html_ingest.py sources/example.yaml

# Process feeds (RSS/Atom)
python pipelines/feed_ingest.py sources/blog.yaml

# Ingest repository content
python pipelines/repo_ingest.py sources/github-repo.yaml
```

### 3. Build Search Index
```bash
# Rebuild the search index
python indexer/build_index.py

# Or use make command
make index
```

### 4. Search and Query

#### REST API
```bash
# Basic search
curl -X POST http://localhost:8001/search \
  -H 'Content-Type: application/json' \
  -d '{"q": "authentication setup", "limit": 10}'

# Advanced search with filters
curl -X POST http://localhost:8001/search \
  -H 'Content-Type: application/json' \
  -d '{
    "q": "API configuration",
    "limit": 5,
    "source_filter": "example-docs",
    "search_type": "hybrid"
  }'

# Get document content
curl "http://localhost:8001/doc?path=docs/vendors/example/setup.md"

# Capture web page
curl -X POST http://localhost:8001/capture \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://example.com/article", "title": "Important Article"}'
```

#### Development Tools
- **VS Code Extension**: Use `Ctrl+Shift+P` → "DocFoundry: Search Docs" for in-editor search
- **Chrome Extension**: Click the toolbar icon to save pages to `docs/research/`
- **MCP Server**: Connect compatible MCP clients for agent-based document access

## 🔧 Configuration

### Environment Variables

```bash
# Database Configuration
DATABASE_URL=sqlite:///./docfoundry.db  # or postgresql://...
DATABASE_TYPE=sqlite  # or postgres

# API Configuration
API_HOST=0.0.0.0
API_PORT=8001
API_LOG_LEVEL=info

# Search Configuration
SEARCH_LIMIT_DEFAULT=10
SEARCH_LIMIT_MAX=100

# Crawling Configuration
CRAWL_DELAY_DEFAULT=1.0
CRAWL_MAX_DEPTH=5
CRAWL_RESPECT_ROBOTS=true

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=docfoundry
ENABLE_METRICS=true
```

### Database Setup

#### SQLite (Default)
```bash
# No additional setup required
# Database file created automatically
```

#### PostgreSQL with pgvector
```bash
# Install PostgreSQL and pgvector extension
# Create database
createdb docfoundry

# Set environment variable
export DATABASE_URL="postgresql://user:password@localhost/docfoundry"
export DATABASE_TYPE="postgres"

# Run migrations
python -c "from server.postgres_adapter import PostgresAdapter; PostgresAdapter().create_tables()"
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build the image
docker build -t docfoundry .

# Run with SQLite
docker run -p 8001:8001 -v $(pwd)/data:/app/data docfoundry

# Run with PostgreSQL
docker run -p 8001:8001 \
  -e DATABASE_URL="postgresql://user:pass@host/db" \
  -e DATABASE_TYPE="postgres" \
  docfoundry
```

### Production Considerations

- **Reverse Proxy**: Use nginx or similar for SSL termination and load balancing
- **Process Management**: Use systemd, supervisor, or container orchestration
- **Monitoring**: Enable OpenTelemetry for observability
- **Backup**: Regular database backups, especially for PostgreSQL
- **Security**: Configure proper authentication and rate limiting

## 🛣️ Roadmap

### Current Features ✅
- Multi-source content ingestion (HTML, feeds, repositories)
- Hybrid search with semantic and keyword matching
- RESTful API with comprehensive endpoints
- VS Code and Chrome browser extensions
- SQLite and PostgreSQL support with pgvector
- OpenTelemetry observability integration
- MCP server protocol support

### Planned Enhancements 🚧
- **Advanced Workflow Orchestration**: Temporal integration for complex pipelines
- **Enhanced AI Integration**: LLM-powered content summarization and tagging
- **Enterprise Features**: RBAC, audit logging, and compliance tools
- **Performance Optimizations**: Distributed indexing and caching layers
- **Content Intelligence**: Automatic content categorization and relationship mapping
- **API Enhancements**: GraphQL support and webhook integrations

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/codessa-prime/docfoundry.git
cd docfoundry
make setup

# Run tests
make test

# Run linting
make lint

# Start development server
make dev
```

### Code Standards
- Follow PEP 8 for Python code
- Add type hints for new functions
- Include docstrings for public APIs
- Write tests for new features
- Update documentation as needed

### Submitting Changes
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full documentation](https://docfoundry.readthedocs.io) (coming soon)
- **Issues**: [GitHub Issues](https://github.com/codessa-prime/docfoundry/issues)
- **Discussions**: [GitHub Discussions](https://github.com/codessa-prime/docfoundry/discussions)
- **Email**: support@docfoundry.dev

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for the REST API
- Powered by [pgvector](https://github.com/pgvector/pgvector) for vector similarity search
- Uses [OpenTelemetry](https://opentelemetry.io/) for observability
- Inspired by the need for intelligent documentation management

---

**DocFoundry** - Transform your documentation into an intelligent, searchable knowledge base. 🚀
