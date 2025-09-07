# DocFoundry

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)

A comprehensive **Documentation Intelligence System** that transforms scattered documentation into an intelligent, searchable knowledge base. DocFoundry discovers, ingests, normalizes, indexes, and serves documentation from multiple sources including websites, repositories, feeds, and local files. 

**Key Value Propositions:**
- 🎯 **Unified Knowledge Access**: Centralize documentation from multiple sources into a single searchable interface
- 🧠 **Intelligent Search**: Combine full-text search with semantic understanding for precise results
- 🔌 **Developer-First**: Seamless integration with VS Code, Chrome, and development workflows
- 📊 **Analytics-Driven**: Comprehensive observability and search analytics for continuous improvement
- 🚀 **Production-Ready**: Scalable architecture supporting both SQLite and PostgreSQL backends

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

### System Requirements

**Minimum Requirements:**
- Python 3.11 or higher
- 4GB RAM (8GB recommended for production)
- 2GB available disk space
- Git for version control

**Optional Dependencies:**
- Docker and Docker Compose for containerized deployment
- PostgreSQL 14+ with pgvector extension for production deployments
- Redis for caching (future enhancement)
- Node.js 18+ for browser extension development

### Installation

```bash
# Clone the repository
git clone https://github.com/AvaPrime/docfoundry.git
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

We welcome contributions from the community! DocFoundry is built with collaboration in mind, and we appreciate all forms of contribution including code, documentation, bug reports, and feature suggestions.

### Getting Started

**Before Contributing:**
1. Read our [Code of Conduct](CONTRIBUTING.md#code-of-conduct)
2. Check existing [issues](https://github.com/AvaPrime/docfoundry/issues) and [discussions](https://github.com/AvaPrime/docfoundry/discussions)
3. Review our [project roadmap](#-roadmap) to understand current priorities

### Development Environment Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/docfoundry.git
cd docfoundry

# 2. Set up development environment
make setup  # Installs dev dependencies and pre-commit hooks

# 3. Verify installation
make test   # Run test suite
make lint   # Check code quality

# 4. Start development server
make dev    # Starts API with auto-reload
```

### Development Workflow

**Code Quality Standards:**
- **Python Style**: Follow PEP 8, enforced by `black` and `flake8`
- **Type Safety**: Add type hints for all new functions and classes
- **Documentation**: Include comprehensive docstrings following Google style
- **Testing**: Maintain >90% test coverage for new code
- **Security**: Follow OWASP guidelines, no hardcoded secrets

**Commit Guidelines:**
- Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- Keep commits atomic and well-described
- Reference issues in commit messages: `fixes #123`

### Types of Contributions

**🐛 Bug Reports**
- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml)
- Include reproduction steps, expected vs actual behavior
- Provide system information and logs when relevant

**✨ Feature Requests**
- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml)
- Explain the use case and expected benefits
- Consider implementation complexity and maintenance burden

**📝 Documentation**
- Improve existing documentation clarity and accuracy
- Add examples and tutorials for common use cases
- Translate documentation to other languages

**🔧 Code Contributions**
- Start with "good first issue" labeled items
- Discuss major changes in issues before implementation
- Follow the pull request template requirements

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/descriptive-name
   ```

2. **Implement Changes**
   - Write code following our standards
   - Add comprehensive tests
   - Update documentation as needed

3. **Quality Checks**
   ```bash
   make test     # Run full test suite
   make lint     # Check code quality
   make docs     # Verify documentation builds
   ```

4. **Submit Pull Request**
   - Use the provided PR template
   - Link related issues
   - Request review from maintainers

5. **Review Process**
   - Address reviewer feedback promptly
   - Keep PR scope focused and manageable
   - Ensure CI/CD checks pass

### Recognition

Contributors are recognized in:
- [CHANGELOG.md](CHANGELOG.md) for each release
- GitHub contributors page
- Special mentions for significant contributions

For questions about contributing, join our [GitHub Discussions](https://github.com/AvaPrime/docfoundry/discussions) or reach out to the maintainers.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔧 Troubleshooting

### Common Issues

**Installation Problems**
```bash
# Python version issues
python --version  # Should be 3.11+

# Virtual environment activation
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Dependency conflicts
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**Database Issues**
```bash
# SQLite permissions
chmod 664 docfoundry.db
chmod 775 $(dirname docfoundry.db)

# PostgreSQL connection
psql $DATABASE_URL -c "SELECT version();"

# Reset database
rm docfoundry.db  # SQLite only
python indexer/build_index.py
```

**API Server Problems**
```bash
# Port already in use
lsof -i :8001  # Find process using port
kill -9 <PID>  # Kill process

# Check server logs
uvicorn server.rag_api:app --log-level debug

# Test API health
curl http://localhost:8001/health
```

**Search Issues**
```bash
# Rebuild search index
python indexer/build_index.py --force

# Check index statistics
curl http://localhost:8001/stats

# Verify source configuration
python -c "from sources.loader import load_sources; print(load_sources())"
```

### Performance Optimization

**For Large Datasets:**
- Use PostgreSQL with pgvector for production
- Enable database connection pooling
- Configure appropriate chunk sizes in source configs
- Monitor memory usage during indexing

**Search Performance:**
- Use specific source filters when possible
- Limit result counts for broad queries
- Consider semantic search for conceptual queries
- Use full-text search for exact term matching

### Getting Help

If you encounter issues not covered here:
1. Check the [FAQ section](https://github.com/AvaPrime/docfoundry/discussions/categories/q-a)
2. Search existing [issues](https://github.com/AvaPrime/docfoundry/issues)
3. Create a new issue with detailed information
4. Join our community discussions for real-time help

## 🆘 Support & Community

### Documentation & Resources
- **📚 Full Documentation**: [DocFoundry Docs](https://docfoundry.readthedocs.io) (coming soon)
- **🎯 Quick Start Guide**: See [Quick Start](#-quick-start) section above
- **🔧 API Reference**: Available at `/docs` endpoint when server is running
- **📋 Examples**: Check the `examples/` directory for usage patterns

### Community Support
- **💬 GitHub Discussions**: [Join conversations](https://github.com/AvaPrime/docfoundry/discussions)
  - Q&A for usage questions
  - Feature discussions and feedback
  - Show and tell your implementations
- **🐛 Issue Tracker**: [Report bugs](https://github.com/AvaPrime/docfoundry/issues)
- **📧 Email Support**: support@docfoundry.dev (for security issues)

### Response Times
- **Community Support**: Best effort, typically within 24-48 hours
- **Bug Reports**: Acknowledged within 48 hours, fix timeline depends on severity
- **Security Issues**: Acknowledged within 24 hours, patches prioritized

### Contributing to Support
Help others by:
- Answering questions in discussions
- Improving documentation
- Sharing usage examples and tutorials
- Reporting and helping fix bugs

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for the REST API
- Powered by [pgvector](https://github.com/pgvector/pgvector) for vector similarity search
- Uses [OpenTelemetry](https://opentelemetry.io/) for observability
- Inspired by the need for intelligent documentation management

---

**DocFoundry** - Transform your documentation into an intelligent, searchable knowledge base. 🚀
