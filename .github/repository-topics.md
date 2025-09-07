# Repository Topics and Configuration

This document defines the recommended topics and configuration for the DocFoundry GitHub repository.

## Recommended Topics

The following topics should be added to the GitHub repository for better discoverability:

### Primary Topics
- `rag` - Retrieval-Augmented Generation
- `documentation` - Documentation management and search
- `semantic-search` - Vector-based semantic search
- `knowledge-base` - Knowledge management system
- `python` - Primary programming language
- `fastapi` - Web framework used
- `postgresql` - Database technology
- `vector-database` - Vector storage and search
- `embeddings` - Text embeddings and similarity

### Technology Topics
- `opentelemetry` - Observability and monitoring
- `docker` - Containerization
- `vscode-extension` - VS Code integration
- `chrome-extension` - Browser integration
- `pgvector` - PostgreSQL vector extension
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `asyncio` - Asynchronous programming

### Domain Topics
- `information-retrieval` - Information retrieval systems
- `nlp` - Natural Language Processing
- `machine-learning` - ML applications
- `search-engine` - Search functionality
- `document-processing` - Document ingestion and processing
- `web-crawling` - Web content extraction
- `api` - REST API services
- `enterprise` - Enterprise-ready features

### Use Case Topics
- `developer-tools` - Tools for developers
- `research` - Research and analysis tools
- `productivity` - Productivity enhancement
- `automation` - Process automation
- `integration` - System integration

## Repository Configuration

### Description
```
Intelligent documentation system with RAG capabilities, semantic search, and multi-source ingestion. Features VS Code/Chrome extensions, PostgreSQL with pgvector, and comprehensive observability.
```

### Website
- Documentation: Link to deployed documentation site
- Demo: Link to live demo if available

### Repository Settings
- **Visibility**: Public
- **Features**:
  - ✅ Issues
  - ✅ Projects
  - ✅ Wiki (if used)
  - ✅ Discussions (recommended)
  - ✅ Actions (CI/CD)
  - ✅ Security (vulnerability alerts)

### Branch Protection
- **Default branch**: `main`
- **Protection rules**:
  - Require pull request reviews
  - Require status checks to pass
  - Require branches to be up to date
  - Include administrators in restrictions

### Labels
See `scripts/gh-setup-labels.sh` for automated label setup.

### Issue Templates
Recommended issue templates:
- Bug Report
- Feature Request
- Documentation Improvement
- Performance Issue
- Security Vulnerability

### Pull Request Template
See `.github/pull_request_template.md` for standardized PR descriptions.

## How to Apply Topics

### Via GitHub Web Interface
1. Go to the repository main page
2. Click the gear icon (⚙️) next to "About"
3. Add topics in the "Topics" field
4. Save changes

### Via GitHub CLI
```bash
# Add topics using GitHub CLI
gh repo edit --add-topic rag,documentation,semantic-search,knowledge-base,python,fastapi,postgresql,vector-database,embeddings,opentelemetry,docker,vscode-extension,chrome-extension,pgvector,information-retrieval,nlp,machine-learning,search-engine,developer-tools,research,productivity
```

### Via API
```bash
# Using GitHub API
curl -X PUT \
  -H "Accept: application/vnd.github.mercy-preview+json" \
  -H "Authorization: token YOUR_TOKEN" \
  https://api.github.com/repos/OWNER/REPO/topics \
  -d '{"names":["rag","documentation","semantic-search","knowledge-base","python","fastapi","postgresql","vector-database","embeddings"]}'
```

## Topic Guidelines

### Best Practices
- Use lowercase, hyphenated format
- Maximum 20 topics per repository
- Focus on most relevant and searchable terms
- Include both technical and domain-specific topics
- Update topics as the project evolves

### Topic Categories
1. **Technology Stack**: Languages, frameworks, databases
2. **Domain/Purpose**: What the project does
3. **Use Cases**: Who would use this project
4. **Features**: Key capabilities and integrations

### Avoid
- Generic terms like "software" or "tool"
- Overly specific terms with limited search volume
- Duplicate or redundant topics
- Trademarked terms without permission

## Monitoring and Updates

- Review topics quarterly
- Update based on new features or technology changes
- Monitor topic effectiveness via repository insights
- Align with community standards and trending topics

---

**Note**: This configuration enhances repository discoverability and helps developers find DocFoundry when searching for related technologies and use cases.