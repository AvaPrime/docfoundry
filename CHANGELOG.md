# Changelog

All notable changes to DocFoundry will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation overhaul with enhanced README
- VS Code extension with intelligent documentation search
- Chrome extension for web page capture
- Performance optimizations with database connection pooling
- Advanced caching system for improved search response times
- Comprehensive monitoring and observability framework
- OpenTelemetry integration for distributed tracing
- PostgreSQL migration with pgvector support
- Hybrid search combining semantic and keyword search
- Policy guardrails with license compliance checking
- MCP (Model Context Protocol) server integration
- Learning-to-rank search optimization
- Multi-source documentation ingestion

### Changed
- Migrated from SQLite to PostgreSQL for better performance
- Enhanced search algorithms with reranking capabilities
- Improved API documentation and OpenAPI specifications
- Standardized configuration management
- Updated project structure and organization

### Fixed
- Database connection handling and pooling issues
- Search performance bottlenecks
- Memory usage optimization
- Error handling and logging improvements

### Security
- Added content filtering and policy compliance
- Implemented secure database connection practices
- Enhanced input validation and sanitization

## [0.1.0] - Initial Release

### Added
- Basic RAG (Retrieval-Augmented Generation) functionality
- SQLite database support
- Simple web crawling and indexing
- REST API for document search
- Basic documentation structure

---

## Release Notes

### Version Numbering

DocFoundry follows [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

### Release Process

1. **Development**: Features developed in feature branches
2. **Testing**: Comprehensive testing including performance benchmarks
3. **Documentation**: Update documentation and changelog
4. **Release**: Tag version and create GitHub release
5. **Deployment**: Update deployment guides and examples

### Upgrade Guidelines

#### From 0.x to 1.x (Future Major Release)
- Database migration required (automated scripts provided)
- Configuration file format changes
- API endpoint modifications
- Extension updates required

#### Minor Version Updates
- Backwards compatible
- New features available
- Optional configuration updates
- Extension updates recommended

#### Patch Updates
- Bug fixes and security updates
- No breaking changes
- Automatic deployment safe

### Support Policy

- **Current Major Version**: Full support with new features and bug fixes
- **Previous Major Version**: Security updates and critical bug fixes for 6 months
- **Older Versions**: Community support only

### Breaking Changes Policy

Breaking changes will be:
1. Clearly documented in the changelog
2. Announced in advance when possible
3. Accompanied by migration guides
4. Introduced only in major version releases

---

**Note**: This changelog is automatically updated during the release process. For the most current development status, see the [project roadmap](doc_foundry_roadmap_issues_labels_api_spec.md).