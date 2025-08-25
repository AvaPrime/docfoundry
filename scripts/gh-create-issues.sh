#!/bin/bash

# GitHub Issues Creation Script for DocFoundry
# This script creates milestone-based issues for the DocFoundry project roadmap

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    print_error "GitHub CLI (gh) is not installed. Please install it first."
    print_status "Visit: https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    print_error "Not authenticated with GitHub CLI. Please run 'gh auth login' first."
    exit 1
fi

print_status "Creating DocFoundry roadmap issues..."

# Function to create an issue
create_issue() {
    local title="$1"
    local body="$2"
    local labels="$3"
    local milestone="$4"
    
    print_status "Creating issue: $title"
    
    local cmd="gh issue create --title '$title' --body '$body'"
    
    if [[ -n "$labels" ]]; then
        cmd="$cmd --label '$labels'"
    fi
    
    if [[ -n "$milestone" ]]; then
        # Check if milestone exists, create if not
        if ! gh api repos/:owner/:repo/milestones --jq '.[].title' | grep -q "^${milestone}$"; then
            print_status "Creating milestone: $milestone"
            gh api repos/:owner/:repo/milestones -f title="$milestone" -f description="Milestone for $milestone development" || {
                print_warning "Failed to create milestone $milestone, continuing without it"
                milestone=""
            }
        fi
        
        if [[ -n "$milestone" ]]; then
            cmd="$cmd --milestone '$milestone'"
        fi
    fi
    
    eval "$cmd" || {
        print_error "Failed to create issue: $title"
        return 1
    }
    
    print_success "Created issue: $title"
}

# Create milestones first
print_status "Setting up milestones..."

# V0.1 - Foundation
gh api repos/:owner/:repo/milestones -f title="v0.1 - Foundation" -f description="Core infrastructure and basic functionality" -f due_on="$(date -d '+2 months' -Iseconds)" 2>/dev/null || print_warning "Milestone v0.1 may already exist"

# V0.2 - Enhancement
gh api repos/:owner/:repo/milestones -f title="v0.2 - Enhancement" -f description="Advanced features and optimizations" -f due_on="$(date -d '+4 months' -Iseconds)" 2>/dev/null || print_warning "Milestone v0.2 may already exist"

# V1.0 - Production
gh api repos/:owner/:repo/milestones -f title="v1.0 - Production" -f description="Production-ready release" -f due_on="$(date -d '+6 months' -Iseconds)" 2>/dev/null || print_warning "Milestone v1.0 may already exist"

# V0.1 Issues - Foundation
print_status "Creating v0.1 Foundation issues..."

create_issue "Implement PostgreSQL + pgvector Migration" \
"## Description
Migrate from SQLite to PostgreSQL with pgvector extension for production-grade vector storage.

## Tasks
- [ ] Set up PostgreSQL database schema
- [ ] Install and configure pgvector extension
- [ ] Create migration scripts from SQLite
- [ ] Update database connection configuration
- [ ] Test vector operations with pgvector
- [ ] Update documentation

## Acceptance Criteria
- PostgreSQL database is fully functional
- Vector embeddings are stored in pgvector
- All existing functionality works with new database
- Migration path is documented

## Technical Notes
- Use pgvector for embedding storage
- Maintain backward compatibility during migration
- Consider connection pooling for performance" \
"type/feature,component/database,priority/high,effort/l" \
"v0.1 - Foundation"

create_issue "Enhance Chunker with Heading Hierarchy" \
"## Description
Improve the document chunker to be heading-aware and maintain document structure hierarchy.

## Tasks
- [ ] Parse markdown headings (H1-H6)
- [ ] Build hierarchical document structure
- [ ] Generate deterministic chunk IDs
- [ ] Implement content hashing for deduplication
- [ ] Add chunk overlap configuration
- [ ] Test with various document formats

## Acceptance Criteria
- Chunker preserves document hierarchy
- Chunk IDs are deterministic and unique
- Content hashing prevents duplicates
- Configurable chunk size and overlap

## Technical Notes
- Use markdown parsing library
- Implement tree structure for headings
- Consider chunk size optimization" \
"type/improvement,component/indexer,priority/high,effort/m" \
"v0.1 - Foundation"

create_issue "Add Source Schema Validation" \
"## Description
Implement comprehensive source document validation using JSON Schema.

## Tasks
- [ ] Define source document schema
- [ ] Implement validation functions
- [ ] Add error handling and reporting
- [ ] Create validation utilities
- [ ] Add schema versioning support
- [ ] Write validation tests

## Acceptance Criteria
- All source documents are validated
- Clear error messages for invalid documents
- Schema versioning is supported
- Validation is performant

## Technical Notes
- Use JSON Schema for validation
- Support multiple document types
- Consider async validation for large documents" \
"type/feature,component/indexer,priority/medium,effort/s" \
"v0.1 - Foundation"

create_issue "Implement Basic Observability" \
"## Description
Add OpenTelemetry tracing and Prometheus metrics for system observability.

## Tasks
- [ ] Set up OpenTelemetry configuration
- [ ] Add distributed tracing
- [ ] Implement Prometheus metrics
- [ ] Create custom metrics collectors
- [ ] Add structured logging
- [ ] Set up health check endpoints

## Acceptance Criteria
- Tracing is available for all major operations
- Key metrics are collected and exposed
- Structured logging is implemented
- Health checks are functional

## Technical Notes
- Use OpenTelemetry Python SDK
- Export metrics to Prometheus format
- Consider performance impact of instrumentation" \
"type/feature,component/observability,priority/medium,effort/m" \
"v0.1 - Foundation"

# V0.2 Issues - Enhancement
print_status "Creating v0.2 Enhancement issues..."

create_issue "Advanced Search Ranking Algorithm" \
"## Description
Implement sophisticated search result ranking combining multiple signals.

## Tasks
- [ ] Design ranking algorithm
- [ ] Implement BM25 scoring
- [ ] Add semantic similarity scoring
- [ ] Combine multiple ranking signals
- [ ] Add personalization features
- [ ] Performance optimization

## Acceptance Criteria
- Search results are well-ranked
- Multiple ranking signals are combined
- Performance is acceptable
- Ranking is explainable

## Technical Notes
- Consider learning-to-rank approaches
- Balance relevance and diversity
- Implement result re-ranking" \
"type/feature,component/search,priority/high,effort/l" \
"v0.2 - Enhancement"

create_issue "Real-time Search Analytics" \
"## Description
Implement real-time analytics for search queries and user behavior.

## Tasks
- [ ] Track search queries and results
- [ ] Implement click-through tracking
- [ ] Add search performance metrics
- [ ] Create analytics dashboard
- [ ] Add A/B testing framework
- [ ] Implement query suggestions

## Acceptance Criteria
- Search analytics are collected in real-time
- Dashboard shows key metrics
- A/B testing is functional
- Query suggestions improve over time

## Technical Notes
- Use event streaming for real-time data
- Consider privacy implications
- Implement efficient data aggregation" \
"type/feature,component/observability,priority/medium,effort/l" \
"v0.2 - Enhancement"

create_issue "Advanced Embedding Models" \
"## Description
Integrate advanced embedding models and support model switching.

## Tasks
- [ ] Support multiple embedding models
- [ ] Implement model comparison framework
- [ ] Add fine-tuning capabilities
- [ ] Optimize embedding generation
- [ ] Add embedding quality metrics
- [ ] Support custom models

## Acceptance Criteria
- Multiple embedding models are supported
- Model switching is seamless
- Embedding quality is measurable
- Performance is optimized

## Technical Notes
- Support HuggingFace models
- Consider model size vs. quality tradeoffs
- Implement efficient model loading" \
"type/feature,component/embeddings,priority/medium,effort/l" \
"v0.2 - Enhancement"

create_issue "Enhanced Web UI with Modern Design" \
"## Description
Create a modern, responsive web interface for DocFoundry search.

## Tasks
- [ ] Design modern UI/UX
- [ ] Implement responsive layout
- [ ] Add advanced search features
- [ ] Implement search filters
- [ ] Add result previews
- [ ] Optimize for mobile devices

## Acceptance Criteria
- UI is modern and intuitive
- Responsive design works on all devices
- Advanced search features are accessible
- Performance is excellent

## Technical Notes
- Use modern CSS framework
- Implement progressive enhancement
- Consider accessibility requirements" \
"type/feature,component/ui,priority/high,effort/m" \
"v0.2 - Enhancement"

# V1.0 Issues - Production
print_status "Creating v1.0 Production issues..."

create_issue "Production Deployment Pipeline" \
"## Description
Implement comprehensive CI/CD pipeline for production deployment.

## Tasks
- [ ] Set up GitHub Actions workflows
- [ ] Implement automated testing
- [ ] Add security scanning
- [ ] Create deployment scripts
- [ ] Set up monitoring and alerting
- [ ] Add rollback procedures

## Acceptance Criteria
- Automated deployment pipeline is functional
- All tests pass before deployment
- Security scanning is integrated
- Monitoring and alerting are active

## Technical Notes
- Use infrastructure as code
- Implement blue-green deployment
- Consider container orchestration" \
"type/feature,component/deployment,priority/high,effort/xl" \
"v1.0 - Production"

create_issue "Comprehensive Security Audit" \
"## Description
Conduct thorough security audit and implement security best practices.

## Tasks
- [ ] Perform security vulnerability assessment
- [ ] Implement authentication and authorization
- [ ] Add input validation and sanitization
- [ ] Set up security monitoring
- [ ] Add rate limiting and DDoS protection
- [ ] Implement secure configuration management

## Acceptance Criteria
- Security vulnerabilities are identified and fixed
- Authentication and authorization are robust
- Security monitoring is active
- System is protected against common attacks

## Technical Notes
- Follow OWASP guidelines
- Use security scanning tools
- Implement defense in depth" \
"type/security,priority/critical,effort/l" \
"v1.0 - Production"

create_issue "Performance Optimization and Scaling" \
"## Description
Optimize system performance and implement horizontal scaling capabilities.

## Tasks
- [ ] Profile application performance
- [ ] Optimize database queries
- [ ] Implement caching strategies
- [ ] Add load balancing
- [ ] Optimize embedding operations
- [ ] Implement auto-scaling

## Acceptance Criteria
- System performance meets SLA requirements
- Horizontal scaling is functional
- Caching improves response times
- Load balancing distributes traffic effectively

## Technical Notes
- Use performance profiling tools
- Implement Redis for caching
- Consider CDN for static assets" \
"type/performance,priority/high,effort/l" \
"v1.0 - Production"

create_issue "Comprehensive Documentation" \
"## Description
Create comprehensive documentation for users, developers, and operators.

## Tasks
- [ ] Write user documentation
- [ ] Create API documentation
- [ ] Add developer setup guide
- [ ] Write deployment documentation
- [ ] Create troubleshooting guide
- [ ] Add architecture documentation

## Acceptance Criteria
- Documentation is comprehensive and up-to-date
- Users can easily get started
- Developers can contribute effectively
- Operators can deploy and maintain the system

## Technical Notes
- Use documentation-as-code approach
- Include examples and tutorials
- Keep documentation synchronized with code" \
"type/documentation,priority/medium,effort/m" \
"v1.0 - Production"

# Cross-cutting Issues
print_status "Creating cross-cutting issues..."

create_issue "Evaluation Framework Implementation" \
"## Description
Implement comprehensive evaluation framework for search quality assessment.

## Tasks
- [ ] Create evaluation datasets
- [ ] Implement relevance metrics
- [ ] Add performance benchmarks
- [ ] Create A/B testing framework
- [ ] Add automated evaluation pipeline
- [ ] Generate evaluation reports

## Acceptance Criteria
- Evaluation framework is comprehensive
- Search quality can be measured objectively
- Performance benchmarks are automated
- Evaluation reports are generated regularly

## Technical Notes
- Use standard IR evaluation metrics
- Implement statistical significance testing
- Consider human evaluation integration" \
"type/feature,component/evaluation,priority/medium,effort/l" \
"v0.2 - Enhancement"

create_issue "Multi-language Support" \
"## Description
Add support for multiple languages in search and indexing.

## Tasks
- [ ] Implement language detection
- [ ] Add language-specific tokenization
- [ ] Support multilingual embeddings
- [ ] Add language-aware search
- [ ] Implement translation features
- [ ] Test with various languages

## Acceptance Criteria
- Multiple languages are supported
- Language detection is accurate
- Search works across languages
- Translation features are functional

## Technical Notes
- Use language detection libraries
- Consider language-specific embedding models
- Implement proper Unicode handling" \
"type/feature,priority/low,effort/xl" \
"v1.0 - Production"

print_success "All roadmap issues have been created successfully!"
print_status "You can view all issues at: $(gh repo view --json url --jq '.url')/issues"
print_status "You can view milestones at: $(gh repo view --json url --jq '.url')/milestones"

print_success "Issue creation complete! 🎉"
print_status "Next steps:"
print_status "1. Review and prioritize the created issues"
print_status "2. Assign issues to team members"
print_status "3. Start working on v0.1 Foundation milestone"