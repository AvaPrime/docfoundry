#!/bin/bash

# GitHub Labels Setup Script for DocFoundry
# This script creates standardized issue labels for the DocFoundry project

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

print_status "Setting up DocFoundry issue labels..."

# Function to create or update a label
create_label() {
    local name="$1"
    local color="$2"
    local description="$3"
    
    if gh label list --json name --jq '.[].name' | grep -q "^${name}$"; then
        print_warning "Label '${name}' already exists, updating..."
        gh label edit "${name}" --color "${color}" --description "${description}" || {
            print_error "Failed to update label '${name}'"
            return 1
        }
    else
        print_status "Creating label '${name}'..."
        gh label create "${name}" --color "${color}" --description "${description}" || {
            print_error "Failed to create label '${name}'"
            return 1
        }
    fi
    print_success "Label '${name}' configured successfully"
}

# Priority Labels
print_status "Creating priority labels..."
create_label "priority/critical" "d73a4a" "Critical priority - immediate attention required"
create_label "priority/high" "ff6b6b" "High priority - should be addressed soon"
create_label "priority/medium" "ffa726" "Medium priority - normal timeline"
create_label "priority/low" "66bb6a" "Low priority - can be addressed later"

# Type Labels
print_status "Creating type labels..."
create_label "type/bug" "d73a4a" "Something isn't working correctly"
create_label "type/feature" "0052cc" "New feature or enhancement request"
create_label "type/improvement" "1d76db" "Enhancement to existing functionality"
create_label "type/documentation" "0075ca" "Documentation related changes"
create_label "type/refactor" "5319e7" "Code refactoring without functional changes"
create_label "type/performance" "ff9800" "Performance optimization"
create_label "type/security" "e91e63" "Security related issue or improvement"
create_label "type/maintenance" "795548" "Maintenance and housekeeping tasks"

# Component Labels
print_status "Creating component labels..."
create_label "component/api" "1f77b4" "API server and endpoints"
create_label "component/indexer" "ff7f0e" "Document indexing and processing"
create_label "component/search" "2ca02c" "Search functionality and algorithms"
create_label "component/ui" "d62728" "User interface and frontend"
create_label "component/database" "9467bd" "Database schema and operations"
create_label "component/embeddings" "8c564b" "Embedding generation and management"
create_label "component/crawler" "e377c2" "Web crawling and document extraction"
create_label "component/observability" "7f7f7f" "Monitoring, logging, and telemetry"
create_label "component/evaluation" "bcbd22" "Testing and evaluation framework"
create_label "component/deployment" "17becf" "Deployment and infrastructure"

# Status Labels
print_status "Creating status labels..."
create_label "status/blocked" "b60205" "Blocked by external dependency or issue"
create_label "status/in-progress" "fbca04" "Currently being worked on"
create_label "status/needs-review" "0e8a16" "Ready for review"
create_label "status/needs-testing" "1d76db" "Needs testing or validation"
create_label "status/waiting-feedback" "d4c5f9" "Waiting for feedback or clarification"

# Effort Labels
print_status "Creating effort labels..."
create_label "effort/xs" "c2e0c6" "Extra small effort (< 1 day)"
create_label "effort/s" "7bcf7f" "Small effort (1-2 days)"
create_label "effort/m" "57a957" "Medium effort (3-5 days)"
create_label "effort/l" "2d5a2d" "Large effort (1-2 weeks)"
create_label "effort/xl" "1a3d1a" "Extra large effort (> 2 weeks)"

# Special Labels
print_status "Creating special labels..."
create_label "good-first-issue" "7057ff" "Good for newcomers to the project"
create_label "help-wanted" "008672" "Extra attention is needed from community"
create_label "breaking-change" "d93f0b" "Introduces breaking changes to API or behavior"
create_label "dependencies" "0366d6" "Pull requests that update dependencies"
create_label "duplicate" "cfd3d7" "This issue or pull request already exists"
create_label "invalid" "e4e669" "This doesn't seem right or is not actionable"
create_label "question" "d876e3" "Further information is requested"
create_label "wontfix" "ffffff" "This will not be worked on"

# Milestone-specific Labels
print_status "Creating milestone-specific labels..."
create_label "milestone/v0.1" "f9d71c" "Issues for version 0.1 release"
create_label "milestone/v0.2" "daa520" "Issues for version 0.2 release"
create_label "milestone/v1.0" "b8860b" "Issues for version 1.0 release"

# Search-specific Labels
print_status "Creating search-specific labels..."
create_label "search/hybrid" "ff6b9d" "Hybrid search functionality"
create_label "search/semantic" "c44569" "Semantic search functionality"
create_label "search/keyword" "8b4513" "Keyword/full-text search functionality"
create_label "search/ranking" "4682b4" "Search result ranking and scoring"

print_success "All labels have been created successfully!"
print_status "You can view all labels at: $(gh repo view --json url --jq '.url')/labels"

# Optional: List all labels
if [[ "${1:-}" == "--list" ]]; then
    print_status "Current labels in the repository:"
    gh label list --json name,color,description --template '{{range .}}{{printf "%-25s" .name}} {{printf "#%-7s" .color}} {{.description}}{{"\n"}}{{end}}'
fi

print_success "Label setup complete! 🎉"