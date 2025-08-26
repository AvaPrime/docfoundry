"use client";

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Search, Filter, Download, BookOpen, Clock, Globe, Star, TrendingUp, X, ChevronDown, ChevronUp } from 'lucide-react';

interface SearchResult {
  id: string;
  title: string;
  url: string;
  snippet: string;
  score: number;
  source: string;
  language: string;
  retrieved_at: string;
  document_id: string;
  hierarchical_path?: string[];
  metadata?: {
    author?: string;
    published_date?: string;
    content_type?: string;
    word_count?: number;
    reading_time?: number;
  };
}

interface SearchFilters {
  sources: string[];
  languages: string[];
  dateRange: {
    start?: string;
    end?: string;
  };
  scoreThreshold: number;
  contentTypes: string[];
  sortBy: 'relevance' | 'date' | 'score' | 'title';
  sortOrder: 'asc' | 'desc';
}

interface SearchStats {
  totalResults: number;
  searchTime: number;
  topSources: Array<{ source: string; count: number }>;
  avgScore: number;
  languageDistribution: Array<{ language: string; count: number }>;
}

interface EnhancedSearchInterfaceProps {
  onSearch: (query: string, filters: SearchFilters, mode: string) => Promise<{
    results: SearchResult[];
    stats: SearchStats;
  }>;
  availableSources: string[];
  availableLanguages: string[];
  recentSearches: string[];
  onSaveSearch?: (query: string, filters: SearchFilters) => void;
}

const EnhancedSearchInterface: React.FC<EnhancedSearchInterfaceProps> = ({
  onSearch,
  availableSources,
  availableLanguages,
  recentSearches,
  onSaveSearch
}) => {
  // State management
  const [query, setQuery] = useState('');
  const [searchMode, setSearchMode] = useState<'hybrid' | 'semantic' | 'fulltext'>('hybrid');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [stats, setStats] = useState<SearchStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [savedSearches, setSavedSearches] = useState<Array<{ query: string; filters: SearchFilters; timestamp: string }>>([]);
  
  // Filter state
  const [filters, setFilters] = useState<SearchFilters>({
    sources: [],
    languages: [],
    dateRange: {},
    scoreThreshold: 0,
    contentTypes: [],
    sortBy: 'relevance',
    sortOrder: 'desc'
  });

  // Load saved searches from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('docfoundry_saved_searches');
    if (saved) {
      try {
        setSavedSearches(JSON.parse(saved));
      } catch (e) {
        console.error('Error loading saved searches:', e);
      }
    }
  }, []);

  // Save searches to localStorage
  const saveCurrentSearch = useCallback(() => {
    if (!query.trim()) return;
    
    const searchToSave = {
      query,
      filters,
      timestamp: new Date().toISOString()
    };
    
    const updated = [searchToSave, ...savedSearches.filter(s => s.query !== query)].slice(0, 10);
    setSavedSearches(updated);
    localStorage.setItem('docfoundry_saved_searches', JSON.stringify(updated));
    
    if (onSaveSearch) {
      onSaveSearch(query, filters);
    }
  }, [query, filters, savedSearches, onSaveSearch]);

  // Perform search
  const performSearch = useCallback(async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await onSearch(query, filters, searchMode);
      setResults(response.results);
      setStats(response.stats);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
      setResults([]);
      setStats(null);
    } finally {
      setLoading(false);
    }
  }, [query, filters, searchMode, onSearch]);

  // Handle search submission
  const handleSearch = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    performSearch();
  }, [performSearch]);

  // Filter results based on current filters
  const filteredResults = useMemo(() => {
    let filtered = [...results];
    
    // Apply score threshold
    if (filters.scoreThreshold > 0) {
      filtered = filtered.filter(r => r.score >= filters.scoreThreshold);
    }
    
    // Apply source filter
    if (filters.sources.length > 0) {
      filtered = filtered.filter(r => filters.sources.includes(r.source));
    }
    
    // Apply language filter
    if (filters.languages.length > 0) {
      filtered = filtered.filter(r => filters.languages.includes(r.language));
    }
    
    // Apply date range filter
    if (filters.dateRange.start || filters.dateRange.end) {
      filtered = filtered.filter(r => {
        const date = new Date(r.retrieved_at);
        const start = filters.dateRange.start ? new Date(filters.dateRange.start) : null;
        const end = filters.dateRange.end ? new Date(filters.dateRange.end) : null;
        
        return (!start || date >= start) && (!end || date <= end);
      });
    }
    
    // Apply sorting
    filtered.sort((a, b) => {
      let comparison = 0;
      
      switch (filters.sortBy) {
        case 'score':
          comparison = a.score - b.score;
          break;
        case 'date':
          comparison = new Date(a.retrieved_at).getTime() - new Date(b.retrieved_at).getTime();
          break;
        case 'title':
          comparison = a.title.localeCompare(b.title);
          break;
        case 'relevance':
        default:
          comparison = a.score - b.score;
          break;
      }
      
      return filters.sortOrder === 'desc' ? -comparison : comparison;
    });
    
    return filtered;
  }, [results, filters]);

  // Update filter
  const updateFilter = useCallback(<K extends keyof SearchFilters>(key: K, value: SearchFilters[K]) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  }, []);

  // Clear all filters
  const clearFilters = useCallback(() => {
    setFilters({
      sources: [],
      languages: [],
      dateRange: {},
      scoreThreshold: 0,
      contentTypes: [],
      sortBy: 'relevance',
      sortOrder: 'desc'
    });
  }, []);

  // Load saved search
  const loadSavedSearch = useCallback((savedSearch: { query: string; filters: SearchFilters }) => {
    setQuery(savedSearch.query);
    setFilters(savedSearch.filters);
  }, []);

  // Format score for display
  const formatScore = (score: number) => {
    return (score * 100).toFixed(1) + '%';
  };

  // Format date for display
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Calculate reading time
  const calculateReadingTime = (wordCount?: number) => {
    if (!wordCount) return null;
    const wordsPerMinute = 200;
    const minutes = Math.ceil(wordCount / wordsPerMinute);
    return `${minutes} min read`;
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          DocFoundry Enhanced Search
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Intelligent document discovery with advanced filtering and analytics
        </p>
      </div>

      {/* Search Form */}
      <form onSubmit={handleSearch} className="space-y-4">
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-5 w-5 text-gray-400" />
          </div>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search documents, ask questions, or explore topics..."
            className="block w-full pl-10 pr-12 py-4 text-lg border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-800 dark:border-gray-600 dark:text-white dark:placeholder-gray-400"
            disabled={loading}
          />
          <div className="absolute inset-y-0 right-0 flex items-center space-x-2 pr-3">
            <button
              type="button"
              onClick={() => setShowFilters(!showFilters)}
              className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              title="Toggle filters"
            >
              <Filter className="h-5 w-5" />
            </button>
            <button
              type="button"
              onClick={saveCurrentSearch}
              className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              title="Save search"
              disabled={!query.trim()}
            >
              <Star className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Search Mode Toggle */}
        <div className="flex justify-center space-x-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
          {(['hybrid', 'semantic', 'fulltext'] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => setSearchMode(mode)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                searchMode === mode
                  ? 'bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
              }`}
            >
              {mode.charAt(0).toUpperCase() + mode.slice(1)}
            </button>
          ))}
        </div>

        {/* Search Button */}
        <div className="flex justify-center">
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
          >
            {loading ? (
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                <span>Searching...</span>
              </div>
            ) : (
              'Search'
            )}
          </button>
        </div>
      </form>

      {/* Filters Panel */}
      {showFilters && (
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Search Filters</h3>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
              >
                {showAdvanced ? 'Hide' : 'Show'} Advanced
                {showAdvanced ? <ChevronUp className="inline h-4 w-4 ml-1" /> : <ChevronDown className="inline h-4 w-4 ml-1" />}
              </button>
              <button
                onClick={clearFilters}
                className="text-sm text-red-600 dark:text-red-400 hover:underline"
              >
                Clear All
              </button>
              <button
                onClick={() => setShowFilters(false)}
                className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Sources Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Sources
              </label>
              <div className="space-y-2 max-h-32 overflow-y-auto">
                {availableSources.map((source) => (
                  <label key={source} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={filters.sources.includes(source)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          updateFilter('sources', [...filters.sources, source]);
                        } else {
                          updateFilter('sources', filters.sources.filter(s => s !== source));
                        }
                      }}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">{source}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Languages Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Languages
              </label>
              <div className="space-y-2 max-h-32 overflow-y-auto">
                {availableLanguages.map((language) => (
                  <label key={language} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={filters.languages.includes(language)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          updateFilter('languages', [...filters.languages, language]);
                        } else {
                          updateFilter('languages', filters.languages.filter(l => l !== language));
                        }
                      }}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">{language}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Sort Options */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Sort By
              </label>
              <select
                value={filters.sortBy}
                onChange={(e) => updateFilter('sortBy', e.target.value as SearchFilters['sortBy'])}
                className="block w-full rounded-md border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="relevance">Relevance</option>
                <option value="score">Score</option>
                <option value="date">Date</option>
                <option value="title">Title</option>
              </select>
              <div className="mt-2 flex items-center space-x-4">
                <label className="flex items-center">
                  <input
                    type="radio"
                    name="sortOrder"
                    value="desc"
                    checked={filters.sortOrder === 'desc'}
                    onChange={(e) => updateFilter('sortOrder', e.target.value as 'desc')}
                    className="text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-1 text-sm text-gray-600 dark:text-gray-400">Descending</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="radio"
                    name="sortOrder"
                    value="asc"
                    checked={filters.sortOrder === 'asc'}
                    onChange={(e) => updateFilter('sortOrder', e.target.value as 'asc')}
                    className="text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-1 text-sm text-gray-600 dark:text-gray-400">Ascending</span>
                </label>
              </div>
            </div>
          </div>

          {/* Advanced Filters */}
          {showAdvanced && (
            <div className="border-t border-gray-200 dark:border-gray-700 pt-4 space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Score Threshold */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Minimum Score: {formatScore(filters.scoreThreshold)}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={filters.scoreThreshold}
                    onChange={(e) => updateFilter('scoreThreshold', parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                  />
                </div>

                {/* Date Range */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Date Range
                  </label>
                  <div className="flex space-x-2">
                    <input
                      type="date"
                      value={filters.dateRange.start || ''}
                      onChange={(e) => updateFilter('dateRange', { ...filters.dateRange, start: e.target.value })}
                      className="flex-1 rounded-md border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white focus:ring-blue-500 focus:border-blue-500"
                    />
                    <input
                      type="date"
                      value={filters.dateRange.end || ''}
                      onChange={(e) => updateFilter('dateRange', { ...filters.dateRange, end: e.target.value })}
                      className="flex-1 rounded-md border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Saved Searches */}
      {savedSearches.length > 0 && (
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Saved Searches</h3>
          <div className="flex flex-wrap gap-2">
            {savedSearches.slice(0, 5).map((saved, index) => (
              <button
                key={index}
                onClick={() => loadSavedSearch(saved)}
                className="px-3 py-1 bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-full text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
              >
                {saved.query}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Recent Searches */}
      {recentSearches.length > 0 && !results.length && (
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Recent Searches</h3>
          <div className="flex flex-wrap gap-2">
            {recentSearches.slice(0, 8).map((recent, index) => (
              <button
                key={index}
                onClick={() => setQuery(recent)}
                className="px-3 py-1 bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-full text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
              >
                <Clock className="inline h-3 w-3 mr-1" />
                {recent}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <X className="h-5 w-5 text-red-400" />
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Search Stats */}
      {stats && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{stats.totalResults}</div>
              <div className="text-sm text-blue-800 dark:text-blue-200">Results</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{stats.searchTime.toFixed(2)}s</div>
              <div className="text-sm text-blue-800 dark:text-blue-200">Search Time</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{formatScore(stats.avgScore)}</div>
              <div className="text-sm text-blue-800 dark:text-blue-200">Avg Score</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{stats.topSources.length}</div>
              <div className="text-sm text-blue-800 dark:text-blue-200">Sources</div>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {filteredResults.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Search Results ({filteredResults.length})
            </h2>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => {
                  const data = filteredResults.map(r => ({
                    title: r.title,
                    url: r.url,
                    score: r.score,
                    source: r.source,
                    snippet: r.snippet
                  }));
                  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `search-results-${new Date().toISOString().split('T')[0]}.json`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}
                className="flex items-center space-x-1 px-3 py-1 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              >
                <Download className="h-4 w-4" />
                <span>Export</span>
              </button>
            </div>
          </div>

          <div className="space-y-4">
            {filteredResults.map((result) => (
              <div key={result.id} className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 hover:shadow-md transition-shadow">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white hover:text-blue-600 dark:hover:text-blue-400">
                        <a href={result.url} target="_blank" rel="noopener noreferrer">
                          {result.title}
                        </a>
                      </h3>
                      <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs rounded-full">
                        {formatScore(result.score)}
                      </span>
                    </div>
                    
                    {result.hierarchical_path && (
                      <div className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                        {result.hierarchical_path.join(' > ')}
                      </div>
                    )}
                    
                    <p className="text-gray-600 dark:text-gray-300 mb-3 leading-relaxed">
                      {result.snippet}
                    </p>
                    
                    <div className="flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                      <div className="flex items-center space-x-1">
                        <Globe className="h-4 w-4" />
                        <span>{result.source}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Clock className="h-4 w-4" />
                        <span>{formatDate(result.retrieved_at)}</span>
                      </div>
                      {result.language && (
                        <div className="flex items-center space-x-1">
                          <span className="px-2 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-xs">
                            {result.language}
                          </span>
                        </div>
                      )}
                      {result.metadata?.reading_time && (
                        <div className="flex items-center space-x-1">
                          <BookOpen className="h-4 w-4" />
                          <span>{calculateReadingTime(result.metadata.word_count)}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && results.length === 0 && query && (
        <div className="text-center py-12">
          <Search className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">No results found</h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Try adjusting your search terms or filters
          </p>
        </div>
      )}

      {/* Welcome State */}
      {!loading && !error && results.length === 0 && !query && (
        <div className="text-center py-12">
          <TrendingUp className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">Welcome to DocFoundry</h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Start by entering a search query above to discover relevant documents
          </p>
        </div>
      )}
    </div>
  );
};

export default EnhancedSearchInterface;