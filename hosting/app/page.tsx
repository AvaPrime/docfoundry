'use client'

import { useState, useEffect, useCallback } from 'react'
import { MagnifyingGlassIcon as SearchIcon, FunnelIcon as FilterIcon, BookOpenIcon, ClockIcon, TagIcon, ArrowTopRightOnSquareIcon } from '@heroicons/react/24/outline'

interface SearchResult {
  doc_id: string
  chunk_id: string
  score: number
  title: string
  url: string
  snippet: string
  retrieved_at: string
  h_path?: string[]
}

interface SearchResponse {
  query: string
  results: SearchResult[]
}

interface SearchFilters {
  source: string[]
  lang: string
}

export default function Home() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [filters, setFilters] = useState<SearchFilters>({ source: [], lang: '' })
  const [showFilters, setShowFilters] = useState(false)
  const [searchMode, setSearchMode] = useState<'hybrid' | 'semantic' | 'fulltext'>('hybrid')
  const [recentSearches, setRecentSearches] = useState<string[]>([])

  // Load recent searches from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('docfoundry-recent-searches')
    if (saved) {
      setRecentSearches(JSON.parse(saved))
    }
  }, [])

  const saveRecentSearch = useCallback((searchQuery: string) => {
    const updated = [searchQuery, ...recentSearches.filter(q => q !== searchQuery)].slice(0, 5)
    setRecentSearches(updated)
    localStorage.setItem('docfoundry-recent-searches', JSON.stringify(updated))
  }, [recentSearches])

  const performSearch = useCallback(async (searchQuery: string) => {
    if (!searchQuery.trim()) return

    setLoading(true)
    setError(null)
    
    try {
      const endpoint = searchMode === 'semantic' ? '/search/semantic' : 
                      searchMode === 'fulltext' ? '/search' : '/search'
      
      const requestBody = {
        q: searchQuery,
        query: searchQuery,
        top_k: 10,
        limit: 10,
        hybrid: searchMode === 'hybrid',
        use_reranker: true,
        filters: {
          source: filters.source.length > 0 ? filters.source : undefined,
          lang: filters.lang || undefined
        }
      }

      const response = await fetch(`http://localhost:8001${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      })

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`)
      }

      const data: SearchResponse = await response.json()
      setResults(data.results || [])
      saveRecentSearch(searchQuery)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed')
      setResults([])
    } finally {
      setLoading(false)
    }
  }, [searchMode, filters, saveRecentSearch])

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    performSearch(query)
  }

  const handleRecentSearch = (recentQuery: string) => {
    setQuery(recentQuery)
    performSearch(recentQuery)
  }

  const formatScore = (score: number) => {
    return (score * 100).toFixed(1)
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString()
  }

  return (
    <div className="py-8">
      {/* Search Interface */}
      <div className="search-container mb-8">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-foreground mb-2">
            Search Your Documents
          </h2>
          <p className="text-lg text-muted-foreground">
            Find exactly what you're looking for with AI-powered search
          </p>
        </div>

        {/* Search Form */}
        <form onSubmit={handleSearch} className="mb-6">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1 relative">
              <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search documentation..."
                className="search-input pl-10"
                disabled={loading}
              />
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setShowFilters(!showFilters)}
                className="px-4 py-3 border border-gray-300 rounded-xl hover:bg-gray-50 focus:ring-2 focus:ring-blue-500 transition-all duration-200"
              >
                <FilterIcon className="w-5 h-5" />
              </button>
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className="search-button disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <div className="loading-spinner" />
                ) : (
                  'Search'
                )}
              </button>
            </div>
          </div>
        </form>

        {/* Search Mode Toggle */}
        <div className="flex justify-center mb-6">
          <div className="inline-flex rounded-lg border border-gray-200 bg-white p-1">
            {(['hybrid', 'semantic', 'fulltext'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setSearchMode(mode)}
                className={`px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
                  searchMode === mode
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-gray-700 hover:bg-gray-50'
                }`}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Filters Panel */}
        {showFilters && (
          <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-gray-200 mb-6 slide-up">
            <h3 className="text-lg font-semibold mb-4">Search Filters</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Source
                </label>
                <input
                  type="text"
                  placeholder="e.g., openrouter, docs"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  onChange={(e) => setFilters(prev => ({ ...prev, source: e.target.value ? [e.target.value] : [] }))}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Language
                </label>
                <select
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  onChange={(e) => setFilters(prev => ({ ...prev, lang: e.target.value }))}
                >
                  <option value="">Any language</option>
                  <option value="en">English</option>
                  <option value="es">Spanish</option>
                  <option value="fr">French</option>
                  <option value="de">German</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Recent Searches */}
        {recentSearches.length > 0 && !results.length && !loading && (
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
              <ClockIcon className="inline-block w-4 h-4 mr-1" />
              Recent Searches
            </h3>
            <div className="flex flex-wrap gap-2">
              {recentSearches.map((recentQuery, index) => (
                <button
                  key={index}
                  onClick={() => handleRecentSearch(recentQuery)}
                  className="filter-chip"
                >
                  {recentQuery}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="search-container mb-6">
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700">
            <p className="font-medium">Search Error</p>
            <p className="text-sm">{error}</p>
          </div>
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="search-container">
          <div className="mb-4">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Found {results.length} results for "{query}"
            </p>
          </div>
          
          <div className="space-y-4">
            {results.map((result, index) => (
              <div key={`${result.doc_id}-${result.chunk_id}`} className="result-card fade-in">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
                      {result.title || 'Untitled Document'}
                    </h3>
                    {result.h_path && result.h_path.length > 0 && (
                      <div className="flex items-center text-sm text-gray-500 mb-2">
                        <TagIcon className="w-4 h-4 mr-1" />
                        {result.h_path.join(' > ')}
                      </div>
                    )}
                  </div>
                  <div className="flex items-center space-x-2 ml-4">
                    <span className="text-xs font-medium px-2 py-1 bg-blue-100 text-blue-800 rounded-full">
                      {formatScore(result.score)}% match
                    </span>
                    {result.url && (
                      <a
                        href={result.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-800 transition-colors"
                      >
                        <ArrowTopRightOnSquareIcon className="w-4 h-4" />
                      </a>
                    )}
                  </div>
                </div>
                
                <p className="text-gray-700 dark:text-gray-300 mb-3 leading-relaxed">
                  {result.snippet}
                </p>
                
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>Document ID: {result.doc_id}</span>
                  <span>Retrieved: {formatDate(result.retrieved_at)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && results.length === 0 && query && (
        <div className="search-container text-center py-12">
          <BookOpenIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No results found
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            Try adjusting your search terms or filters
          </p>
        </div>
      )}

      {/* Welcome State */}
      {!query && !loading && (
        <div className="search-container text-center py-12">
          <BookOpenIcon className="w-20 h-20 text-blue-600 mx-auto mb-6" />
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            Welcome to DocFoundry
          </h2>
          <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Search through your documentation using advanced AI-powered search. 
            Choose between hybrid search (combines keyword and semantic), 
            pure semantic search, or traditional full-text search.
          </p>
        </div>
      )}
    </div>
  )
}
