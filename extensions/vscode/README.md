# DocFoundry - VS Code Extension

🔍 **Intelligent Documentation Search for VS Code**

Search and retrieve documentation directly from your VS Code editor using DocFoundry's powerful hybrid search capabilities.

## ✨ Features

- **In-Editor Search**: Search your DocFoundry knowledge base without leaving VS Code
- **Hybrid Search**: Combines semantic and keyword search for accurate results
- **Quick Access**: Use Command Palette for instant documentation lookup
- **Multi-Source**: Search across all your configured documentation sources
- **Contextual Results**: Get relevant documentation based on your current work
- **Seamless Integration**: Works with your local DocFoundry API instance

## 🚀 Installation

### From VS Code Marketplace
*Coming soon - extension will be published to the marketplace*

### Development Installation

1. **Prerequisites**:
   - VS Code 1.85.0 or higher
   - Node.js and npm installed
   - DocFoundry API running locally

2. **Build the Extension**:
   ```bash
   cd extensions/vscode
   npm install
   npm run compile
   ```

3. **Install in VS Code**:
   - Open VS Code
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Extensions: Install from VSIX"
   - Navigate to the extension directory and select the generated `.vsix` file

4. **Verify Installation**:
   - Ensure DocFoundry API is running at `http://127.0.0.1:8001`
   - The extension will connect to this endpoint automatically

## 📖 Usage

### Basic Search

1. **Open Command Palette**: `Ctrl+Shift+P` (or `Cmd+Shift+P`)
2. **Search Command**: Type "DocFoundry: Search Docs"
3. **Enter Query**: Type your documentation search query
4. **Browse Results**: Select from the returned documentation snippets
5. **Insert Content**: Choose to insert relevant content into your editor

### Keyboard Shortcuts

- **Quick Search**: `Ctrl+Alt+D` (customizable)
- **Search Selection**: Highlight text and use `Ctrl+Alt+S` to search for it

### Search Types

The extension supports DocFoundry's search capabilities:
- **Semantic Search**: Understanding context and meaning
- **Keyword Search**: Exact term matching
- **Hybrid Search**: Best of both approaches (default)

## 🔧 Configuration

### Extension Settings

Configure the extension through VS Code settings:

```json
{
  "docfoundry.apiUrl": "http://127.0.0.1:8001",
  "docfoundry.searchLimit": 10,
  "docfoundry.searchType": "hybrid",
  "docfoundry.autoInsert": false,
  "docfoundry.showPreview": true
}
```

### Settings Description

- **apiUrl**: DocFoundry API endpoint (default: `http://127.0.0.1:8001`)
- **searchLimit**: Maximum number of search results (default: 10)
- **searchType**: Search method - `semantic`, `keyword`, or `hybrid` (default: `hybrid`)
- **autoInsert**: Automatically insert first result (default: false)
- **showPreview**: Show content preview in results (default: true)

### Environment Variables

Alternatively, set the API endpoint using environment variables:
```bash
export DOCFOUNDRY_API=http://127.0.0.1:8001
```

## 🛠️ Development

### Project Structure

```
extensions/vscode/
├── src/
│   ├── extension.ts      # Main extension logic
│   └── docfoundry.ts     # DocFoundry API client
├── package.json          # Extension manifest
├── tsconfig.json         # TypeScript configuration
├── README.md            # This documentation
└── dist/                # Compiled output
```

### Building from Source

```bash
# Install dependencies
npm install

# Compile TypeScript
npm run compile

# Watch for changes during development
npm run watch

# Package extension
npm run package
```

### API Integration

The extension communicates with DocFoundry's REST API:

```typescript
// Search endpoint
POST /search
{
  "q": "search query",
  "limit": 10,
  "search_type": "hybrid"
}

// Response format
{
  "results": [
    {
      "doc_id": "...",
      "title": "...",
      "content": "...",
      "score": 0.95,
      "metadata": {...}
    }
  ]
}
```

## 🔍 Troubleshooting

### Common Issues

**Extension not finding DocFoundry API:**
- Verify API is running: `curl http://127.0.0.1:8001/healthz`
- Check API URL in extension settings
- Review VS Code Developer Console for errors

**No search results:**
- Ensure your DocFoundry index is built: `python indexer/build_index.py`
- Verify you have content in your documentation sources
- Try different search terms or search types

**Extension not loading:**
- Check VS Code version compatibility (1.85.0+)
- Verify extension is properly compiled: `npm run compile`
- Review VS Code extension logs

### Debug Mode

Enable debug logging in VS Code settings:

```json
{
  "docfoundry.debug": true
}
```

## 🚀 Features Roadmap

### Current Features ✅
- Basic search functionality
- Command palette integration
- Configurable API endpoint
- Multiple search types

### Planned Features 🚧
- **Contextual Search**: Search based on current file/project context
- **Inline Documentation**: Hover providers for quick doc lookup
- **Snippet Management**: Save and organize frequently used documentation
- **Multi-Workspace**: Support for multiple DocFoundry instances
- **Offline Mode**: Cache frequently accessed documentation
- **Custom Keybindings**: User-configurable keyboard shortcuts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes in the `extensions/vscode/` directory
4. Test thoroughly with a local DocFoundry instance
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Guidelines

- Follow TypeScript best practices
- Add appropriate error handling
- Update documentation for new features
- Test with different VS Code versions
- Ensure compatibility with DocFoundry API changes

## 📄 License

This extension is part of the DocFoundry project and is licensed under the MIT License.

## 🔗 Related

- [DocFoundry Main Documentation](../../README.md)
- [Chrome Extension](../../browser/chrome-extension/)
- [API Documentation](../../server/)
- [DocFoundry GitHub Repository](https://github.com/codessa-prime/docfoundry)

---

**DocFoundry VS Code Extension** - Bring intelligent documentation search directly to your development workflow. 💻
