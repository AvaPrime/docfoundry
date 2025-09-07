# DocFoundry Capture - Chrome Extension

🔗 **Intelligent Web Page Capture for DocFoundry**

Capture web pages and documentation directly into your DocFoundry knowledge base for intelligent search and retrieval.

## ✨ Features

- **One-Click Capture**: Save any web page to your DocFoundry research collection
- **Automatic Processing**: Pages are automatically processed and indexed for search
- **Metadata Extraction**: Captures page title, URL, and content with proper formatting
- **Seamless Integration**: Works directly with your local DocFoundry API
- **Research Organization**: Saves captured content to `docs/research/` with structured frontmatter

## 🚀 Installation

### Development Installation

1. **Enable Developer Mode**:
   - Open Chrome and navigate to `chrome://extensions`
   - Toggle "Developer mode" in the top-right corner

2. **Load the Extension**:
   - Click "Load unpacked"
   - Navigate to and select the `browser/chrome-extension` directory
   - The DocFoundry Capture icon should appear in your toolbar

3. **Verify Installation**:
   - Ensure your DocFoundry API is running at `http://127.0.0.1:8001`
   - The extension will communicate with this endpoint for page capture

## 📖 Usage

### Basic Capture
1. Navigate to any web page you want to save
2. Click the DocFoundry Capture icon in your browser toolbar
3. The page will be automatically captured and saved to your research collection
4. A notification will confirm successful capture

### What Gets Captured
- **Page Content**: Main text content, cleaned and formatted
- **Metadata**: Title, URL, capture timestamp
- **Structure**: Preserved as Markdown with frontmatter
- **Location**: Saved to `docs/research/` in your DocFoundry instance

### Integration with DocFoundry
Captured pages are immediately available for:
- **Search**: Find content using DocFoundry's hybrid search
- **API Access**: Query via REST API endpoints
- **VS Code Extension**: Search from within your editor
- **MCP Integration**: Access through compatible AI agents

## 🔧 Configuration

### API Endpoint
By default, the extension connects to `http://127.0.0.1:8001/capture`. If your DocFoundry API runs on a different port or host, you'll need to modify the endpoint in the extension code.

### Permissions
The extension requires:
- **Active Tab**: To access the current page content
- **Scripting**: To extract page content
- **Host Permissions**: To capture content from any website

## 🛠️ Development

### File Structure
```
browser/chrome-extension/
├── manifest.json          # Extension configuration
├── background.js          # Service worker for capture logic
├── README.md             # This documentation
└── icons/                # Extension icons (if added)
```

### Building from Source
```bash
# Navigate to the extension directory
cd browser/chrome-extension

# The extension is ready to load - no build step required
# Simply load as unpacked extension in Chrome
```

## 🔍 Troubleshooting

### Common Issues

**Extension not capturing pages:**
- Verify DocFoundry API is running at `http://127.0.0.1:8001`
- Check browser console for error messages
- Ensure the API `/capture` endpoint is accessible

**Permission errors:**
- Reload the extension after making changes
- Check that all required permissions are granted

**Content not appearing in search:**
- Rebuild your DocFoundry index: `python indexer/build_index.py`
- Verify files are being saved to `docs/research/`

### Getting Help
- Check the main DocFoundry documentation
- Review API logs for capture endpoint errors
- Inspect browser developer tools for extension errors

## 🔗 Related
- [DocFoundry Main Documentation](../../README.md)
- [VS Code Extension](../../extensions/vscode/)
- [API Documentation](../../server/)

---

**DocFoundry Capture** - Seamlessly build your intelligent documentation knowledge base from the web. 🌐
