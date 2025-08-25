import * as vscode from 'vscode';
import * as https from 'https';
import * as http from 'http';
import * as url from 'url';

function getApiBase(): string {
  return process.env.DOCFOUNDRY_API || 'http://127.0.0.1:8001';
}

function httpPostJson(endpoint: string, data: any): Promise<any> {
  return new Promise((resolve, reject) => {
    const u = new url.URL(endpoint);
    const payload = Buffer.from(JSON.stringify(data));

    const isHttps = u.protocol === 'https:';
    const opts: https.RequestOptions = {
      hostname: u.hostname,
      port: u.port ? Number(u.port) : (isHttps ? 443 : 80),
      path: u.pathname + u.search,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': payload.length
      }
    };

    const req = (isHttps ? https : http).request(opts, (res) => {
      let chunks: Buffer[] = [];
      res.on('data', d => chunks.push(d));
      res.on('end', () => {
        try {
          const txt = Buffer.concat(chunks).toString('utf-8');
          resolve(JSON.parse(txt));
        } catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.write(payload);
    req.end();
  });
}

export function activate(context: vscode.ExtensionContext) {
  const search = vscode.commands.registerCommand('docfoundry.search', async () => {
    const q = await vscode.window.showInputBox({ prompt: 'Search DocFoundry', placeHolder: 'rate limit, provider routing, mcp server...' });
    if (!q) { return; }
    const api = getApiBase();
    try {
      const res = await httpPostJson(`${api}/search`, { q, k: 10 });
      const items = (res.results || []).map((r: any) => ({
        label: `${r.title} › ${r.heading}`,
        description: r.source_url || r.path,
        detail: r.snippet?.replace(/<[^>]+>/g, '') || '',
        path: r.path,
        anchor: r.anchor
      }));

      const picked = await vscode.window.showQuickPick(items, { matchOnDetail: true });
      if (picked) {
        const localMkDocs = 'http://127.0.0.1:8000';
        const anchor = picked.anchor ? ('#' + picked.anchor) : '';
        const path = picked.path.replace('docs/', '');
        vscode.env.openExternal(vscode.Uri.parse(`${localMkDocs}/${path}${anchor}`));
      }
    } catch (e: any) {
      vscode.window.showErrorMessage(`DocFoundry search failed: ${e?.message || e}`);
    }
  });

  context.subscriptions.push(search);
}

export function deactivate() {}
