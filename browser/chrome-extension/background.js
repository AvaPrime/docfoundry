chrome.action.onClicked.addListener(async (tab) => {
  if (!tab.id || !tab.url) return;
  const [{ result: content }] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => document.querySelector('main, article, #content')?.innerText || document.body.innerText.slice(0, 5000)
  });
  const title = tab.title || tab.url;
  try {
    await fetch('http://127.0.0.1:8001/capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: tab.url, title, content })
    });
  } catch (e) {
    console.error('DocFoundry capture failed', e);
  }
});
