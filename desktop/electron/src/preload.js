const { contextBridge, webUtils } = require('electron');

const API_PORT = process.env.WAN_API_PORT || '7861';
const API_HOST = process.env.WAN_API_HOST || '127.0.0.1';
const BASE_URL = `http://${API_HOST}:${API_PORT}`;

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });

  const body = await res.json();
  if (!res.ok) {
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return body;
}

contextBridge.exposeInMainWorld('wanApi', {
  getBaseUrl: () => BASE_URL,
  health: () => request('/health'),
  listModels: () => request('/models'),
  downloadModel: (repoId, localName) =>
    request('/models/download', {
      method: 'POST',
      body: JSON.stringify({ repo_id: repoId, local_name: localName || null }),
    }),
  loadModel: (modelPath) =>
    request('/models/load', {
      method: 'POST',
      body: JSON.stringify({ model_path: modelPath }),
    }),
  unloadModel: () => request('/models/unload', { method: 'POST' }),
  parseImage: (imagePath) =>
    request('/image/parse', {
      method: 'POST',
      body: JSON.stringify({ image_path: imagePath }),
    }),
  getPathForFile: (file) => webUtils.getPathForFile(file),
  generate: (payload) =>
    request('/generate', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getSettings: () => request('/settings'),
  saveSettings: (values) =>
    request('/settings', {
      method: 'POST',
      body: JSON.stringify({ values }),
    }),
  savePromptToFile: (saveDir, outputText, additionalInstruction) =>
    request('/prompt/save', {
      method: 'POST',
      body: JSON.stringify({
        save_dir: saveDir,
        output_text: outputText,
        additional_instruction: additionalInstruction || '',
      }),
    }),
});
