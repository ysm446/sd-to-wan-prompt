const byId = (id) => document.getElementById(id);

const health = byId('health');
const healthStatus = byId('healthStatus');
const modelSelect = byId('modelSelect');
const metadataOut = byId('metadataOut');
const outputOut = byId('outputOut');
const statusOut = byId('statusOut');
const generateBtn = byId('generateBtn');
const dropZone = byId('dropZone');
const dropHint = byId('dropHint');
const imagePreview = byId('imagePreview');

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setStatus(message, detail = null) {
  const payload = detail ? { message, detail } : { message };
  statusOut.textContent = JSON.stringify(payload, null, 2);
}

function normalizeMultilineText(value) {
  const text = String(value || '');
  // Session JSON may contain escaped newlines (\\n). Render them as real newlines.
  return text.replace(/\\n/g, '\n').trim();
}

function renderMetadataPretty(metadata) {
  if (!metadata || typeof metadata !== 'object') {
    metadataOut.textContent = '-';
    return;
  }

  const lines = [];
  const filename = metadata.filename || '';
  const size = Array.isArray(metadata.size) ? `${metadata.size[0]} x ${metadata.size[1]}` : '';

  if (filename) lines.push(`File: ${filename}`);
  if (size) lines.push(`Size: ${size}`);
  if (filename || size) lines.push('');

  const prompt = normalizeMultilineText(metadata.prompt);
  const negative = normalizeMultilineText(metadata.negative_prompt);
  const settings = metadata.settings && typeof metadata.settings === 'object' ? metadata.settings : {};

  lines.push('Prompt:');
  lines.push(prompt || '-');
  lines.push('');
  lines.push('Negative Prompt:');
  lines.push(negative || '-');
  lines.push('');
  lines.push('Settings:');

  const settingKeys = Object.keys(settings);
  if (settingKeys.length === 0) {
    lines.push('-');
  } else {
    settingKeys.forEach((key) => {
      lines.push(`${key}: ${settings[key]}`);
    });
  }

  metadataOut.textContent = lines.join('\n');
}

async function normalizeDroppedPath(file, event) {
  if (file && typeof window.wanApi?.getPathForFile === 'function') {
    try {
      const nativePath = await window.wanApi.getPathForFile(file);
      if (nativePath) return nativePath;
    } catch (_err) {
      // Fall through to text-based extraction.
    }
  }

  if (file && file.path) return file.path;

  const dataTransfer = event?.dataTransfer;
  const textUri = dataTransfer?.getData('text/uri-list') || '';
  const textPlain = dataTransfer?.getData('text/plain') || '';
  const downloadUrl = dataTransfer?.getData('DownloadURL') || '';
  const raw = textUri || textPlain || downloadUrl;
  if (!raw) return '';

  let value = raw.split('\n').find((line) => line && !line.startsWith('#')) || '';
  value = value.trim();

  if (downloadUrl && value.includes(':')) {
    const parts = value.split(':');
    value = parts[parts.length - 1] || value;
  }

  if (!value.toLowerCase().startsWith('file://')) {
    return value.replace(/\//g, '\\');
  }

  value = decodeURIComponent(value.replace(/^file:\/\//i, ''));
  if (/^\/[A-Za-z]:\//.test(value)) {
    value = value.slice(1);
  }
  return value.replace(/\//g, '\\');
}

function isImagePath(value) {
  return /\.(png|jpe?g|webp|bmp|gif)$/i.test(value || '');
}

function isJsonPath(value) {
  return /\.json$/i.test(value || '');
}

function toFileUrl(winPath) {
  const normalized = (winPath || '').replace(/\\/g, '/');
  if (/^[a-zA-Z]:\//.test(normalized)) {
    return `file:///${encodeURI(normalized)}`;
  }
  return `file://${encodeURI(normalized)}`;
}

function updateImagePreview(imagePath) {
  if (!imagePreview) return;
  if (!imagePath) {
    imagePreview.style.display = 'none';
    imagePreview.removeAttribute('src');
    if (dropHint) dropHint.style.display = 'block';
    return;
  }
  imagePreview.src = toFileUrl(imagePath);
  imagePreview.style.display = 'block';
  if (dropHint) dropHint.style.display = 'none';
}

function collectSettingsPayload() {
  const selectedSections = Array.from(document.querySelectorAll('.section-check:checked')).map((el) => el.value);
  return {
    inference_settings: {
      temperature: Number(byId('temperatureInput').value || 0.7),
      max_tokens: Number(byId('maxTokensInput').value || 1024),
    },
    generation_settings: {
      language: byId('languageSelect').value || 'English',
      style_preset: byId('styleSelect').value || null,
    },
    output_sections: selectedSections,
    auto_unload: Boolean(byId('autoUnloadInput').checked),
  };
}

function applySettings(settings) {
  if (!settings) return;

  const inf = settings.inference_settings || {};
  const gen = settings.generation_settings || {};

  if (typeof inf.temperature !== 'undefined') byId('temperatureInput').value = inf.temperature;
  if (typeof inf.max_tokens !== 'undefined') byId('maxTokensInput').value = inf.max_tokens;
  if (typeof gen.language !== 'undefined') byId('languageSelect').value = gen.language;
  if (typeof gen.style_preset !== 'undefined') byId('styleSelect').value = gen.style_preset || '';
  if (typeof settings.auto_unload !== 'undefined') byId('autoUnloadInput').checked = Boolean(settings.auto_unload);
  if (Array.isArray(settings.output_sections) && settings.output_sections.length > 0) {
    const selected = new Set(settings.output_sections);
    document.querySelectorAll('.section-check').forEach((el) => {
      el.checked = selected.has(el.value);
    });
  }
}

async function refreshHealth() {
  try {
    const res = await window.wanApi.health();
    const text = `Backend: ${res.status} (${res.version})`;
    if (health) health.textContent = text;
    if (healthStatus) healthStatus.textContent = text;
    return true;
  } catch (err) {
    const text = `Backend: down (${err.message})`;
    if (health) health.textContent = text;
    if (healthStatus) healthStatus.textContent = text;
    return false;
  }
}

async function refreshModels() {
  setStatus('Refreshing local models...');
  const res = await window.wanApi.listModels();
  modelSelect.innerHTML = '';
  if (!res.models || res.models.length === 0) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'No models found';
    modelSelect.appendChild(option);
    setStatus('No models found under models directory.');
    return;
  }

  for (const model of res.models) {
    const option = document.createElement('option');
    option.value = model.path;
    option.textContent = `${model.name} (${model.size})`;
    if (model.is_selected) option.selected = true;
    modelSelect.appendChild(option);
  }
  setStatus(`Loaded ${res.models.length} local models.`);
}

async function loadModel() {
  const modelPath = modelSelect.value;
  if (!modelPath) {
    throw new Error('Please select a model first.');
  }
  setStatus('Loading model. This may take a while...');
  const state = await window.wanApi.loadModel(modelPath);
  setStatus('Model loaded.', state);
}

async function unloadModel() {
  setStatus('Unloading model...');
  const state = await window.wanApi.unloadModel();
  setStatus('Model unloaded.', state);
}

async function downloadModel() {
  const repoId = byId('repoIdInput').value.trim();
  const localName = byId('localNameInput').value.trim();

  if (!repoId) {
    setStatus('Repository ID is required. Example: Qwen/Qwen2.5-VL-7B-Instruct');
    return;
  }

  setStatus('Downloading model from Hugging Face...');
  const res = await window.wanApi.downloadModel(repoId, localName || null);
  setStatus('Model download completed.', res);
  await refreshModels();
}

async function parseImage() {
  const imagePath = byId('imagePathInput').value.trim();
  if (!imagePath) {
    throw new Error('Please enter image path.');
  }
  updateImagePreview(imagePath);
  setStatus('Parsing image metadata...');
  const res = await window.wanApi.parseImage(imagePath);
  renderMetadataPretty(res.metadata);
  setStatus('Image metadata parsed.');
}

function applyLoadedSession(session) {
  if (!session) return;
  const displayPath = session.image_path || session.image_url || '';
  if (displayPath) {
    byId('imagePathInput').value = displayPath;
    updateImagePreview(displayPath);
  }
  if (session.metadata) {
    renderMetadataPretty(session.metadata);
  }
  if (typeof session.prompt === 'string') {
    outputOut.value = session.prompt;
  }
  if (typeof session.additional_instruction === 'string') {
    byId('instructionInput').value = session.additional_instruction;
  }
}

function bindDropZone() {
  if (!dropZone) return;

  const preventDefaults = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  // Prevent browser-like navigation behavior on file drop.
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((name) => {
    window.addEventListener(name, preventDefaults);
  });

  ['dragenter', 'dragover'].forEach((name) => {
    dropZone.addEventListener(name, (event) => {
      preventDefaults(event);
      dropZone.classList.add('active');
    });
  });

  ['dragleave', 'drop'].forEach((name) => {
    dropZone.addEventListener(name, (event) => {
      preventDefaults(event);
      dropZone.classList.remove('active');
    });
  });

  dropZone.addEventListener('drop', async (event) => {
    const files = event.dataTransfer?.files;
    const file = files && files.length > 0 ? files[0] : null;
    const droppedPath = await normalizeDroppedPath(file, event);
    if (!droppedPath) {
      const types = Array.from(event.dataTransfer?.types || []);
      setStatus('Failed to read dropped file path.', { dataTransferTypes: types });
      return;
    }

    if (isJsonPath(droppedPath)) {
      setStatus('JSON dropped. Loading session...');
      try {
        const session = await window.wanApi.loadSessionJson(droppedPath);
        applyLoadedSession(session);
        setStatus('Session JSON loaded.', {
          json_path: session.json_path,
          image_filename: session.image_filename,
          image_path: session.image_path || '(not found)',
        });
      } catch (err) {
        const detail = err && err.stack ? err.stack : null;
        setStatus(`Failed to load session JSON: ${err.message}`, detail);
      }
      return;
    }

    if (file?.type?.startsWith('image/') || isImagePath(droppedPath)) {
      byId('imagePathInput').value = droppedPath;
      updateImagePreview(droppedPath);
      setStatus('Image dropped. Parsing metadata...');
      await runSafe(parseImage);
      return;
    }

    setStatus('Unsupported file type. Drop an image or session JSON.');
  });
}

function bindTabs() {
  const tabButtons = Array.from(document.querySelectorAll('.tab-btn'));
  const tabPanels = Array.from(document.querySelectorAll('.tab-panel'));
  if (tabButtons.length === 0 || tabPanels.length === 0) return;

  const activateTab = (targetId) => {
    tabButtons.forEach((btn) => {
      const isActive = btn.dataset.tabTarget === targetId;
      btn.classList.toggle('active', isActive);
    });

    tabPanels.forEach((panel) => {
      panel.classList.toggle('active', panel.id === targetId);
    });
  };

  tabButtons.forEach((btn) => {
    btn.addEventListener('click', () => activateTab(btn.dataset.tabTarget));
  });
}

async function saveSettings() {
  const payload = collectSettingsPayload();
  const saved = await window.wanApi.saveSettings(payload);
  applySettings(saved);
  setStatus('Settings saved.', saved);
}

async function loadSettings() {
  const settings = await window.wanApi.getSettings();
  applySettings(settings);
  if (settings.last_model) {
    for (const option of modelSelect.options) {
      if (option.value === settings.last_model) {
        option.selected = true;
        break;
      }
    }
  }
  setStatus('Settings loaded.');
}

async function generate() {
  const imagePath = byId('imagePathInput').value.trim();
  if (!imagePath) {
    throw new Error('Set image path and parse metadata first.');
  }

  const payload = {
    additional_instruction: byId('instructionInput').value || '',
    style_preset: byId('styleSelect').value || null,
    output_language: byId('languageSelect').value,
    output_sections: Array.from(document.querySelectorAll('.section-check:checked')).map((el) => el.value),
    temperature: Number(byId('temperatureInput').value || 0.7),
    max_tokens: Number(byId('maxTokensInput').value || 1024),
    auto_unload: Boolean(byId('autoUnloadInput').checked),
  };

  if (Number.isNaN(payload.temperature) || payload.temperature < 0 || payload.temperature > 2) {
    throw new Error('Temperature must be between 0 and 2.');
  }
  if (Number.isNaN(payload.max_tokens) || payload.max_tokens < 64) {
    throw new Error('Max tokens must be 64 or higher.');
  }
  if (!payload.output_sections || payload.output_sections.length === 0) {
    throw new Error('Please select at least one output section.');
  }

  generateBtn.disabled = true;
  const originalLabel = generateBtn.textContent;
  generateBtn.textContent = 'Generating...';
  setStatus('Generating WAN prompt...');

  try {
    await saveSettings();
    outputOut.value = '';

    const response = await fetch(`${window.wanApi.getBaseUrl()}/generate/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok || !response.body) {
      const body = await response.text();
      throw new Error(body || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let doneEvent = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        const event = JSON.parse(trimmed);

        if (event.type === 'chunk') {
          outputOut.value += event.content || '';
        } else if (event.type === 'done') {
          doneEvent = event;
        } else if (event.type === 'error') {
          throw new Error(event.message || 'Streaming generation failed');
        }
      }
    }

    if (doneEvent) {
      setStatus('Generation completed.', doneEvent.model || null);
      if (payload.auto_unload) {
        await refreshModels();
      }
    } else {
      setStatus('Generation finished (no done event).');
    }
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = originalLabel;
  }
}

async function savePromptText() {
  const outputText = outputOut.value.trim();
  const additionalInstruction = byId('instructionInput').value || '';

  if (!byId('imagePathInput').value.trim()) {
    throw new Error('Load an image before saving session JSON.');
  }

  setStatus('Saving session JSON next to source image...');
  const res = await window.wanApi.saveSessionJson(outputText, additionalInstruction);
  setStatus('Session JSON saved.', { saved_path: res.saved_path });
  await saveSettings();
}

function bindEvents() {
  bindTabs();
  byId('refreshModelsBtn').addEventListener('click', () => runSafe(refreshModels));
  byId('loadModelBtn').addEventListener('click', () => runSafe(loadModel));
  byId('unloadModelBtn').addEventListener('click', () => runSafe(unloadModel));
  byId('downloadModelBtn').addEventListener('click', () => runSafe(downloadModel));
  byId('parseImageBtn').addEventListener('click', () => runSafe(parseImage));
  byId('savePromptBtn').addEventListener('click', () => runSafe(savePromptText));
  byId('saveSettingsBtn').addEventListener('click', () => runSafe(saveSettings));
  byId('generateBtn').addEventListener('click', () => runSafe(generate));
  byId('copyPromptBtn').addEventListener('click', async () => {
    const text = outputOut.value.trim();
    if (!text) return;
    await navigator.clipboard.writeText(text);
    const btn = byId('copyPromptBtn');
    btn.classList.add('copied');
    setTimeout(() => btn.classList.remove('copied'), 1500);
  });
  bindDropZone();
}

async function runSafe(fn) {
  try {
    await fn();
  } catch (err) {
    const detail = err && err.stack ? err.stack : null;
    setStatus(`Error: ${err.message}`, detail);
  }
}

async function waitForBackend(maxRetry = 20, intervalMs = 500) {
  for (let i = 0; i < maxRetry; i += 1) {
    const ok = await refreshHealth();
    if (ok) return true;
    await sleep(intervalMs);
  }
  return false;
}

(async function init() {
  if (!window.wanApi) {
    if (health) health.textContent = 'Backend: bridge unavailable';
    if (healthStatus) healthStatus.textContent = 'Backend: bridge unavailable';
    setStatus('Error: Electron preload bridge (window.wanApi) is not available.');
    return;
  }

  bindEvents();
  const ready = await waitForBackend();
  if (!ready) {
    setStatus('Error: backend did not become ready in time.');
    return;
  }

  await runSafe(refreshModels);
  await runSafe(loadSettings);
})();
