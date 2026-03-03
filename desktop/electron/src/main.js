const { app, BrowserWindow, Menu } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

const API_PORT = process.env.WAN_API_PORT || '7861';
const API_HOST = process.env.WAN_API_HOST || '127.0.0.1';
let backendProcess = null;

function createWindow() {
  Menu.setApplicationMenu(null);

  const win = new BrowserWindow({
    width: 1180,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.setMenuBarVisibility(false);
  win.removeMenu();
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
}

function startBackend() {
  const projectRoot = path.resolve(__dirname, '..', '..', '..');
  const python = process.env.PYTHON_EXECUTABLE || 'python';

  backendProcess = spawn(
    python,
    ['app.py', '--mode', 'api', '--host', API_HOST, '--port', API_PORT],
    {
      cwd: projectRoot,
      env: {
        ...process.env,
        PYTHONIOENCODING: 'utf-8',
        PYTHONUTF8: '1',
      },
      stdio: 'inherit',
      shell: false,
    }
  );

  backendProcess.on('exit', (code) => {
    console.log(`Backend process exited with code ${code}`);
    backendProcess = null;
  });
}

app.whenReady().then(() => {
  startBackend();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (backendProcess) {
    backendProcess.kill();
  }
});
