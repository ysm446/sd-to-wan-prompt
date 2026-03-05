"""Entry point for WAN Prompt Generator."""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Keep console output stable on Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.utils.config_loader import ConfigLoader


def run_api(config: dict, host: str, port: int) -> None:
    """Run API mode for desktop wrappers like Electron/Tauri."""
    import uvicorn
    from src.api.server import create_app

    app = create_app(config)
    uvicorn.run(app, host=host, port=port, log_level="info")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WAN Prompt Generator")
    parser.add_argument("--host", default="127.0.0.1", help="API host.")
    parser.add_argument("--port", type=int, default=7861, help="API port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ConfigLoader().load_settings()

    print(f"=== {config['app']['name']} v{config['app']['version']} ===")
    print(f"Models Directory: {config['paths']['models_dir']}")
    print(f"Image Folder: {config['paths']['image_folder']}")
    print("-" * 50)

    run_api(config, args.host, args.port)


if __name__ == "__main__":
    main()
