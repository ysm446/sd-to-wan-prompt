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


def run_gradio(config: dict) -> None:
    """Run the legacy Gradio UI mode."""
    from src.ui.gradio_app import PromptAnalyzerUI

    ui = PromptAnalyzerUI(config)
    interface = ui.create_interface()
    try:
        interface.launch(
            server_port=config["ui"]["server_port"],
            share=config["ui"]["share"],
            inbrowser=True,
            theme=config["ui"]["theme"],
        )
    except OSError as exc:
        if "Cannot find empty port" not in str(exc):
            raise
        print(f"Port {config['ui']['server_port']} is in use. Retrying with another port.")
        interface.launch(
            share=config["ui"]["share"],
            inbrowser=True,
            theme=config["ui"]["theme"],
        )


def run_api(config: dict, host: str, port: int) -> None:
    """Run API mode for desktop wrappers like Electron/Tauri."""
    import uvicorn
    from src.api.server import create_app

    app = create_app(config)
    uvicorn.run(app, host=host, port=port, log_level="info")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WAN Prompt Generator")
    parser.add_argument(
        "--mode",
        choices=["gradio", "api"],
        default="gradio",
        help="Startup mode. Use 'api' for Electron/Tauri integration.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="API host in --mode api.")
    parser.add_argument("--port", type=int, default=7861, help="API port in --mode api.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ConfigLoader().load_settings()

    print(f"=== {config['app']['name']} v{config['app']['version']} ===")
    print(f"Mode: {args.mode}")
    print(f"Models Directory: {config['paths']['models_dir']}")
    print(f"Image Folder: {config['paths']['image_folder']}")
    print("-" * 50)

    if args.mode == "api":
        run_api(config, args.host, args.port)
    else:
        run_gradio(config)


if __name__ == "__main__":
    main()
