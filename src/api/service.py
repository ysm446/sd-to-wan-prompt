"""Application service layer for REST API usage."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Any, Dict, Generator, List, Optional

from src.core.image_parser import ImageParser
from src.core.model_manager import ModelManager
from src.core.vlm_interface import VLMInterface


class PromptService:
    """Stateful service used by API handlers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_manager = ModelManager(config["paths"]["models_dir"])
        self.current_vlm: Optional[VLMInterface] = None
        self.current_image_path: Optional[str] = None
        self.current_metadata: Optional[Dict[str, Any]] = None
        self.selected_model_path: Optional[str] = None
        self.settings_cache_file = Path("settings_cache.json")
        self.lock = threading.RLock()

    def list_models(self) -> List[Dict[str, Any]]:
        models = self.model_manager.list_local_models()
        settings = self.get_settings()
        last_model = settings.get("last_model")
        if last_model and self.selected_model_path is None:
            self.selected_model_path = last_model
        for model in models:
            model["is_selected"] = model["path"] == self.selected_model_path
        return models

    def download_model(self, repo_id: str, local_name: Optional[str]) -> Dict[str, Any]:
        if not repo_id:
            raise ValueError("repo_id is required")
        path = self.model_manager.download_model(repo_id=repo_id, local_name=local_name or None)
        return {"downloaded_path": path}

    def parse_image(self, image_path: str) -> Dict[str, Any]:
        normalized = self._normalize_image_path(image_path)
        path = Path(normalized)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {normalized}")

        with self.lock:
            metadata = ImageParser.extract_metadata(str(path))
            self.current_image_path = str(path)
            self.current_metadata = metadata
            return metadata

    def clear_image(self) -> None:
        with self.lock:
            self.current_image_path = None
            self.current_metadata = None

    def load_model(self, model_path: str) -> Dict[str, Any]:
        if not model_path:
            raise ValueError("model_path is required")

        with self.lock:
            if self.current_vlm is not None:
                self.current_vlm.unload_model()
                self.current_vlm = None

            settings = self._load_device_settings()
            self.current_vlm = VLMInterface(
                model_path=model_path,
                device=settings["device"],
                dtype=settings["dtype"],
            )
            self.selected_model_path = model_path
            self._save_last_model_path(model_path)
            return self._model_state()

    def unload_model(self) -> Dict[str, Any]:
        with self.lock:
            if self.current_vlm is not None:
                self.current_vlm.unload_model()
                self.current_vlm = None
            return self._model_state()

    def generate_prompt(
        self,
        additional_instruction: str,
        style_preset: Optional[str],
        output_language: str,
        output_sections: Optional[List[str]],
        temperature: float,
        max_tokens: int,
        auto_unload: bool,
    ) -> Dict[str, Any]:
        sections = output_sections or ["scene", "action", "camera", "style", "prompt"]

        with self.lock:
            if self.current_vlm is None:
                if not self.selected_model_path:
                    raise RuntimeError("No model loaded. Load a model first.")
                self.load_model(self.selected_model_path)

            if not self.current_image_path or self.current_metadata is None:
                raise RuntimeError("No image selected. Parse an image first.")

            sd_prompt = self.current_metadata.get("prompt", "")
            response_parts: List[str] = []

            for chunk in self.current_vlm.generate_wan_prompt_stream(
                image_path=self.current_image_path,
                sd_prompt=sd_prompt,
                additional_instruction=additional_instruction or "",
                style_preset=style_preset,
                output_language=output_language or "English",
                output_sections=sections,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            ):
                response_parts.append(chunk)

            output_text = "".join(response_parts)
            status = self._model_state()

            if auto_unload:
                self.unload_model()
                status = self._model_state()

            return {
                "output_text": output_text,
                "image_path": self.current_image_path,
                "model": status,
            }

    def generate_prompt_stream(
        self,
        additional_instruction: str,
        style_preset: Optional[str],
        output_language: str,
        output_sections: Optional[List[str]],
        temperature: float,
        max_tokens: int,
        auto_unload: bool,
    ) -> Generator[str, None, None]:
        sections = output_sections or ["scene", "action", "camera", "style", "prompt"]

        with self.lock:
            if self.current_vlm is None:
                if not self.selected_model_path:
                    raise RuntimeError("No model loaded. Load a model first.")
                self.load_model(self.selected_model_path)

            if not self.current_image_path or self.current_metadata is None:
                raise RuntimeError("No image selected. Parse an image first.")

            sd_prompt = self.current_metadata.get("prompt", "")

            for chunk in self.current_vlm.generate_wan_prompt_stream(
                image_path=self.current_image_path,
                sd_prompt=sd_prompt,
                additional_instruction=additional_instruction or "",
                style_preset=style_preset,
                output_language=output_language or "English",
                output_sections=sections,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            ):
                yield chunk

            if auto_unload:
                self.unload_model()

    def get_settings(self) -> Dict[str, Any]:
        cache = self._read_cache()
        defaults = {
            "inference_settings": {
                "temperature": self.config.get("inference", {}).get("temperature", 0.7),
                "max_tokens": self.config.get("inference", {}).get("max_tokens", 1024),
                "top_p": self.config.get("inference", {}).get("top_p", 0.9),
            },
            "generation_settings": {
                "language": "English",
                "style_preset": None,
            },
            "output_sections": ["scene", "action", "camera", "style", "prompt"],
            "auto_unload": False,
            "save_directory": "",
            "device_settings": self._load_device_settings(),
            "last_model": None,
        }
        merged = {**defaults, **cache}
        return merged

    def save_settings(self, values: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(values, dict):
            raise ValueError("values must be an object")
        cache = self._read_cache()
        cache.update(values)
        self._write_cache(cache)
        return self.get_settings()

    def save_prompt_to_file(
        self, save_dir: str, output_text: str, additional_instruction: str
    ) -> Dict[str, Any]:
        if not self.current_image_path:
            raise RuntimeError("No image selected. Parse an image first.")
        if not output_text or not output_text.strip():
            raise ValueError("output_text is empty")

        # Always save next to the source image.
        target_dir = Path(self.current_image_path).parent

        image_filename = Path(self.current_image_path).stem + ".txt"
        save_path = target_dir / image_filename

        lines: List[str] = []
        original_prompt = (self.current_metadata or {}).get("prompt", "")
        if original_prompt:
            lines.append("=== Original Prompt ===")
            lines.append(original_prompt)
            lines.append("")

        if additional_instruction and additional_instruction.strip():
            lines.append("=== Additional Instruction ===")
            lines.append(additional_instruction.strip())
            lines.append("")

        lines.append("=== Generated WAN Prompt ===")
        lines.append(output_text.strip())
        lines.append("")

        save_path.write_text("\n".join(lines), encoding="utf-8")
        return {"saved_path": str(save_path)}

    def save_session_json(self, prompt_text: str, additional_instruction: str) -> Dict[str, Any]:
        if not self.current_image_path:
            raise RuntimeError("No image selected. Parse an image first.")

        image_path = Path(self.current_image_path)
        save_path = image_path.with_suffix(".json")
        payload = {
            "image_filename": image_path.name,
            "image_path": str(image_path),
            "metadata": self.current_metadata or {},
            "prompt": (prompt_text or "").strip(),
            "additional_instruction": (additional_instruction or "").strip(),
        }
        save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"saved_path": str(save_path), "data": payload}

    def load_session_json(self, json_path: str) -> Dict[str, Any]:
        normalized = self._normalize_image_path(json_path)
        path = Path(normalized)
        if not path.exists():
            raise FileNotFoundError(f"JSON not found: {normalized}")

        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Invalid JSON format: object expected")

        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        image_path = (data.get("image_path") or "").strip()
        image_filename = (data.get("image_filename") or "").strip()
        resolved_image_path = ""

        # Prefer an image next to the JSON file first (portable when files are moved together).
        if image_filename:
            candidate = path.parent / image_filename
            if candidate.exists():
                resolved_image_path = str(candidate.resolve())

        # Fallback to stored absolute/relative path from JSON.
        if not resolved_image_path and image_path:
            candidate = Path(self._normalize_image_path(image_path))
            if candidate.exists():
                resolved_image_path = str(candidate.resolve())

        image_url = ""
        if resolved_image_path:
            image_url = Path(resolved_image_path).as_uri()

        with self.lock:
            self.current_metadata = metadata
            self.current_image_path = resolved_image_path or None

        return {
            "image_filename": image_filename or (Path(resolved_image_path).name if resolved_image_path else ""),
            "image_path": resolved_image_path,
            "image_url": image_url,
            "metadata": metadata,
            "prompt": (data.get("prompt") or "").strip(),
            "additional_instruction": (data.get("additional_instruction") or "").strip(),
            "json_path": str(path),
        }

    def _model_state(self) -> Dict[str, Any]:
        loaded = self.current_vlm is not None
        context_length = self.current_vlm.get_context_length() if loaded else 0
        selected_name = Path(self.selected_model_path).name if self.selected_model_path else None
        return {
            "loaded": loaded,
            "selected_model_path": self.selected_model_path,
            "selected_model_name": selected_name,
            "context_length": context_length,
        }

    def get_model_state(self) -> Dict[str, Any]:
        return self._model_state()

    def _load_device_settings(self) -> Dict[str, str]:
        defaults = {
            "device": self.config.get("model", {}).get("device", "auto"),
            "dtype": self.config.get("model", {}).get("dtype", "bfloat16"),
        }
        try:
            data = self._read_cache()
            return data.get("device_settings", defaults)
        except Exception:
            return defaults
        return defaults

    def _save_last_model_path(self, model_path: str) -> None:
        try:
            data = self._read_cache()
            data["last_model"] = model_path
            self._write_cache(data)
        except Exception:
            # Cache write failures should not break generation.
            pass

    def _read_cache(self) -> Dict[str, Any]:
        if not self.settings_cache_file.exists():
            return {}
        try:
            return json.loads(self.settings_cache_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_cache(self, data: Dict[str, Any]) -> None:
        self.settings_cache_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _normalize_image_path(self, image_path: str) -> str:
        value = (image_path or "").strip().strip('"').strip("'")
        if not value:
            return value

        # Accept file URI input from desktop drag-and-drop.
        if value.lower().startswith("file://"):
            parsed = urlparse(value)
            uri_path = unquote(parsed.path or "")
            # Windows drive letter URI: /C:/path/to/file.png
            if len(uri_path) >= 3 and uri_path[0] == "/" and uri_path[2] == ":":
                uri_path = uri_path[1:]
            uri_path = uri_path.replace("/", "\\")
            return uri_path

        return value
