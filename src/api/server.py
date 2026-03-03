"""FastAPI server for Electron/Tauri integration."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.service import PromptService


class ImageParseRequest(BaseModel):
    image_path: str = Field(..., description="Absolute or relative path to image file")


class ModelLoadRequest(BaseModel):
    model_path: str


class ModelDownloadRequest(BaseModel):
    repo_id: str
    local_name: Optional[str] = None


class GenerateRequest(BaseModel):
    additional_instruction: str = ""
    style_preset: Optional[str] = "cinematic"
    output_language: str = "English"
    output_sections: List[str] = Field(default_factory=lambda: ["scene", "action", "camera", "style", "prompt"])
    temperature: float = 0.7
    max_tokens: int = 1024
    auto_unload: bool = False


class SettingsUpdateRequest(BaseModel):
    values: Dict[str, Any]


class SavePromptRequest(BaseModel):
    save_dir: str
    output_text: str
    additional_instruction: str = ""


class SaveSessionRequest(BaseModel):
    prompt: str = ""
    additional_instruction: str = ""


class LoadSessionRequest(BaseModel):
    json_path: str


def create_app(config: Dict[str, Any]) -> FastAPI:
    service = PromptService(config)
    app = FastAPI(title="WAN Prompt API", version=config["app"]["version"])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "app": config["app"]["name"],
            "version": config["app"]["version"],
        }

    @app.get("/models")
    def list_models() -> Dict[str, Any]:
        return {
            "models": service.list_models(),
        }

    @app.post("/models/load")
    def load_model(req: ModelLoadRequest) -> Dict[str, Any]:
        try:
            return service.load_model(req.model_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/models/download")
    def download_model(req: ModelDownloadRequest) -> Dict[str, Any]:
        try:
            result = service.download_model(req.repo_id, req.local_name)
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/models/unload")
    def unload_model() -> Dict[str, Any]:
        try:
            return service.unload_model()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/image/parse")
    def parse_image(req: ImageParseRequest) -> Dict[str, Any]:
        try:
            metadata = service.parse_image(req.image_path)
            return {"metadata": metadata}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/image/clear")
    def clear_image() -> Dict[str, str]:
        service.clear_image()
        return {"status": "cleared"}

    @app.post("/generate")
    def generate(req: GenerateRequest) -> Dict[str, Any]:
        try:
            return service.generate_prompt(
                additional_instruction=req.additional_instruction,
                style_preset=req.style_preset,
                output_language=req.output_language,
                output_sections=req.output_sections,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                auto_unload=req.auto_unload,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/generate/stream")
    def generate_stream(req: GenerateRequest) -> StreamingResponse:
        def ndjson_stream():
            try:
                for chunk in service.generate_prompt_stream(
                    additional_instruction=req.additional_instruction,
                    style_preset=req.style_preset,
                    output_language=req.output_language,
                    output_sections=req.output_sections,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    auto_unload=req.auto_unload,
                ):
                    yield json.dumps({"type": "chunk", "content": chunk}, ensure_ascii=False) + "\n"

                yield json.dumps(
                    {"type": "done", "model": service.get_model_state()},
                    ensure_ascii=False,
                ) + "\n"
            except Exception as exc:
                yield json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False) + "\n"

        return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")

    @app.get("/settings")
    def get_settings() -> Dict[str, Any]:
        try:
            return service.get_settings()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/settings")
    def save_settings(req: SettingsUpdateRequest) -> Dict[str, Any]:
        try:
            return service.save_settings(req.values)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/prompt/save")
    def save_prompt(req: SavePromptRequest) -> Dict[str, Any]:
        try:
            return service.save_prompt_to_file(
                save_dir=req.save_dir,
                output_text=req.output_text,
                additional_instruction=req.additional_instruction,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/session/save")
    def save_session(req: SaveSessionRequest) -> Dict[str, Any]:
        try:
            return service.save_session_json(
                prompt_text=req.prompt,
                additional_instruction=req.additional_instruction,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/session/load")
    def load_session(req: LoadSessionRequest) -> Dict[str, Any]:
        try:
            return service.load_session_json(req.json_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
