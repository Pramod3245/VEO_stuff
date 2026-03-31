from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from agents.script_generation_agent import generate_script

router = APIRouter()


class FrameRef(BaseModel):
    type: str
    url: str


class ScriptGenerateRequest(BaseModel):
    model_id: Optional[str] = None
    feature: Optional[str] = None
    prompt: str
    target_audience_id: Optional[int] = None
    theme_id: Optional[int] = None
    target_platform_id: Optional[int] = None
    script_generated: Optional[Any] = None
    scenes: Optional[Any] = None
    no_of_scenes: int
    max_time_per_scene: int
    total_duration: Optional[float] = None
    resolution: Optional[str] = None
    aspect_ratio: Optional[str] = None
    start_frame: Optional[FrameRef] = None
    end_frame: Optional[FrameRef] = None
    media: Optional[List[FrameRef]] = None
    repo: Optional[List[Dict[str, Any]]] = None


@router.post("/generate-script")
def generate_script_endpoint(request: ScriptGenerateRequest):
    scenes_payload = generate_script(
        prompt=request.prompt,
        no_of_scenes=request.no_of_scenes,
        max_time_per_scene=request.max_time_per_scene,
        repo=request.repo,
    )
    return {"scenes": scenes_payload}
