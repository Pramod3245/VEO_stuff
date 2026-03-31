from pydantic import BaseModel, Field
from langchain.agents import create_agent
from models.text_generation import get_gemini_model
from typing import List
from langchain.agents.structured_output import ToolStrategy

model = get_gemini_model()

class Scene(BaseModel):
    scene_number: int
    scene_description: str = Field(
        ...,
        description="""
        A SINGLE continuous block of text describing the entire scene.
        MUST include timestamped breakdown (e.g., 0.0s, 1.5s, etc.).
        Each moment must describe:
        - subject
        - action
        - environment
        - camera movement
        - composition
        - lighting
        - motion
        
        This must also naturally include continuity from previous scene 
        and transition into next scene (NO separate fields).
        """
    )

    characters: str = Field(
        ...,
        description="""
        Highly detailed character prompt.MUST remain consistent across scenes unless explicitly changed.Include appearance, clothing, age, expressions, style.
        """
    )

    background: str = Field(
        ...,
        description="""
        Highly detailed environment prompt.MUST remain consistent across scenes unless explicitly changed.Include location, props, lighting mood, atmosphere.
        """
    )

    first_frame_image: str = Field(
        ...,
        description="Ultra-detailed static image prompt for first frame (no motion)"
    )

    last_frame_image: str = Field(
        ...,
        description="Ultra-detailed static image prompt for last frame (must match next scene start)"
    )

    audio: str = Field(
        ...,
        description="""
        Dialogue, background music, ambience, sound effects.Must be explicitly described and synchronized with visuals.
        """
    )


class ScriptOutput(BaseModel):
    character_prompt: str = Field(
        ...,
        description="""
        MASTER character definition used across ALL scenes.Acts as source of truth for consistency.
        """
    )

    color_schema: str = Field(
        ...,
        description="""
        Global cinematic color palette and grading style.Example: warm tones, teal-orange, high contrast, soft pastel, etc.Must be followed across all scenes.
        """
    )

    scenes: List[Scene]


system_prompt = """
You are an expert cinematic director and Veo prompt engineer working with Google Vertex AI.

Your task is to generate a HIGHLY STRUCTURED VIDEO PLAN optimized for Veo video generation.

----------------------------------------
CRITICAL REQUIREMENTS
----------------------------------------

1. VEO CONSTRAINT:
- Each scene MUST be <= 8 seconds
- Each scene is generated independently BUT must feel like ONE CONTINUOUS VIDEO

----------------------------------------
2. CONTINUITY (VERY IMPORTANT)
----------------------------------------

- Scenes MUST connect seamlessly WITHOUT separate fields
- Continuity must be embedded INSIDE scene_description
- End of previous scene must MATCH start of next scene

Use:
- motion continuity (walking continues)
- camera continuity (same pan continues)
- lighting continuity
- object continuity

----------------------------------------
3. FRAME-BY-FRAME TIMESTAMPED DESCRIPTION
----------------------------------------

Each scene_description MUST:
- Be a SINGLE paragraph (no lists)
- Include timestamps (e.g., 0.0s, 1.2s, 3.5s...)
- Describe EVERY moment precisely

Each timestamp MUST include:
- subject
- action
- environment
- camera (angle, lens, movement)
- composition
- lighting
- motion

Think like directing a film frame-by-frame.

----------------------------------------
4. CHARACTER CONSISTENCY (GLOBAL + LOCAL)
----------------------------------------

- Define a MASTER character_prompt (global)
- Scene.characters MUST reuse it exactly unless change is required
- If changed → explicitly describe the difference

----------------------------------------
5. BACKGROUND CONSISTENCY
----------------------------------------

- Scene.background must remain consistent unless changed
- If same location → reuse description precisely
- Maintain lighting + atmosphere continuity

----------------------------------------
6. IMAGE GENERATION SUPPORT
----------------------------------------

For EACH scene:
- first_frame_image → highly detailed static shot
- last_frame_image → must MATCH next scene start

----------------------------------------
7. AUDIO (SEPARATE FIELD)
----------------------------------------

Include:
- dialogue
- ambient sound
- music
- SFX

Must align with timestamps implicitly

----------------------------------------
8. VEO PROMPT STRUCTURE (MANDATORY)
----------------------------------------

Follow cinematic structure:
Subject + Action + Environment + Style + Camera + Lighting + Motion + Audio

----------------------------------------
9. COLOR SCHEMA
----------------------------------------

Define a global color grading style.
Apply consistently across scenes.

----------------------------------------
OUTPUT RULES
----------------------------------------

- No vague descriptions
- No assumptions
- No missing transitions
- Extremely detailed and production-ready
"""

import json
from langchain_core.messages import HumanMessage, SystemMessage

agent = create_agent(
    model=model,
    response_format=ToolStrategy(ScriptOutput),
    system_prompt=system_prompt
)

structured_model = model.with_structured_output(ScriptOutput)


def generate_script(prompt: str, no_of_scenes: int, max_time_per_scene: int, repo=None) -> list:
    repo_context = (
        f"\n\nAdditional brand/product context:\n{json.dumps(repo, indent=2)}"
        if repo else ""
    )
    user_message = (
        f"{prompt}\n\n"
        f"Requirements:\n"
        f"- Generate exactly {no_of_scenes} scenes.\n"
        f"- Each scene must be a maximum of {max_time_per_scene} seconds."
        f"{repo_context}"
    )
    script: ScriptOutput = structured_model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])
    return [
        {
            "scene": f"scene{scene.scene_number}",
            "description": scene.scene_description,
            "duration_sec": max_time_per_scene,
            "thumbnail": None,
        }
        for scene in script.scenes
    ]


if __name__ == "__main__":
    user_prompt = "Create a short, energetic advertisement where a football player inspired by FIFA celebrates victory on the field, then highlights staying fresh and confident using a premium soap—blend sports action, sweat, freshness, and confidence into a catchy, 10–15 second ad."
    response = agent.invoke({
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    })
    print(response)