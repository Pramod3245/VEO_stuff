import sys
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from models.text_generation import get_gemini_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

model = get_gemini_model()


class Scene(BaseModel):
    scene: str
    description: str
    duration_sec: int
    thumbnail: str | None = None


class ScriptOutput(BaseModel):
    scenes: List[Scene]


system_prompt = """
You are an expert cinematic editor and Veo prompt engineer.

Your task is to MODIFY an existing multi-scene script based on user instructions.

----------------------------------------
INPUT
----------------------------------------

You will receive:
1. Previous script (JSON with scenes)
2. User instruction describing changes

----------------------------------------
CORE RESPONSIBILITIES
----------------------------------------

1. UNDERSTAND USER INTENT
- Identify which scene(s) need modification
- Identify what needs to change:
  - character
  - environment
  - lighting
  - camera
  - tone/mood
  - action

----------------------------------------
2. MODIFY ONLY NECESSARY PARTS
----------------------------------------

- Only update relevant scenes
- Keep all other scenes EXACTLY SAME
- Do NOT rewrite everything

----------------------------------------
3. PRESERVE STRUCTURE
----------------------------------------

- Keep same number of scenes
- Keep scene names (scene1, scene2…)
- Keep duration_sec unchanged
- Keep timestamps unless change required

----------------------------------------
4. CONTINUITY (VERY IMPORTANT)
----------------------------------------

- Ensure smooth flow between scenes
- Maintain character consistency
- Maintain environment consistency unless changed

----------------------------------------
5. DESCRIPTION RULES
----------------------------------------

Each scene description must:
- Remain timestamped (0.0s, 1.2s, etc.)
- Be highly detailed
- Maintain cinematic quality

----------------------------------------
6. OUTPUT FORMAT
----------------------------------------

Return FULL UPDATED JSON:

{
  "scenes": [...]
}

NO explanation
NO extra text

----------------------------------------
GOAL
----------------------------------------

Produce an updated script ready for:
Gemini → Imagen → Veo pipeline
"""


agent = create_agent(
    model=model,
    response_format=ToolStrategy(ScriptOutput),
    system_prompt=system_prompt
)



def edit_full_script(user_prompt: str, previous_script: dict):
    """
    user_prompt: string describing changes
    previous_script: JSON dict of scenes
    """

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
USER REQUEST:
{user_prompt}

PREVIOUS SCRIPT:
{previous_script}
"""
            }
        ]
    })

    return response
