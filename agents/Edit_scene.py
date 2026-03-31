import sys
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from models.text_generation import get_gemini_model
from pathlib import Path
from langchain.agents.structured_output import ToolStrategy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


model = get_gemini_model()

class EditScene(BaseModel):
    scene_number: int = Field(..., description="Scene number being edited")
    scene_description: str = Field(
        ...,
        description="""
        Updated scene_description with FULL timestamped cinematic breakdown.
        Must preserve continuity, character consistency, and incorporate user changes.
        """
    )


system_prompt = """
You are an expert cinematic editor and Veo prompt engineer.

Your task is to MODIFY an existing scene based on user instructions.

INPUT WILL CONTAIN:
- scene_number
- existing scene_description
- user change request

----------------------------------------
RULES
----------------------------------------

1. DO NOT CHANGE scene_number

2. MODIFY ONLY scene_description

3. scene_description MUST:
- Remain a SINGLE continuous paragraph
- Include timestamped breakdown (0.0s → end)
- Maintain:
  - subject
  - action
  - environment
  - camera movement
  - lighting
  - motion

4. CONTINUITY (CRITICAL):
- Preserve connection with previous and next scenes
- Do NOT break motion flow or transitions
- Keep character and background consistent unless user asks change

5. APPLY USER EDIT PRECISELY:
- Add / remove / modify only what user requested
- Do NOT rewrite completely unless necessary

6. NO EXTRA TEXT:
- Output ONLY structured JSON
- No explanations

----------------------------------------
GOAL:
Return an updated scene_description ready for Veo pipeline
"""

agent = create_agent(
    model=model,
    response_format=ToolStrategy(EditScene),
    system_prompt=system_prompt
)


# def edit_scene(user_prompt: str):
#     response = agent.invoke({
#         "messages": [
#             {"role": "user", "content": user_prompt}
#         ]
#     })
#     return response

if __name__ == "__main__":
    user_prompt = """
    John enters the cafe and orders coffee.
    """
    response = agent.invoke({
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        })

    print(response)