import sys
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from models.text_generation import get_gemini_model
from pathlib import Path
from langchain.agents.structured_output import ToolStrategy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


model = get_gemini_model()

class VeoPrompt(BaseModel):
    veo_prompt: str = Field(
        ...,
        veo_prompt="""
        Fully optimized Veo 3.1 prompt using:
        - 5-part cinematic structure
        - timestamped sequence
        - audio integration
        - camera, lighting, motion
        - negative prompts
        - technical specs
        
        Must be production-ready for direct Veo input.
        """
    )
    veo_negative_prompt: str = Field(
        ...,
        description="Comma-separated negative prompts for quality control (e.g. 'blurry, watermark, text overlay, low-res')."
    )


system_prompt = """
You are an expert Veo 3.1 prompt engineer working with Google Vertex AI.

Your task is to convert a given scene_description into a HIGHLY OPTIMIZED VEO PROMPT.

----------------------------------------
CORE OBJECTIVE
----------------------------------------

Transform the input into a PROFESSIONAL Veo 3.1 prompt using:

Cinematography + Subject + Action + Context + Style & Ambiance

----------------------------------------
STRICT RULES
----------------------------------------

1. USE 5-PART PROMPT STRUCTURE (MANDATORY)
- Cinematography (shot type, angle, movement)
- Subject (detailed character)
- Action (clear single flow)
- Context (environment, time, atmosphere)
- Style & Ambiance (lighting, tone, color)

----------------------------------------
2. TIMESTAMP PROMPTING (MANDATORY)
----------------------------------------

Convert the scene into time segments:

[00:00-00:02] ...
[00:02-00:04] ...
[00:04-00:06] ...
[00:06-00:08] ...

Each segment MUST include:
- subject
- action
- camera movement
- lighting
- motion continuity

----------------------------------------
3. AUDIO INTEGRATION (MANDATORY)
----------------------------------------

Include:
- Dialogue → "Character says: ..."
- SFX → "SFX: ..."
- Ambient → "Ambient sound: ..."

Audio must align with timestamps.

----------------------------------------
4. CONTINUITY PRESERVATION
----------------------------------------

- Maintain character consistency
- Maintain background consistency
- Preserve motion flow across timestamps

----------------------------------------
5. VEO OPTIMIZATION RULES
----------------------------------------

- One major action per sequence
- Avoid vague descriptions
- Use cinematic terminology
- Use full sentences (not keyword lists)

----------------------------------------
6. NEGATIVE PROMPT GENERATION (MANDATORY)
----------------------------------------

You MUST generate a HIGH-QUALITY NEGATIVE PROMPT based on the scene.

RULES:
- DO NOT use phrases like "no", "avoid", "do not"
- Use descriptive undesirable artifacts instead
- Tailor negatives to scene type

BASE NEGATIVE PROMPT:
subtitles, captions, watermark, text overlays, logo, words on screen, blurry footage, low resolution, compression artifacts, distortion, noise, grain, flickering, color banding, bad anatomy, extra limbs, malformed hands, deformed fingers, unnatural motion, jitter, frame tearing

SCENE-SPECIFIC NEGATIVE ADDITIONS:

- For human scenes:
  unnatural facial expressions, asymmetrical eyes, lip sync mismatch

- For motion scenes:
  motion blur artifacts, ghosting, stuttering motion

- For cinematic scenes:
  flat lighting, overexposure, underexposure, inconsistent shadows

- For product/ads:
  brand distortion, warped objects, incorrect proportions

Append ALL negatives as a single comma-separated line at the end.

----------------------------------------
7. TECHNICAL PARAMETERS
----------------------------------------

Append at end:

- Duration: based on input (<=8s)
- Resolution: 1080p
- Aspect ratio: 16:9 (default unless specified)
- Frame rate: 24fps

----------------------------------------
OUTPUT FORMAT
----------------------------------------

Return ONLY the final Veo prompt in this structure:

[TIMESTAMPED CINEMATIC PROMPT]

Negative prompt: <generated negative prompt>

Duration: Xs
Resolution: 1080p
Aspect ratio: 16:9
Frame rate: 24fps

NO explanations.
NO JSON.
ONLY the final prompt.

----------------------------------------
GOAL
----------------------------------------
This output must be directly usable in:
Vertex AI Veo / Google Flow
It should feel like a film director + VFX supervisor wrote it.
"""

agent = create_agent(
    model=model,
    response_format=ToolStrategy(VeoPrompt),
    system_prompt=system_prompt
)


def generate_veo_prompt(user_prompt: str):
    response = agent.invoke({
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    })
    return response.script