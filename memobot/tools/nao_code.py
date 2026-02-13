"""
Generate Python code for NAO V4 robot (naoqi, ALProxy).
Code is intended to be run on the robot as in vibe_test_client: header is
injected by the client; this module returns only the body (no shebang/imports).
"""

import os

# Load env from repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DOTENV_PATH = os.path.join(_REPO_ROOT, ".env")
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_DOTENV_PATH, override=True)
except ImportError:
    pass

NAO_SYSTEM_INSTRUCTION = """You are a NAO V4 robot code generator. You output only runnable Python 2.7 code.

Environment on the robot:
- Python 2.7 with naoqi (ALProxy, ALBroker).
- Available by default in execution context: time, ALProxy.
- Robot IP is typically passed or fixed (e.g. 127.0.0.1 when running on robot).
- Common proxies: ALTextToSpeech, ALMotion, ALAnimationPlayer, ALBehaviorManager, ALMemory, ALVideoDevice.

Rules:
- Output ONLY raw Python code. No markdown, no ``` blocks, no explanations.
- Use ALProxy("ModuleName", robot_ip, 9559) to get a proxy. Prefer robot_ip variable or 127.0.0.1.
- Keep code short and safe (no infinite loops unless asked).
- Use time.sleep() for delays. Use tts = ALProxy("ALTextToSpeech", robot_ip, 9559) and tts.say("...") for speech.
"""


def generate_nao_code(coding_prompt: str, api_key: str = None) -> str:
    """
    Generate NAO V4 (naoqi) Python code from a natural language coding prompt.
    Returns only the code body (no shebang/imports); the robot client adds those.
    """
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required to generate NAO code")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("google-genai is required. Install with: pip install google-genai")

    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash"

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(parts=[
                types.Part(text=NAO_SYSTEM_INSTRUCTION),
                types.Part(text="User request:\n" + coding_prompt),
            ])
        ],
    )
    code = (response.text or "").strip()

    # Strip markdown code blocks if present
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()

    return code
