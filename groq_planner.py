"""
Turns a natural-language command like "pick up the red block and put it in
the left bin" into a structured task:

    {"action": "pick_place", "object": "red block", "destination": "left bin"}

Uses GroqCloud (OpenAI-compatible): https://api.groq.com/openai/v1/chat/completions

Set your key before running. In Colab:

    import os
    os.environ["GROQ_API_KEY"] = "gsk_..."

or better, use Colab's secrets panel:

    from google.colab import userdata
    os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

Model defaults to openai/gpt-oss-20b. Override with GROQ_MODEL if you want a
list_models() below to see what your key can actually reach right now.

If no key is set or the request fails, this falls back to a dumb keyword
parser so a flaky connection doesn't kill the demo. Set FALLBACK_ENABLED to
False if you'd rather it raise.
"""

import os
import json
import requests

from arm_config import OBJECTS, DESTINATIONS

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
FALLBACK_ENABLED = True


SYSTEM_PROMPT = f"""You control a robot arm. Given a typed command, output ONLY a JSON
object (no prose, no markdown fences) with this exact shape:

{{"action": "pick_place", "object": "<one of: {', '.join(OBJECTS.keys())}>", "destination": "<one of: {', '.join(DESTINATIONS.keys())}>"}}

If the command doesn't clearly name one of the known objects or destinations,
pick the closest reasonable match. If it is not a pick-and-place command at
all, output {{"action": "unknown"}}. Respond with JSON only.
"""


def _api_key():
    # GROQ_API_KEY is the name Groq uses. XAI_API_KEY is accepted too so an
    # older notebook cell doesn't silently fall back to keyword matching.
    return os.environ.get("GROQ_API_KEY") or os.environ.get("XAI_API_KEY")


def list_models(timeout=15):
    """Prints the model ids your key can use. Run this if you get a 404."""
    key = _api_key()
    if not key:
        print("[groq_planner] No GROQ_API_KEY set.")
        return []
    resp = requests.get(GROQ_MODELS_URL,
                        headers={"Authorization": f"Bearer {key}"},
                        timeout=timeout)
    resp.raise_for_status()
    ids = sorted(m["id"] for m in resp.json().get("data", []))
    for i in ids:
        print(i)
    return ids


def _fallback_parse(command_text):
    """Very dumb keyword match, used only if the API call fails."""
    text = command_text.lower()
    obj = next((o for o in OBJECTS if o.split()[0] in text), None)
    dest = next((d for d in DESTINATIONS if d.split()[0] in text), None)
    if obj and dest:
        return {"action": "pick_place", "object": obj, "destination": dest}
    return {"action": "unknown"}


def _post(payload, key, timeout):
    resp = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload, timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_command(command_text, timeout=15):
    """Calls Groq to convert command_text into a structured task dict."""
    key = _api_key()
    model = os.environ.get("GROQ_MODEL", GROQ_MODEL)

    if not key:
        if FALLBACK_ENABLED:
            print("[groq_planner] No GROQ_API_KEY set, using fallback parser.")
            return _fallback_parse(command_text)
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": command_text},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }

    try:
        try:
            content = _post(payload, key, timeout)
        except requests.HTTPError as http_err:
            # Not every Groq model supports response_format. Retry plain.
            if http_err.response is not None and http_err.response.status_code == 400:
                payload.pop("response_format", None)
                content = _post(payload, key, timeout)
            else:
                raise
        task = json.loads(content.strip().strip("`").removeprefix("json").strip())
    except Exception as e:
        detail = ""
        if isinstance(e, requests.HTTPError) and e.response is not None:
            detail = f" -- {e.response.status_code}: {e.response.text[:200]}"
        print(f"[groq_planner] Groq API call failed ({e}){detail}")
        print(f"[groq_planner] Model was '{model}'. Run groq_planner.list_models() "
              f"to see valid ids, then set os.environ['GROQ_MODEL'].")
        if FALLBACK_ENABLED:
            print("[groq_planner] Falling back to keyword parser.")
            return _fallback_parse(command_text)
        raise

    if task.get("action") == "pick_place":
        if task.get("object") not in OBJECTS or task.get("destination") not in DESTINATIONS:
            print(f"[groq_planner] Groq returned unknown names: {task}, falling back.")
            return _fallback_parse(command_text)

    return task


if __name__ == "__main__":
    import sys
    cmd = " ".join(sys.argv[1:]) or "pick up the red block and put it in the left bin"
    print(parse_command(cmd))