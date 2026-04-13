"""Quick debug script: submit the saved prompt and print the full error."""
import json
import urllib.request
import urllib.error
import os

here = os.path.dirname(os.path.abspath(__file__))
prompt_path = os.path.join(here, "debug_prompt.json")

with open(prompt_path, encoding="utf-8") as f:
    prompt = json.load(f)

data = json.dumps({"prompt": prompt}).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8000/prompt",
    data=data,
    headers={"Content-Type": "application/json"},
)

try:
    with urllib.request.urlopen(req) as resp:
        print("SUCCESS:", resp.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8")
    print(f"HTTP {e.code} Error")
    print()
    try:
        parsed = json.loads(body)
        print(json.dumps(parsed, indent=2)[:3000])
    except Exception:
        print(body[:3000])
