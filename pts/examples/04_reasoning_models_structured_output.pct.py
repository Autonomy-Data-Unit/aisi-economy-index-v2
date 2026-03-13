# ---
# jupyter:
#   kernelspec:
#     display_name: ai-index (3.12.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reasoning Models & Structured Output
#
# Reasoning models (DeepSeek-R1, Qwen3/3.5, GPT-OSS) emit a thinking/analysis
# chain before the actual answer. This can break structured JSON output because
# the raw response contains both the reasoning prefix and the JSON payload.
#
# This notebook:
# 1. Tests each suspected reasoning model with the same prompt and JSON schema
#    used by `llm_summarise`
# 2. Develops a robust `extract_json` function that reliably extracts the JSON
#    from reasoning model output

# %%
import asyncio
import json
import re

from pydantic import BaseModel
from ai_index.utils import allm_generate

# %% [markdown]
# ## Schema and prompt (same as llm_summarise)

# %%
class JobInfoModel(BaseModel):
    short_description: str
    tasks: list[str]
    skills: list[str]
    domain: str
    level: str

json_schema = JobInfoModel.model_json_schema()

SYSTEM_PROMPT = """You are a precise data extraction system for job advertisements. Extract structured information from each job ad exactly as specified.

Rules:
- short_description: A single sentence summarising the role (max 30 words).
- tasks: The most important duties. List at most 5. Each task should be a concise phrase (max 10 words).
- skills: The most important required skills or qualifications. List at most 5. Each skill should be a concise phrase (max 10 words).
- domain: The industry or professional domain (e.g. "Healthcare", "Software Engineering", "Retail", "Finance").
- level: "Entry-Level" if the role requires fewer than 3 years of experience or is described as junior/entry/graduate. Otherwise "Experienced"."""

USER_PROMPT = """Extract the short_description, tasks, skills, domain, and level from this job ad:

Senior Software Engineer
IT Jobs

We are looking for an experienced senior software engineer to join our team. You will be responsible for designing and implementing scalable backend systems using Python and AWS. The ideal candidate has strong experience with microservices architecture, CI/CD pipelines, and cloud-native development. Requirements: 5+ years experience, strong Python skills, AWS certification preferred. You will mentor junior developers and participate in architecture decisions."""

# %% [markdown]
# ## `extract_json`: robust JSON extraction from reasoning model output
#
# Reasoning models emit a thinking prefix, then the JSON payload as the last
# thing in the response. So the JSON always runs from some `{` to the end of
# the string.
#
# Approach: strip whitespace, check the string ends with `}`, then try
# `text[i:]` for each `{` position until `json.loads` succeeds. No need to
# explicitly strip `<think>...</think>` tags: any `{` inside them will have
# trailing `</think>...` text that makes `json.loads` fail, so the algorithm
# naturally skips past them.

# %%
def extract_json(text: str, model_cls: type[BaseModel] | None = None) -> dict | None:
    """Extract a JSON object from text that may contain a reasoning prefix.

    The JSON is assumed to be the last thing in the response, running from
    some '{' to the end of the string. We try each '{' position from left
    to right until we find one where text[i:] parses as valid JSON.

    Args:
        text: Raw model output, potentially with <think> tags or other prefixes.
        model_cls: Optional Pydantic model class for schema validation. If provided,
            only returns JSON that validates against this model.

    Returns:
        Parsed dict, or None if no valid JSON found.
    """
    text = text.strip()
    if not text.endswith("}"):
        return None

    pos = 0
    while True:
        idx = text.find("{", pos)
        if idx == -1:
            return None
        try:
            parsed = json.loads(text[idx:])
            if model_cls is None or _validates(parsed, model_cls):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        pos = idx + 1


def _validates(data: dict, model_cls: type[BaseModel]) -> bool:
    """Check if data validates against a Pydantic model."""
    try:
        model_cls.model_validate(data)
        return True
    except Exception:
        return False

# %% [markdown]
# ## Test `extract_json` on synthetic examples

# %%
# Clean JSON (no prefix)
assert extract_json('{"a": 1}') == {"a": 1}

# <think> tags with braces inside thinking
assert extract_json(
    '<think>I should return {"a": 1} but let me think...</think>{"b": 2}'
) == {"b": 2}

# Reasoning prefix without tags (like gpt-oss)
assert extract_json(
    'analysis We need to create a response. Here it is: {"b": 2}'
) == {"b": 2}

# Braces in reasoning text: first '{' fails json.loads, skips to the real JSON
assert extract_json(
    "Let me think about {this problem} carefully. "
    '{"short_description": "test", "tasks": [], "skills": [], '
    '"domain": "IT", "level": "Experienced"}'
) == {
    "short_description": "test",
    "tasks": [],
    "skills": [],
    "domain": "IT",
    "level": "Experienced",
}

# With Pydantic validation: first valid JSON doesn't match schema, second does
assert extract_json(
    '<think>{"not_valid": true}</think>'
    '{"also": "wrong"}'
    ' oh wait: {"short_description": "test", '
    '"tasks": ["a"], "skills": ["b"], "domain": "IT", "level": "Experienced"}',
    model_cls=JobInfoModel,
) == {
    "short_description": "test",
    "tasks": ["a"],
    "skills": ["b"],
    "domain": "IT",
    "level": "Experienced",
}

print("All extract_json tests passed")

# %% [markdown]
# ## Models to test
#
# These are the models suspected of having reasoning/thinking behavior.

# %%
REASONING_MODELS = [
    "gpt-oss-120b-sbatch",
    "deepseek-r1-qwen-32b-sbatch",
    "qwen3.5-122b-sbatch",
    "qwen3-235b-sbatch",
]

# %% [markdown]
# ## Run each model and inspect output

# %%
#|eval: false
async def _call_model(model_key):
    """Call a single model and return (model_key, raw_response) or (model_key, error)."""
    try:
        responses = await allm_generate(
            [USER_PROMPT],
            model=model_key,
            system_message=SYSTEM_PROMPT,
            max_new_tokens=2048,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            json_schema=json_schema,
            cache=False,
            time="00:30:00",
        )
        return model_key, responses[0]
    except Exception as e:
        return model_key, e

# Launch all models in parallel
raw_results = await asyncio.gather(*[_call_model(m) for m in REASONING_MODELS])

# %%
#|eval: false
results = {}

for model_key, raw_or_err in raw_results:
    print(f"\n{'=' * 80}")
    print(f"Model: {model_key}")
    print(f"{'=' * 80}")

    if isinstance(raw_or_err, Exception):
        print(f"ERROR: {raw_or_err}")
        results[model_key] = {"error": str(raw_or_err)}
        continue

    raw = raw_or_err
    results[model_key] = {"raw": raw}

    # Show raw output
    print(f"\nRaw length: {len(raw)} chars")
    print(f"\n--- RAW (first 500 chars) ---")
    print(raw[:500])
    print(f"--- END ---")

    # Try direct parse
    try:
        parsed = JobInfoModel.model_validate_json(raw)
        print(f"\nDirect JSON parse: OK")
        print(f"  domain={parsed.domain}, level={parsed.level}")
        results[model_key]["direct_parse"] = True
        results[model_key]["has_reasoning_prefix"] = False
        continue
    except Exception as e:
        print(f"\nDirect JSON parse: FAIL ({e.__class__.__name__})")
        results[model_key]["direct_parse"] = False

    # Try extract_json (robust extraction)
    extracted = extract_json(raw, model_cls=JobInfoModel)
    if extracted is not None:
        parsed = JobInfoModel.model_validate(extracted)
        print(f"extract_json: OK")
        print(f"  domain={parsed.domain}, level={parsed.level}")
        results[model_key]["extract_json"] = True
        results[model_key]["has_reasoning_prefix"] = True

        # Characterise the prefix
        has_think_tags = "<think>" in raw
        results[model_key]["has_think_tags"] = has_think_tags
        if has_think_tags:
            think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
            if think_match:
                print(f"  <think> block: {len(think_match.group(1))} chars")
                print(f"  <think> preview: {think_match.group(1)[:150]!r}...")
        else:
            # Find where the JSON starts
            json_str = json.dumps(extracted)
            # Look for the first { that starts the extracted JSON
            first_brace = raw.find("{")
            print(f"  No <think> tags, raw prefix: {first_brace} chars")
            print(f"  Prefix preview: {raw[:min(first_brace, 200)]!r}...")
    else:
        print(f"extract_json: FAIL (no valid JSON found)")
        results[model_key]["extract_json"] = False
        results[model_key]["has_reasoning_prefix"] = None

# %% [markdown]
# ## Summary

# %%
#|eval: false
print(f"\n{'=' * 80}")
print("SUMMARY")
print(f"{'=' * 80}")
print(f"{'Model':<35} {'Direct':>8} {'Extracted':>10} {'Prefix':>10} {'<think>':>10}")
print("-" * 78)

for model_key in REASONING_MODELS:
    r = results.get(model_key, {})
    if "error" in r:
        print(f"{model_key:<35} {'ERROR':>8}")
        continue

    direct = "OK" if r.get("direct_parse") else "FAIL"
    extracted = "OK" if r.get("extract_json") else ("FAIL" if "extract_json" in r else "n/a")
    has_prefix = "yes" if r.get("has_reasoning_prefix") else ("no" if r.get("has_reasoning_prefix") is False else "?")
    has_think = "yes" if r.get("has_think_tags") else "no"

    print(f"{model_key:<35} {direct:>8} {extracted:>10} {has_prefix:>10} {has_think:>10}")
