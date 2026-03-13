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
# Reasoning models (DeepSeek-R1, Qwen3/3.5, GPT-OSS, Nemotron-Ultra) emit a
# thinking/analysis chain before the actual answer. This can break structured
# JSON output because the raw response contains both the reasoning prefix and
# the JSON payload.
#
# This notebook tests each suspected reasoning model with the same prompt and
# JSON schema used by `llm_summarise`, to see what the raw output looks like
# and whether we can reliably extract the JSON.

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
# ## Models to test
#
# These are the models suspected of having reasoning/thinking behavior.

# %%
REASONING_MODELS = [
    "gpt-oss-120b-sbatch",
    "deepseek-r1-qwen-32b-sbatch",
    "qwen3.5-122b-sbatch",
    "qwen3-235b-sbatch",
    "nemotron-ultra-253b-sbatch",
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
        print(f"\nDirect parse: SUCCESS")
        print(f"  domain={parsed.domain}, level={parsed.level}")
        results[model_key]["direct_parse"] = True
        continue
    except Exception as e:
        print(f"\nDirect parse: FAILED ({e.__class__.__name__})")
        results[model_key]["direct_parse"] = False

    # Try extracting JSON from first '{'
    json_start = raw.find("{")
    if json_start >= 0:
        json_str = raw[json_start:]
        print(f"\nPrefix before JSON: {json_start} chars")
        print(f"Prefix content: {raw[:json_start][:200]!r}...")

        try:
            parsed = JobInfoModel.model_validate_json(json_str)
            print(f"Prefix-stripped parse: SUCCESS")
            print(f"  domain={parsed.domain}, level={parsed.level}")
            results[model_key]["stripped_parse"] = True
        except Exception as e:
            print(f"Prefix-stripped parse: FAILED ({e.__class__.__name__}: {e})")
            results[model_key]["stripped_parse"] = False

            # Try finding the last complete JSON object
            # Some models may have JSON embedded in the middle
            try:
                for i in range(len(raw)):
                    if raw[i] == "{":
                        try:
                            parsed = JobInfoModel.model_validate_json(raw[i:])
                            print(f"JSON found at offset {i}: SUCCESS")
                            print(f"  domain={parsed.domain}, level={parsed.level}")
                            results[model_key]["offset_parse"] = i
                            break
                        except Exception:
                            continue
                else:
                    print("No valid JSON object found at any offset")
            except Exception:
                pass
    else:
        print("\nNo '{' found in response at all")

# %% [markdown]
# ## Summary

# %%
#|eval: false
print(f"\n{'=' * 80}")
print("SUMMARY")
print(f"{'=' * 80}")
print(f"{'Model':<35} {'Direct':>8} {'Stripped':>10} {'Prefix len':>12}")
print("-" * 70)

for model_key in REASONING_MODELS:
    r = results.get(model_key, {})
    if "error" in r:
        print(f"{model_key:<35} {'ERROR':>8}")
        continue

    direct = "OK" if r.get("direct_parse") else "FAIL"
    stripped = "OK" if r.get("stripped_parse") else ("FAIL" if "stripped_parse" in r else "n/a")
    raw = r.get("raw", "")
    prefix_len = raw.find("{") if "{" in raw else -1
    prefix_str = str(prefix_len) if prefix_len >= 0 else "no JSON"

    print(f"{model_key:<35} {direct:>8} {stripped:>10} {prefix_str:>12}")
