# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Structured JSON Output with `llm_runner`
#
# This notebook demonstrates how to constrain LLM output to valid JSON
# matching a schema. All three backends (transformers, vLLM, API) support
# a `json_schema` parameter that guarantees the output is parseable JSON
# conforming to the given schema.
#
# - **transformers**: Uses [outlines](https://github.com/dottxt-ai/outlines)
#   logits processing (`pip install outlines`)
# - **vLLM**: Uses built-in `GuidedDecodingParams`
# - **API**: Uses OpenAI-compatible `response_format` (supported by
#   OpenAI, Gemini via litellm, etc.)

# %%
from ai_index import const

# %% [markdown]
# ## Define a schema
#
# We use a Pydantic model to define the expected output structure, then
# convert it to a JSON schema dict via `model_json_schema()`.

# %%
from pydantic import BaseModel

class JobSummary(BaseModel):
    job_title: str
    skills: list[str]
    years_experience: int | None

schema = JobSummary.model_json_schema()
print(schema)

# %% [markdown]
# ## Local transformers (CPU)
#
# Uses `outlines` to constrain token generation via logits processing.
# The `qwen-0.5b-mac` model key resolves to CPU/float32 so it runs anywhere.

# %%
from ai_index.utils import llm_generate

prompts = [
    "Extract structured info from this job ad: "
    "Senior Python Developer needed with 5+ years experience in Django, "
    "FastAPI, and PostgreSQL. Must know Docker and CI/CD.",
]

responses = llm_generate(
    prompts,
    model="qwen-0.5b-mac",
    max_new_tokens=120,
    system_message="Extract job information as JSON.",
    json_schema=schema,
)

print("Raw response:")
print(responses[0])

# %% [markdown]
# ### Validate with Pydantic

# %%
import json

parsed = JobSummary.model_validate_json(responses[0])
print(f"Title: {parsed.job_title}")
print(f"Skills: {parsed.skills}")
print(f"Years experience: {parsed.years_experience}")

# %% [markdown]
# ## API backend
#
# Uses OpenAI-compatible `response_format` with `json_schema`. Requires
# an API key (e.g. `OPENAI_API_KEY`).

# %%
#|eval: false
responses_api = llm_generate(
    prompts,
    model="gpt-5.2",
    max_new_tokens=200,
    system_message="Extract job information as JSON.",
    json_schema=schema,
)

parsed_api = JobSummary.model_validate_json(responses_api[0])
print(f"Title: {parsed_api.job_title}")
print(f"Skills: {parsed_api.skills}")
print(f"Years experience: {parsed_api.years_experience}")

# %% [markdown]
# ## Sbatch backend (Isambard)
#
# Submits an sbatch job to Isambard, which uses vLLM's built-in guided
# decoding on a remote GPU. Requires an active Clifton cert.

# %%
#|eval: false
responses_sbatch = llm_generate(
    prompts,
    model="qwen-7b-sbatch",
    max_new_tokens=200,
    system_message="Extract job information as JSON.",
    json_schema=schema,
)

parsed_sbatch = JobSummary.model_validate_json(responses_sbatch[0])
print(f"Title: {parsed_sbatch.job_title}")
print(f"Skills: {parsed_sbatch.skills}")
print(f"Years experience: {parsed_sbatch.years_experience}")
