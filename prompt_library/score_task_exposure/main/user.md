Evaluate this occupation-task pair using the three-level exposure rule.

Occupation: {occupation}
Task: {task}

Decision rule (three-level exposure):

Assign exposure = 0 (NO CHANGE) if:
  - The task requires physical manipulation, in-person presence, or real-time sensory judgment that LLM agents cannot provide.
  - LLM assistance would not meaningfully reduce time or improve quality.
  - The task's core value depends on human judgment, relationships, or accountability in ways that cannot be delegated.

Assign exposure = 1 (HUMAN + LLM COLLABORATION) if:
  - A frontier LLM agent (as of November 2025: LLM + web search + document retrieval + spreadsheets + simple code) can substantially assist the task, reducing time by ≥30% or noticeably improving quality.
  - BUT the human remains essential for: final judgment, quality assurance, contextual adaptation, stakeholder interaction, or accountability.
  - Examples: drafting that needs human review, analysis that informs human decisions, research that a human must synthesize.

Assign exposure = 2 (LLM INDEPENDENT) if:
  - The LLM agent can complete the task end-to-end with minimal or no human involvement, at comparable or better quality.
  - Human role reduces to: initiating the request, spot-checking outputs, or handling rare exceptions.
  - The task has clear success criteria, limited need for contextual judgment, and low cost of errors or easy error detection.
  - Examples: routine data extraction, template generation, standard code for well-defined specs, factual lookups and summarization.

Key distinction between 1 and 2:
  - If a competent worker would need to substantially review, edit, or adapt the LLM output before it's usable → exposure = 1
  - If a competent worker could trust the output with only cursory checking → exposure = 2

Assumptions:
  - Typical worker proficiency in occupation.
  - LLM agent can plan, draft, transform text, summarize, write code, generate analysis, retrieve factual info, and structure workflows.
  - No physical tools or robotic execution available.
  - When uncertain, prefer lower exposure (0 over 1, 1 over 2).