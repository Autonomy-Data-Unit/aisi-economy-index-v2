TASK
Audit occupation matches. Select which candidates are functional matches for the job.

There are {n_candidates} candidates (1-based).

RANKING NOTE
Candidates are ranked by embedding similarity (rank 1 = highest similarity). This is a weak prior. Do NOT assume rank 1 is correct.

JOB ADVERTISEMENT
Title: {job_ad_title}
Category: {job_sector_category}

Description:
{full_ad_excerpt}

CANDIDATES (1-based)
{candidates_str}

RULES
- Identify the functional anchor from the job title and description (function, not seniority).
- You MUST keep the anchor unless core evidence contradicts it.
- Title keyword lock: if the title contains a clear functional keyword and a direct-match candidate exists, you MUST keep it.
- Manager rule: if title does NOT include Manager/Lead/Director, keep manager roles only if the description mentions supervision, rotas, hiring, budgeting.
- IT lock: if title/description mention concrete tech (Python/SQL/APIs/systems), keep relevant IT roles even if the category is non-IT.
- When in doubt, KEEP rather than DROP if functionally plausible.

Respond with the indices of the candidates to KEEP.