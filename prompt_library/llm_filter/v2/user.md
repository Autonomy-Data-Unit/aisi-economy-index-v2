TASK
Audit occupation matches. DROP candidates that are NOT functional matches for the job.

KEEP POLICY
There are {n_candidates} candidates (1-based).
- Default: KEEP 2-3 candidates.
- KEEP 1 ONLY if you are certain every other candidate is clearly wrong.
- If more than 3 are valid, drop the most generic ones to fit the 3-candidate cap.
- When in doubt, KEEP rather than DROP if functionally plausible.

RANKING NOTE
Candidates are ranked by a reranker model (rank 1 = highest reranker score). This is a moderate prior but not infallible.

JOB ADVERTISEMENT
Title: {job_ad_title}
Category: {job_sector_category}

Description:
{full_ad_excerpt}

CANDIDATES (1-based)
{candidates_str}

ANCHOR (FUNCTION FIRST)
- Identify the functional anchor from the job title and description (function, not seniority).
- You MUST keep the anchor unless core evidence contradicts it.
- Title keyword lock: if the title contains a clear functional keyword and a direct-match candidate exists, you MUST keep it.

GATES
- Manager rule: if title does NOT include Manager/Lead/Director, keep manager roles only if the description mentions supervision, rotas, hiring, budgeting.
- IT lock: if title/description mention concrete tech (Python/SQL/APIs/systems), keep relevant IT roles even if the category is non-IT.

FINAL CHECK
- Keep 2-3 by default.
- Keeping only 1 requires clear mismatch for all others.