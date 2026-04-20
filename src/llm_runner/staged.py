"""Resolver registry for staged inputs on Isambard.

When a manifest entry has type="staged", the CLI calls resolve_staged_input()
which looks up the named resolver and calls it with the source file paths
and chunk parameters. The resolver reads the pre-staged files from the
Lustre filesystem and builds the model inputs locally on the GPU node.
"""

import json
from pathlib import Path
from typing import Any, Callable

_RESOLVERS: dict[str, Callable] = {}


def resolver(name: str):
    """Register a function as a staged input resolver."""
    def decorator(fn):
        _RESOLVERS[name] = fn
        return fn
    return decorator


def resolve_staged_input(entry: dict) -> Any:
    """Resolve a staged manifest entry into actual model inputs.

    Args:
        entry: Dict with keys: type="staged", resolver, sources, params.

    Returns:
        The resolved input value (e.g. list of items for rerank_pairs).
    """
    resolver_name = entry["resolver"]
    fn = _RESOLVERS[resolver_name]
    sources = entry["sources"]
    params = entry["params"]
    return fn(sources, params)


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------

@resolver("rerank_pairs_items")
def _resolve_rerank_pairs_items(sources: dict, params: dict) -> list:
    """Build rerank_pairs items from staged files.

    Reads filtered_matches.parquet, ad_texts parquet, and onet_docs JSON
    from the Lustre filesystem. Filters to the chunk's ad_ids, then builds
    the (query, doc_texts) items list.

    Sources:
        filtered_matches: parquet with columns (ad_id, onet_code, ...)
        ad_texts: parquet with columns (id, title, description)
        onet_docs: JSON mapping onet_code -> document text

    Params:
        ad_ids: list of ad IDs for this chunk
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa

    ad_ids = params["ad_ids"]

    # Read onet docs
    onet_docs_path = sources["onet_docs"]["remote_path"]
    with open(onet_docs_path) as f:
        onet_docs = json.load(f)

    # Read filtered matches for this chunk using predicate pushdown
    matches_path = sources["filtered_matches"]["remote_path"]
    matches_table = pq.read_table(
        matches_path,
        filters=pc.field("ad_id").isin(ad_ids),
    )

    if len(matches_table) == 0:
        return []

    # Group candidates by ad_id
    candidates_by_ad = {}
    ad_id_col = matches_table.column("ad_id").to_pylist()
    onet_code_col = matches_table.column("onet_code").to_pylist()
    for i in range(len(matches_table)):
        aid = ad_id_col[i]
        code = onet_code_col[i]
        if aid not in candidates_by_ad:
            candidates_by_ad[aid] = []
        candidates_by_ad[aid].append(code)

    # Read ad texts for ads that have candidates
    ads_with_candidates = [aid for aid in ad_ids if aid in candidates_by_ad]
    if not ads_with_candidates:
        return []

    ad_texts_path = sources["ad_texts"]["remote_path"]
    ads_table = pq.read_table(
        ad_texts_path,
        filters=pc.field("id").isin(ads_with_candidates),
        columns=["id", "title", "description"],
    )

    # Build lookup: ad_id -> (title, description)
    ads_lookup = {}
    id_col = ads_table.column("id").to_pylist()
    title_col = ads_table.column("title").to_pylist()
    desc_col = ads_table.column("description").to_pylist()
    for i in range(len(ads_table)):
        ads_lookup[id_col[i]] = (title_col[i], desc_col[i])

    # Build items: (query, [doc_texts]) for each ad
    items = []
    for ad_id in ads_with_candidates:
        if ad_id not in ads_lookup:
            continue
        title, description = ads_lookup[ad_id]
        query = f"{title}. {str(description or '')[:6000]}"
        onet_codes = candidates_by_ad[ad_id]
        doc_texts = [onet_docs[code] for code in onet_codes]
        items.append((query, doc_texts))

    print(f"  staged resolver [rerank_pairs_items]: {len(items)} items from {len(ad_ids)} ad_ids")
    return items
