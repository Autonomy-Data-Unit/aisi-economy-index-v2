# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dev Utils
#
# Helpers for developing pipeline node notebooks interactively. The main entry
# point is `set_node_func_args`, which populates the notebook's global namespace
# with the input data a node would receive when running inside the Net.

# %%
#|default_exp utils

# %%
#|export
import asyncio
import builtins
import importlib
import inspect
import sys
from collections import namedtuple
from pathlib import Path

from netrun.core import Net, NetConfig
from netrun.net._net._context import NodeExecutionContext
from netrun.net.config._nodes import NodeVariable

# %% [markdown]
# ## Config loading

# %%
#|export
def _load_net_config(run_name: str | None = None) -> NetConfig:
    """Load the pipeline NetConfig from ai_index assets, with run_defs injected.

    Args:
        run_name: Which run definition to use for filling node vars.
            Defaults to RUN_NAME env var, then "baseline".
    """
    import os
    from ai_index.const import run_defs_path, netrun_config_path
    from ai_index.run_pipeline import _load_run_defs, _resolve_run_defs

    config_path = netrun_config_path
    run_name = run_name or os.environ.get("RUN_NAME", "baseline")
    run_defs = _load_run_defs(run_defs_path)
    global_vars, node_vars = _resolve_run_defs(run_defs, run_name)

    config = NetConfig.from_file(
        str(config_path),
        global_node_vars=global_vars,
        node_vars=node_vars,
    )
    return config


def _get_merged_node_vars(config: NetConfig, node_name: str) -> dict[str, NodeVariable]:
    """Get merged global + per-node NodeVariable dict for a specific node.

    Respects inherit semantics: if a per-node var has inherit=True and no value,
    the global var is used. If it has inherit=True with a value, the value
    overrides but type/options come from the global var.
    """
    resolved_config = config.resolve_env_vars()
    merged = dict(resolved_config.node_vars or {})
    if resolved_config.graph:
        for node in resolved_config.graph.nodes:
            if node.name == node_name:
                if node.execution_config and node.execution_config.node_vars:
                    for name, var in node.execution_config.node_vars.items():
                        if var.inherit and name in merged:
                            if var.value is not None:
                                # Override value, keep global type/options
                                global_var = merged[name]
                                merged[name] = NodeVariable(
                                    value=var.value,
                                    type=global_var.type,
                                    options=global_var.options,
                                )
                            # else: inherit everything from global (keep merged[name] as-is)
                        else:
                            merged[name] = var
                break
    return merged

# %% [markdown]
# ## Node name resolution

# %%
#|export
def _resolve_node_name(config: NetConfig, bare_name: str) -> str:
    """Resolve a bare node name to its (possibly prefixed) name in the graph.

    With subgraphs, node names are prefixed (e.g., ``embed_onet`` becomes
    ``matching.embed_onet``). This helper finds the full name so that
    ``set_node_func_args`` works without changes to individual node notebooks.

    Args:
        config: The resolved NetConfig.
        bare_name: The unqualified node name (e.g. "embed_onet").

    Returns:
        The full node name (e.g. "matching.embed_onet"), or ``bare_name``
        unchanged if it already matches a node directly.

    Raises:
        ValueError: If the name cannot be found, or matches multiple nodes.
    """
    # Resolve the graph to get all flattened node names
    resolved = config.graph.resolve(net_config=config)
    all_names = [n.name for n in resolved.nodes]

    # Exact match — return as-is
    if bare_name in all_names:
        return bare_name

    # Look for suffix match (e.g. "embed_onet" matches "matching.embed_onet")
    matches = [n for n in all_names if n.endswith(f".{bare_name}")]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise ValueError(
            f"Ambiguous node name '{bare_name}' — matches multiple nodes: {matches}"
        )

    raise ValueError(
        f"Node '{bare_name}' not found in graph. "
        f"Available nodes: {sorted(all_names)}"
    )

# %% [markdown]
# ## Node function resolution

# %%
#|export
def _get_node_func(config: NetConfig, node_name: str):
    """Import and return the Python function for a node.

    Looks up the node's ``factory_args.func`` dotted path from the resolved
    graph config, imports the module, and returns the function object.
    """
    resolved = config.graph.resolve(net_config=config)
    node_map = {n.name: n for n in resolved.nodes}
    if node_name not in node_map:
        raise ValueError(f"Node '{node_name}' not found in graph.")
    node = node_map[node_name]
    func_path = node.factory_args.get("func")
    if not func_path:
        raise ValueError(f"Node '{node_name}' has no 'func' in factory_args.")
    module_path, _, attr_name = func_path.rpartition(".")
    mod = importlib.import_module(module_path)
    return getattr(mod, attr_name)

# %% [markdown]
# ## Salvo retrieval

# %%
#|export
async def _get_input_salvo(config: NetConfig, node_name: str, verbose: bool = True) -> dict[str, list]:
    """Get input salvo for a node — from cache if available, otherwise by running upstream.

    Returns:
        dict mapping port_name -> list of packet values.
    """
    net = Net(config)
    try:
        # Source nodes have no input ports — nothing to retrieve
        node_info = net.nodes[node_name]
        if not node_info.in_port_names:
            if verbose:
                print(f"set_node_func_args: '{node_name}' is a source node (no inputs)")
            return {}

        cached = net.get_cached_input_salvos(node_name)
        if cached:
            if verbose:
                print(f"set_node_func_args: using cached inputs for '{node_name}' ({len(cached)} cached run(s))")
            return cached[-1]  # already dict[str, list[Any]]

        if verbose:
            print(f"set_node_func_args: no cache for '{node_name}', running upstream nodes...")
            _running = {}
            def _on_start(name, epoch_id):
                print(f"  Running {name}...", end="", flush=True)
                _running[epoch_id] = name
            def _on_end(name, epoch_id, record):
                if epoch_id in _running:
                    print(" done")
                    del _running[epoch_id]
            net.on_epoch_start(_on_start)
            net.on_epoch_end(_on_end)

        salvos = await net.run_to_targets(node_name)
        if not salvos:
            return {}
        return salvos[0].packets  # extract dict from TargetInputSalvo
    finally:
        await net.stop()


def _run_async(coro):
    """Run an async coroutine, handling Jupyter's already-running event loop."""
    try:
        asyncio.get_running_loop()
        # Inside Jupyter or another async context — patch to allow nested run()
        import nest_asyncio
        nest_asyncio.apply()
    except RuntimeError:
        pass
    return asyncio.run(coro)

# %% [markdown]
# ## Main entry point

# %%
#|export
_SPECIAL_PARAMS = frozenset({"ctx", "print"})


def set_node_func_args(node_name: str | None = None, *, run_name: str | None = None, return_args=False, load_env=True, verbose=True):
    """Populate the caller's namespace with the inputs a pipeline node would receive.

    Loads input data for a node from the netrun cache (or by running upstream
    nodes via ``Net.run_to_targets``) and injects the values into the caller's
    global namespace so that subsequent notebook cells can use them directly.

    The node's function is looked up from the graph config via its
    ``factory_args.func`` import path, then inspected to determine which
    special parameters (``ctx``, ``print``) it expects.

    Args:
        node_name: The node name (e.g. ``"fetch_onet"``). Bare names are
            resolved against subgraph-prefixed names automatically. If omitted,
            the name is inferred from the current Jupyter notebook filename
            via ``ipynbname``.
        return_args: If True, return the arguments as a namedtuple instead of
            setting them in the caller's globals.
        load_env: If True (default), load ``.env`` via dotenv before resolving
            the netrun config. Set to False if env is already configured.

    Returns:
        If ``return_args`` is True, a namedtuple whose fields are the function
        parameter names. Otherwise ``None`` (values are set in caller's globals).

    Example::

        from dev_utils import set_node_func_args
        set_node_func_args()  # infers node name from notebook filename
        # => adzuna_meta, ctx, print are now available as globals
    """
    if load_env:
        from dotenv import load_dotenv
        load_dotenv()

    inferred = node_name is None
    if inferred:
        import ipynbname
        node_name = ipynbname.name()

    config = _load_net_config(run_name)

    # Resolve bare name to prefixed name (e.g. "embed_onet" -> "matching.embed_onet")
    try:
        name = _resolve_node_name(config, node_name)
    except ValueError as e:
        if inferred:
            raise ValueError(
                f"Node name '{node_name}' was inferred from the notebook filename "
                f"but no matching node was found in the graph. "
                f"Pass the node name explicitly if it differs from the notebook name."
            ) from e
        raise

    # Import the node's function to inspect its signature
    func = _get_node_func(config, name)

    # Retrieve input salvo (cached or computed)
    salvo = _run_async(_get_input_salvo(config, name, verbose=verbose))

    # Extract port values — each port typically has one packet
    args = {}
    for port_name, values in salvo.items():
        args[port_name] = values[0] if len(values) == 1 else values

    # Add special parameters based on what the function signature expects
    sig = inspect.signature(func)
    if "ctx" in sig.parameters:
        node_vars = _get_merged_node_vars(config, name)
        args["ctx"] = NodeExecutionContext(
            epoch_id="dev-0",
            node_name=name,
            _node_vars=node_vars,
        )
    if "print" in sig.parameters:
        args["print"] = builtins.print

    if return_args:
        NodeArgs = namedtuple("NodeArgs", list(args.keys()))
        return NodeArgs(**args)

    # Set in caller's globals
    caller_globals = sys._getframe(1).f_globals
    caller_globals.update(args)

# %% [markdown]
# ## Show node vars

# %%
#|export
def show_node_vars(node_name: str | None = None, *filter_names: str, run_name: str | None = None, load_env=True):
    """Print the node variables available to a pipeline node.

    Shows global and per-node variables with their values, types, and source
    (global, inherited, or node-level override).

    Args:
        node_name: The node name (e.g. ``"fetch_adzuna"``). If omitted,
            inferred from the current Jupyter notebook filename.
        *filter_names: Optional variable names to filter by. If provided,
            only these variables are shown.
        run_name: Which run definition to use. Defaults to RUN_NAME env var,
            then ``"baseline"``.
        load_env: If True (default), load ``.env`` via dotenv first.

    Example::

        from dev_utils import show_node_vars
        show_node_vars()                        # all vars for current node
        show_node_vars("fetch_adzuna")          # all vars for a specific node
        show_node_vars("fetch_adzuna", "years") # only the "years" var
    """
    if load_env:
        from dotenv import load_dotenv
        load_dotenv()

    inferred = node_name is None
    if inferred:
        import ipynbname
        node_name = ipynbname.name()

    config = _load_net_config(run_name)

    # Load raw config (without run_defs) to get declared types, since
    # global_node_vars injection replaces NodeVariable objects and loses type info.
    from ai_index.const import netrun_config_path
    raw_config = NetConfig.from_file(str(netrun_config_path))
    declared_types: dict[str, str] = {
        k: v.type for k, v in (raw_config.node_vars or {}).items()
    }

    try:
        name = _resolve_node_name(config, node_name)
    except ValueError as e:
        if inferred:
            raise ValueError(
                f"Node name '{node_name}' was inferred from the notebook filename "
                f"but no matching node was found in the graph. "
                f"Pass the node name explicitly if it differs from the notebook name."
            ) from e
        raise

    resolved_config = config.resolve_env_vars()

    # Collect global vars
    global_vars: dict[str, NodeVariable] = dict(resolved_config.node_vars or {})

    # Collect raw per-node vars (before inherit resolution)
    per_node_raw: dict[str, NodeVariable] = {}
    if resolved_config.graph:
        for node in resolved_config.graph.nodes:
            if node.name == name:
                if node.execution_config and node.execution_config.node_vars:
                    per_node_raw = dict(node.execution_config.node_vars)
                break

    # Build display rows: (var_name, value, type, source)
    all_var_names = sorted(set(global_vars) | set(per_node_raw))
    if filter_names:
        all_var_names = [n for n in all_var_names if n in filter_names]

    rows = []
    for var_name in all_var_names:
        in_global = var_name in global_vars
        in_node = var_name in per_node_raw

        if in_node and per_node_raw[var_name].inherit:
            if per_node_raw[var_name].value is not None:
                gvar = global_vars[var_name]
                value = per_node_raw[var_name].value
                source = "inherited (overridden)"
            else:
                gvar = global_vars[var_name]
                value = gvar.value
                source = "inherited"
        elif in_node:
            nvar = per_node_raw[var_name]
            value = nvar.value
            source = "node-level"
        else:
            gvar = global_vars[var_name]
            value = gvar.value
            source = "global"

        var_type = declared_types.get(var_name, global_vars.get(var_name, per_node_raw.get(var_name)).type)

        rows.append((var_name, value, var_type, source))

    # Print table
    if not rows:
        print(f"No node vars for '{name}'")
        return

    # Column widths
    headers = ("Name", "Value", "Type", "Source")
    col_widths = [len(h) for h in headers]
    str_rows = []
    for var_name, value, var_type, source in rows:
        vals = (var_name, str(value), var_type, source)
        str_rows.append(vals)
        for i, v in enumerate(vals):
            col_widths[i] = max(col_widths[i], len(v))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(f"Node vars for '{name}':")
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in col_widths)))
    for vals in str_rows:
        print(fmt.format(*vals))
