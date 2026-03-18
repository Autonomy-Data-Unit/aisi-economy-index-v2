# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # deploy.run_pipeline
#
# Run commands on the remote server in the background (detached from SSH).
# The process continues even if you close the terminal.
#
# Usage:
#     uv run remote-run-bg <command...>               # run any command in background
#     uv run remote-run-pipeline <run_name>            # shortcut for run-pipeline
#     uv run remote-bg-log [--follow] [N]              # check progress
#     uv run remote-bg-kill                             # kill the background process

# %%
#|default_exp run_pipeline

# %%
#|export
import subprocess
import sys

from deploy.config import get_server_ip, load_deploy_config, run_ssh

REMOTE_LOG = "/root/bg-job.log"
REMOTE_PID = "/root/bg-job.pid"

# %%
#|export
def _check_running(ip: str) -> tuple[bool, str | None]:
    """Check if a background job is running. Returns (is_running, pid_or_none)."""
    result = run_ssh(ip, f"test -f {REMOTE_PID} && kill -0 $(cat {REMOTE_PID}) 2>/dev/null && echo running || echo stopped",
                     check=False, capture=True)
    if result.stdout.strip() == "running":
        pid = run_ssh(ip, f"cat {REMOTE_PID}", capture=True).stdout.strip()
        return True, pid
    return False, None

# %%
#|export
def run_bg():
    """Run an arbitrary command on the remote, detached from SSH."""
    if len(sys.argv) < 2:
        print("Usage: remote-run-bg <command...>", file=sys.stderr)
        sys.exit(1)

    user_cmd = " ".join(sys.argv[1:])
    config = load_deploy_config()
    ip = get_server_ip(config["server"]["name"])
    repo_path = config["repo"]["path"]

    is_running, pid = _check_running(ip)
    if is_running:
        print(f"A background job is already running (PID {pid}).")
        print(f"  Log:   uv run remote-bg-log")
        print(f"  Kill:  uv run remote-bg-kill")
        sys.exit(1)

    cmd = (
        f"cd {repo_path} && "
        f"nohup /root/.local/bin/uv run --no-dev {user_cmd} "
        f"> {REMOTE_LOG} 2>&1 </dev/null & echo $! > {REMOTE_PID}"
    )
    subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
         "-f", f"root@{ip}", cmd],
        check=True,
    )
    pid = run_ssh(ip, f"cat {REMOTE_PID}", capture=True).stdout.strip()
    print(f"Background job started (PID {pid}): {user_cmd}")
    print(f"  Log:      uv run remote-bg-log")
    print(f"  Follow:   uv run remote-bg-log --follow")
    print(f"  Kill:     uv run remote-bg-kill")

# %%
#|export
def start():
    """Start the pipeline on the remote (convenience wrapper around run_bg)."""
    if len(sys.argv) < 2:
        print("Usage: remote-run-pipeline <run_name>", file=sys.stderr)
        sys.exit(1)

    run_name = sys.argv[1]
    # Rewrite argv so run_bg sees the full command
    sys.argv = [sys.argv[0], "run-pipeline", run_name]
    run_bg()

# %%
#|export
def log():
    """Tail the remote background job log."""
    args = sys.argv[1:]
    follow = "--follow" in args or "-f" in args
    args = [a for a in args if a not in ("--follow", "-f")]
    n_lines = int(args[0]) if args else 50

    config = load_deploy_config()
    ip = get_server_ip(config["server"]["name"])

    is_running, _ = _check_running(ip)
    status = "running" if is_running else "stopped"

    if follow:
        print(f"[job {status}] Following {REMOTE_LOG} (Ctrl+C to stop)...")
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no",
             f"root@{ip}",
             f"tail -f {REMOTE_LOG}"],
        )
    else:
        print(f"[job {status}] Last {n_lines} lines of {REMOTE_LOG}:")
        print()
        run_ssh(ip, f"tail -n {n_lines} {REMOTE_LOG}")

# %%
#|export
def kill_bg():
    """Kill the remote background job."""
    config = load_deploy_config()
    ip = get_server_ip(config["server"]["name"])

    is_running, pid = _check_running(ip)
    if not is_running:
        print("No background job is running.")
        return

    run_ssh(ip, f"kill {pid}")
    print(f"Killed background job (PID {pid}).")
