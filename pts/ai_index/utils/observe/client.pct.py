# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # observe.client
#
# `NetObserverClient` — async HTTP client that mirrors the `NetObserver` interface.
# Uses async httpx so it works in notebooks/pipelines sharing an event loop with
# the `ObserveServer`.

# %%
#|default_exp utils.observe.client

# %%
#|export
import httpx
from ai_index.utils.observe.models import (
    NetStatus, NodeStatus, EpochInfo, EdgeStatus, LogEntry,
)

# %%
#|export
class NetObserverClient:
    """Async HTTP client that mirrors the NetObserver interface.

    Uses async httpx internally so it works when the ObserveServer runs on
    the same event loop (e.g. in a notebook or alongside a netrun pipeline).

    Usage::

        client = NetObserverClient("http://localhost:8000")
        status = await client.get_status()
        nodes = await client.get_nodes()
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url)

    async def get_status(self) -> NetStatus:
        r = await self._client.get("/status")
        r.raise_for_status()
        return NetStatus(**r.json())

    async def get_nodes(self) -> list[NodeStatus]:
        r = await self._client.get("/nodes")
        r.raise_for_status()
        return [NodeStatus(**n) for n in r.json()]

    async def get_node(self, name: str) -> NodeStatus:
        r = await self._client.get(f"/nodes/{name}")
        r.raise_for_status()
        return NodeStatus(**r.json())

    async def get_epochs(self) -> list[EpochInfo]:
        r = await self._client.get("/epochs")
        r.raise_for_status()
        return [EpochInfo(**e) for e in r.json()]

    async def get_edges(self) -> list[EdgeStatus]:
        r = await self._client.get("/edges")
        r.raise_for_status()
        return [EdgeStatus(**e) for e in r.json()]

    async def get_node_logs(self, node_name: str) -> list[LogEntry]:
        r = await self._client.get(f"/logs/{node_name}")
        r.raise_for_status()
        return [LogEntry(**entry) for entry in r.json()]

    async def get_all_logs(self) -> list[LogEntry]:
        r = await self._client.get("/logs")
        r.raise_for_status()
        return [LogEntry(**entry) for entry in r.json()]

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
