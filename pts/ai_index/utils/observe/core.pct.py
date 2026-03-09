# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # observe.core
#
# `NetObserver` — extracts status information from a live `Net` object.

# %%
#|default_exp utils.observe.core

# %%
#|export
from netrun.core import Net
from ai_index.utils.observe.models import (
    NetStatus, NodeStatus, EpochInfo, EdgeStatus, LogEntry, ControlResponse,
)

def _state_str(state) -> str:
    """Convert an EpochState (Rust-backed enum without .name) to a clean string."""
    return str(state).split(".")[-1]

# %%
#|export
class NetObserver:
    """Reads status and logs from a live Net object."""

    def __init__(self, net: Net):
        self.net = net

    def get_status(self) -> NetStatus:
        net = self.net
        node_names = list(net.nodes.keys())
        busy = [n for n in node_names if net.nodes[n].is_busy]
        idle = [n for n in node_names if not net.nodes[n].is_busy]

        epochs_by_state: dict[str, int] = {}
        for record in net.epochs.values():
            state = _state_str(record.state)
            epochs_by_state[state] = epochs_by_state.get(state, 0) + 1

        return NetStatus(
            node_names=node_names,
            edge_count=len(net.edges),
            total_epochs=len(net.epochs),
            epochs_by_state=epochs_by_state,
            busy_nodes=busy,
            idle_nodes=idle,
        )

    def get_nodes(self) -> list[NodeStatus]:
        return [self.get_node(name) for name in self.net.nodes]

    def get_node(self, name: str) -> NodeStatus:
        node = self.net.nodes[name]
        return NodeStatus(
            name=name,
            enabled=node.enabled,
            epoch_count=node.epoch_count,
            is_busy=node.is_busy,
            running_epoch_ids=[str(e.id) for e in node.running_epochs],
            startable_epoch_ids=[str(e.id) for e in node.startable_epochs],
            in_ports=node.in_port_names,
            out_ports=node.out_port_names,
        )

    def get_epochs(self) -> list[EpochInfo]:
        result = []
        for record in self.net.epochs.values():
            duration = None
            if record.started_at and record.ended_at:
                duration = (record.ended_at - record.started_at).total_seconds()

            state = _state_str(record.state)

            result.append(EpochInfo(
                epoch_id=str(record.id),
                node_name=record.node_name,
                state=state,
                was_cancelled=record.was_cancelled,
                was_cache_hit=record.was_cache_hit,
                created_at=str(record.created_at),
                started_at=str(record.started_at) if record.started_at else None,
                ended_at=str(record.ended_at) if record.ended_at else None,
                duration_seconds=duration,
                pool_worker_label=record.pool_worker_label,
            ))
        return result

    def get_edges(self) -> list[EdgeStatus]:
        return [
            EdgeStatus(
                source_node=e.source_node,
                source_port=e.source_port,
                target_node=e.target_node,
                target_port=e.target_port,
                packet_count=e.packet_count,
            )
            for e in self.net.edges
        ]

    def get_node_logs(self, node_name: str) -> list[LogEntry]:
        logs = self.net.get_node_logs(node_name)
        return [
            LogEntry(timestamp=str(ts), message=msg, node_name=node_name)
            for ts, msg in logs
        ]

    def get_all_logs(self) -> list[LogEntry]:
        logs = self.net.get_all_logs_chronological()
        return [
            LogEntry(timestamp=str(ts), message=msg, node_name=node_name, epoch_id=str(epoch_id))
            for ts, epoch_id, node_name, msg in logs
        ]

    # -- Control methods --

    def enable_node(self, name: str) -> ControlResponse:
        self.net.enable_node(name)
        return ControlResponse(ok=True, message=f"Node '{name}' enabled")

    def disable_node(self, name: str) -> ControlResponse:
        self.net.disable_node(name)
        return ControlResponse(ok=True, message=f"Node '{name}' disabled")

    def send_control(self, node_name: str, control_type: str, value: str | int | None = None) -> ControlResponse:
        self.net.send_control(node_name, control_type, value)
        return ControlResponse(ok=True, message=f"Sent '{control_type}' to node '{node_name}'")

    def inject_data(self, node_name: str, port_name: str, values: list) -> ControlResponse:
        self.net.inject_data(node_name, port_name, values)
        return ControlResponse(ok=True, message=f"Injected {len(values)} value(s) into '{node_name}.{port_name}'")
