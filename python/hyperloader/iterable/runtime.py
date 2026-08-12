"""Lane construction, source advancement, snapshots, and recovery."""

from __future__ import annotations

import pickle
import warnings
from collections import deque
from contextlib import nullcontext
from typing import Any

from hyperloader.rng import _user_code_context

from . import sharding
from .lane import IterableLane
from .protocol import capture_source_state, has_state_pair, restore_source_state
from .rng import IterableRngSession
from .snapshot import SnapshotRing, SourceSnapshot
from .state import IterableCheckpoint, LaneCheckpoint
from .worker_info import lane_worker_info


class IterableLaneRuntime:
    """Own live lanes and their engine-side replay state."""

    def __init__(self, loader: Any, epoch: int, lane_count: int) -> None:
        self.loader = loader
        self.epoch = epoch
        self.lane_count = lane_count
        self.all_lanes: dict[int, IterableLane] = {}
        self.rings: dict[int, SnapshotRing] = {}
        self.delivered_arrivals = {lane: 0 for lane in range(lane_count)}

    def build_lanes(self, resume: IterableCheckpoint | None) -> deque[IterableLane]:
        """Instantiate every logical lane and apply optional continuation state."""
        if resume is not None and any(not lane.stateful for lane in resume.lanes):
            self._restart_notice()
            resume = None
        checkpoints = (
            {} if resume is None else {lane.lane: lane for lane in resume.lanes}
        )
        for identity in range(self.lane_count):
            self._build_lane(identity, checkpoints.get(identity))
        order = tuple(range(self.lane_count)) if resume is None else resume.lane_order
        return deque(self.all_lanes[identity] for identity in order)

    def next_batch(self, lane: IterableLane) -> tuple[list[Any], bool]:
        """Advance one lane by one lane-whole batch and capture its boundary."""
        values, exhausted = self._advance(lane, self.loader.batch_size or 1)
        if values:
            lane.produced_batches += 1
            self._capture_snapshot(lane)
        return values, exhausted

    def mark_delivered(self, lane: IterableLane) -> None:
        """Advance one lane's consumer-visible arrival frontier."""
        self.delivered_arrivals[lane.identity] = lane.arrival
        self.rings[lane.identity].discard_before(lane.arrival)
        self.rings[lane.identity].discard_before(lane.arrival)

    def capture_checkpoint(self, lane_order: tuple[int, ...]) -> IterableCheckpoint:
        """Select the newest source state at each delivered frontier."""
        lanes = []
        for identity in range(self.lane_count):
            ring = self.rings[identity]
            delivered = self.delivered_arrivals[identity]
            selected = ring.select(delivered)
            lanes.append(
                LaneCheckpoint(
                    lane=identity,
                    delivered_arrival=delivered,
                    stateful=ring.stateful,
                    snapshot_arrival=(None if selected is None else selected.arrival),
                    snapshot=None if selected is None else selected.payload,
                )
            )
        return IterableCheckpoint(
            root_seed=self.loader.root_seed,
            epoch=self.epoch,
            world_size=self.loader._distributed_topology.world_size,
            lane_count=self.lane_count,
            lane_order=lane_order,
            lanes=tuple(lanes),
            fingerprint=self.loader._fingerprint,
        )

    def recover_lane(
        self,
        identity: int,
        lane_order: tuple[int, ...],
    ) -> deque[IterableLane]:
        """Rebuild an active lane from its delivered source checkpoint."""
        if identity not in lane_order:
            raise ValueError("iterable recovery lane is not active")
        ring = self.rings[identity]
        if not ring.stateful:
            self._restart_notice()
            self.all_lanes.clear()
            self.rings.clear()
            self.delivered_arrivals = {lane: 0 for lane in range(self.lane_count)}
            for lane in range(self.lane_count):
                self._build_lane(lane, None)
            return deque(self.all_lanes[lane] for lane in range(self.lane_count))
        delivered = self.delivered_arrivals[identity]
        selected = ring.select(delivered)
        checkpoint = LaneCheckpoint(
            lane=identity,
            delivered_arrival=delivered,
            stateful=ring.stateful,
            snapshot_arrival=None if selected is None else selected.arrival,
            snapshot=None if selected is None else selected.payload,
        )
        replacement = self._build_lane(identity, checkpoint)
        return deque(
            replacement if lane == identity else self.all_lanes[lane]
            for lane in lane_order
        )

    def _build_lane(
        self,
        identity: int,
        checkpoint: LaneCheckpoint | None,
    ) -> IterableLane:
        payload = self.loader._iterable_payload
        dataset = self.loader.dataset if payload is None else pickle.loads(payload)
        dataset = sharding.apply_source_shard(
            dataset,
            self.loader._distributed_topology,
            identity,
            self.lane_count,
        )
        with self._worker_context(identity, dataset, None):
            if self.loader.num_workers != 0 and self.loader.worker_init_fn is not None:
                self.loader.worker_init_fn(identity)
        stateful = has_state_pair(dataset)
        if checkpoint is not None and checkpoint.stateful != stateful:
            raise ValueError(
                "iterable source stateful protocol changed since checkpoint"
            )
        start, selected = self._restore_selected(identity, dataset, checkpoint)
        with self._worker_context(identity, dataset, None):
            iterator = iter(dataset)
        lane = IterableLane(identity, dataset, iterator, arrival=start)
        ring = self._new_ring(stateful)
        if selected is not None:
            ring.seed(selected)
        self.all_lanes[identity] = lane
        self.rings[identity] = ring
        if stateful and ring.snapshotless:
            self._snapshotless_notice("snapshot cadence is off")
        target = 0 if checkpoint is None else checkpoint.delivered_arrival
        self._replay_to(lane, target)
        self.delivered_arrivals[identity] = target
        self._capture_snapshot(lane, force=True)
        return lane

    def _restore_selected(
        self,
        identity: int,
        dataset: Any,
        checkpoint: LaneCheckpoint | None,
    ) -> tuple[int, SourceSnapshot | None]:
        if checkpoint is None or checkpoint.snapshot is None:
            return 0, None
        restored = pickle.loads(checkpoint.snapshot)
        if not isinstance(restored, dict):
            raise TypeError("iterable source snapshot must decode to a dictionary")
        with self._worker_context(identity, dataset, None):
            restore_source_state(dataset, restored)
        arrival = int(checkpoint.snapshot_arrival)
        return arrival, SourceSnapshot(arrival, checkpoint.snapshot)

    def _new_ring(self, stateful: bool) -> SnapshotRing:
        factors = self.loader.config.factors
        cadence = factors.f_snap
        step = 1 if cadence == "off" else int(cadence)
        depth = max(2, (int(factors.b_buf) + step - 1) // step + 1)
        return SnapshotRing(
            stateful=stateful,
            cadence=cadence,
            maximum_bytes=int(factors.f_snap_bytes),
            depth=depth,
        )

    def _advance(self, lane: IterableLane, count: int) -> tuple[list[Any], bool]:
        values = []
        exhausted = False
        session = IterableRngSession()
        try:
            while len(values) < count:
                sample = session.install(
                    self.loader.root_seed,
                    self.epoch,
                    self.loader._distributed_topology.rank,
                    lane.identity,
                    lane.arrival,
                )
                with (
                    self._worker_context(
                        lane.identity,
                        lane.dataset,
                        sample[0],
                    ),
                    _user_code_context(sample),
                ):
                    try:
                        value = next(lane.iterator)
                    except StopIteration:
                        exhausted = True
                        break
                values.append(value)
                lane.arrival += 1
        finally:
            session.close()
        return values, exhausted

    def _replay_to(self, lane: IterableLane, target: int) -> None:
        remaining = target - lane.arrival
        if remaining < 0:
            raise ValueError("iterable snapshot exceeds the delivered cursor")
        if remaining == 0:
            return
        values, exhausted = self._advance(lane, remaining)
        if exhausted or len(values) != remaining:
            raise RuntimeError("iterable source ended before its replay cursor")

    def _capture_snapshot(self, lane: IterableLane, *, force: bool = False) -> None:
        ring = self.rings[lane.identity]
        if not ring.due(lane.produced_batches, force=force):
            return
        with self._worker_context(lane.identity, lane.dataset, None):
            state = capture_source_state(lane.dataset)
        try:
            payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as error:
            raise TypeError("iterable source state must be pickleable") from error
        if not ring.push(lane.arrival, payload):
            self._snapshotless_notice("source state exceeds factors.f_snap_bytes")

    def _snapshotless_notice(self, reason: str) -> None:
        if self.loader._iterable_snapshot_notice_emitted:
            return
        warnings.warn(
            f"Iterable snapshot storage is disabled because {reason}; "
            "resume replays this lane from arrival zero.",
            UserWarning,
            stacklevel=4,
        )
        self.loader._iterable_snapshot_notice_emitted = True

    def _restart_notice(self) -> None:
        if self.loader._iterable_restart_notice_emitted:
            return
        warnings.warn(
            "A plain iterable source has no state pair; resume restarts its epoch.",
            UserWarning,
            stacklevel=4,
        )
        self.loader._iterable_restart_notice_emitted = True

    def _worker_context(self, identity: int, dataset: Any, seed: int | None) -> Any:
        if self.loader.num_workers == 0:
            return nullcontext()
        return lane_worker_info(identity, self.lane_count, dataset, seed)
