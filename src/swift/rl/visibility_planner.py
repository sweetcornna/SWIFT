from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass


Point2 = tuple[float, float]
Circle = tuple[Point2, float]


@dataclass(frozen=True)
class VisibilityPlannerConfig:
    clearance: float = 0.18
    samples: int = 16

    def __post_init__(self) -> None:
        clearance = float(self.clearance)
        if not math.isfinite(clearance) or clearance <= 0.0:
            raise ValueError("clearance must be positive")
        samples = int(self.samples)
        if samples < 8:
            raise ValueError("samples must be at least 8")
        object.__setattr__(self, "clearance", clearance)
        object.__setattr__(self, "samples", samples)


def visibility_waypoint_from_observation(
    observation: Sequence[float],
    config: VisibilityPlannerConfig | None = None,
) -> tuple[float, float, float]:
    values = tuple(float(value) for value in observation)
    if len(values) < 15 or (len(values) - 15) % 4 != 0:
        raise ValueError("observation must contain 15 values plus zero or more 4-value obstacle slots")

    planner_config = config or VisibilityPlannerConfig()
    relative_goal = values[7:10]
    goal_xy = (relative_goal[0], relative_goal[1])
    circles = _inflated_obstacles(values, planner_config.clearance)
    if not circles or _segment_is_clear((0.0, 0.0), goal_xy, circles):
        return relative_goal

    nodes = _graph_nodes(goal_xy, circles, planner_config.samples)
    path = _shortest_visible_path(nodes, circles)
    if len(path) < 2:
        return relative_goal
    waypoint = nodes[path[1]]
    return (waypoint[0], waypoint[1], relative_goal[2])


def _inflated_obstacles(observation: tuple[float, ...], clearance: float) -> tuple[Circle, ...]:
    slots = []
    if len(observation) == 15:
        slots.append((observation[10:13], observation[13]))
    else:
        for offset in range(15, len(observation), 4):
            slots.append((observation[offset : offset + 3], observation[offset + 3]))

    circles = []
    for relative, radius in slots:
        if radius <= 0.0:
            continue
        circles.append(((relative[0], relative[1]), radius + clearance))
    return tuple(circles)


def _graph_nodes(goal: Point2, circles: tuple[Circle, ...], samples: int) -> tuple[Point2, ...]:
    nodes: list[Point2] = [(0.0, 0.0), goal]
    ring_scale = 1.0 / math.cos(math.pi / float(samples))
    for center, radius in circles:
        ring_radius = radius * ring_scale
        for index in range(samples):
            angle = 2.0 * math.pi * float(index) / float(samples)
            nodes.append(
                (
                    center[0] + ring_radius * math.cos(angle),
                    center[1] + ring_radius * math.sin(angle),
                )
            )
    return tuple(nodes)


def _shortest_visible_path(nodes: tuple[Point2, ...], circles: tuple[Circle, ...]) -> tuple[int, ...]:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if not _segment_is_clear(nodes[left], nodes[right], circles):
                continue
            distance = math.dist(nodes[left], nodes[right])
            adjacency[left].append((right, distance))
            adjacency[right].append((left, distance))

    distances = [math.inf] * len(nodes)
    previous: list[int | None] = [None] * len(nodes)
    distances[0] = 0.0
    queue: list[tuple[float, int]] = [(0.0, 0)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node] + 1e-12:
            continue
        if node == 1:
            break
        for neighbor, edge_length in adjacency[node]:
            candidate = distance + edge_length
            if candidate + 1e-12 >= distances[neighbor]:
                continue
            distances[neighbor] = candidate
            previous[neighbor] = node
            heapq.heappush(queue, (candidate, neighbor))

    if not math.isfinite(distances[1]):
        return ()
    path = [1]
    while path[-1] != 0:
        parent = previous[path[-1]]
        if parent is None:
            return ()
        path.append(parent)
    path.reverse()
    return tuple(path)


def _segment_is_clear(start: Point2, end: Point2, circles: tuple[Circle, ...]) -> bool:
    return all(_segment_distance_to_point(start, end, center) + 1e-9 >= radius for center, radius in circles)


def _segment_distance_to_point(start: Point2, end: Point2, point: Point2) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-18:
        return math.dist(start, point)
    projection = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    projection = min(max(projection, 0.0), 1.0)
    closest = (start[0] + projection * dx, start[1] + projection * dy)
    return math.dist(closest, point)
