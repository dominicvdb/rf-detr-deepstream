"""Compare two sets of detections (e.g. Python reference vs DeepStream) and summarise the differences."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(xyxy_a, cls_a, xyxy_b, cls_b, min_iou: float = 0.5) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching: same class, highest IoU first. Returns (index_a, index_b, iou)."""
    candidates = []
    for i in range(len(xyxy_a)):
        for j in range(len(xyxy_b)):
            if cls_a[i] == cls_b[j]:
                v = iou(xyxy_a[i], xyxy_b[j])
                if v >= min_iou:
                    candidates.append((v, i, j))
    candidates.sort(reverse=True)
    used_a, used_b, pairs = set(), set(), []
    for v, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j, v))
    return pairs


@dataclass
class ParityStats:
    n_a: int = 0
    n_b: int = 0
    ious: list[float] = field(default_factory=list)
    dscore: list[float] = field(default_factory=list)          # score_b - score_a (signed)
    unmatched_a_scores: list[float] = field(default_factory=list)
    unmatched_b_scores: list[float] = field(default_factory=list)

    def add(self, xyxy_a, cls_a, score_a, xyxy_b, cls_b, score_b) -> None:
        pairs = match(xyxy_a, cls_a, xyxy_b, cls_b)
        self.n_a += len(xyxy_a)
        self.n_b += len(xyxy_b)
        self.ious += [p[2] for p in pairs]
        self.dscore += [float(score_b[j]) - float(score_a[i]) for i, j, _ in pairs]
        ma, mb = {p[0] for p in pairs}, {p[1] for p in pairs}
        self.unmatched_a_scores += [float(score_a[i]) for i in range(len(xyxy_a)) if i not in ma]
        self.unmatched_b_scores += [float(score_b[j]) for j in range(len(xyxy_b)) if j not in mb]

    def report(self, name_a: str = "A", name_b: str = "B", threshold: float | None = None) -> str:
        ious, ds = np.array(self.ious), np.array(self.dscore)
        absd = np.abs(ds)
        lines = [
            f"detections: {name_a}={self.n_a}  {name_b}={self.n_b}  matched={len(ious)}  "
            f"({100 * len(ious) / max(self.n_a, 1):.2f}% of {name_a})",
            f"IoU:        mean={ious.mean():.4f}  p5={np.percentile(ious, 5):.4f}  min={ious.min():.4f}",
            f"|dscore|:   mean={absd.mean():.4f}  p95={np.percentile(absd, 95):.4f}  max={absd.max():.4f}",
            f"dscore ({name_b}-{name_a}) mean={ds.mean():+.4f}  (systematic bias if far from 0)",
        ]
        for label, scores in ((f"only in {name_a}", self.unmatched_a_scores), (f"only in {name_b}", self.unmatched_b_scores)):
            s = np.array(scores)
            if len(s) == 0:
                lines.append(f"{label}: 0")
                continue
            near = "" if threshold is None else f", {np.mean(s < threshold + 0.05) * 100:.0f}% within 0.05 of threshold"
            lines.append(f"{label}: {len(s)}  (median score {np.median(s):.3f}{near})")
        return "\n".join(lines)
