from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class NumericBinConfig:
    """
    Simple log/linear binning for medication metadata (dosage, rate, duration).
    bin_id: 0 -> missing/unknown; 1..bins -> valid bins
    """
    offset: int
    bins: int
    min_val: float
    max_val: float
    log: bool = True

    def _clamp(self, v: float) -> float:
        return max(self.min_val, min(self.max_val, v))

    def _normalized(self, value: Optional[float]) -> Optional[float]:
        """
        Normalize a numeric metadata value to [0, 1] using log1p or linear scaling.
        Returns None on missing/invalid input.
        """
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None

        v = self._clamp(v)
        if self.log:
            lo = math.log1p(self.min_val)
            hi = math.log1p(self.max_val)
            span = hi - lo if hi > lo else 1.0
            norm = (math.log1p(v) - lo) / span
        else:
            span = (self.max_val - self.min_val) if self.max_val > self.min_val else 1.0
            norm = (v - self.min_val) / span

        return max(0.0, min(1.0, norm))

    def normalize(self, value: Optional[float]) -> float:
        """
        Public helper for folding numeric metadata during collation.
        Returns a clamped, log/linearly scaled value in [0, 1]; 0.0 if missing.
        """
        norm = self._normalized(value)
        return 0.0 if norm is None else norm

    def bin_id(self, value: Optional[float]) -> int:
        if value is None:
            return 0
        norm = self._normalized(value)
        if norm is None:
            return 0

        idx = 1 + int(math.floor(norm * (self.bins - 1))) if self.bins > 1 else 1
        return max(1, min(self.bins, idx))

    def encode(self, value: Optional[float]) -> int:
        """Return global ID = offset + bin_id."""
        return self.offset + self.bin_id(value)
