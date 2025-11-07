# src/ehr_hier/tokenizers/base_encoder.py
from __future__ import annotations

from typing import Protocol, List, Any, Dict

from src.ehr_hier.data.token_types import TokenTriplet, TokenCategory
from src.ehr_hier.tokenizers.measurement_encoder import (
    MeasurementTokenEncoder,
    MeasurementEncoderConfig,
)


class EventTokenEncoder(Protocol):
    """
    Common interface for all event-type encoders.

    Implementations to add:
      - MeasurementTokenEncoder  → measurement events (labs/vitals) → value tokens
      - DiagnosisTokenEncoder    → ICD/MEDTOK → diagnosis tokens
      - ProcedureTokenEncoder    → procedures → tokens
      - MedicationTokenEncoder   → meds/infusions → tokens
      - etc.
    """
    # Which TokenCategory this encoder is responsible for
    category: TokenCategory

    def encode_event(self, ev: Any, dt_hours: float) -> List[TokenTriplet]:
        """
        Turn a single MEDS event into zero or more TokenTriplet(s).

        Parameters
        ----------
        ev : meds_reader Event (or compatible)
            Must provide at least .code and .time; additional attributes are
            encoder-specific (e.g., .numeric_value for measurements).
        dt_hours : float
            Time since previous *emitted token* in hours (from timeline builder).

        Returns
        -------
        List[TokenTriplet]
            [] if the event is irrelevant / unusable for this encoder,
            or [TokenTriplet, ...] if it produces tokens.
        """
        ...

    def reset_state(self) -> None:
        """
        Optional hook: reset any per-subject internal state.

        Called once per subject by build_subject_timeline before iterating
        that subject's events. Encoders that are stateless can implement a
        simple `return None`.
        """
        ...


def build_base_encoders(
    meas_cfg: MeasurementEncoderConfig,
) -> Dict[TokenCategory, EventTokenEncoder]:
    """
    Construct the base mapping from TokenCategory → encoder instance.

    Currently:
      - MEASUREMENT → MeasurementTokenEncoder

    Parameters
    ----------
    meas_cfg : MeasurementEncoderConfig
        Config for MeasurementTokenEncoder (cVAE/tokenizer paths, stats, code2id, etc.).

    Returns
    -------
    encoders : Dict[TokenCategory, EventTokenEncoder]
        Use this dict with build_subject_timeline(...)
    """
    encoders: Dict[TokenCategory, EventTokenEncoder] = {}

    # 1) Measurement encoder (numeric value tokens)
    meas_encoder = MeasurementTokenEncoder(meas_cfg)
    encoders[TokenCategory.MEASUREMENT] = meas_encoder

    return encoders
