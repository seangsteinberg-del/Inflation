"""Parquet-based caching layer for intermediate pipeline results.

Caches cleaned surfaces, SABR parameters, extracted distributions, GMM
estimates, and final disaster probabilities so that a 3-4 hour full sample
run can be resumed if interrupted.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inflation_disaster.config import settings
from inflation_disaster.data.schemas import MarkovParams, DisasterProbability

log = logging.getLogger("inflation_disaster.data.cache")


class ResultsCache:
    """File-based cache for pipeline intermediate and final results.

    Directory structure:
        cache_dir/
            distributions/     # N and Q bin probabilities per date
            markov_params/     # GMM-estimated parameters per date
            disaster_probs/    # Final DisasterProbability results
            audit_log.csv      # Data quality and convergence audit trail
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or settings.processed_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "distributions").mkdir(exist_ok=True)
        (self.cache_dir / "markov_params").mkdir(exist_ok=True)
        (self.cache_dir / "disaster_probs").mkdir(exist_ok=True)

        self._audit_log_path = self.cache_dir / "audit_log.csv"
        self._audit_records: list[dict] = []

    # --- Distributions ---

    def save_distributions(
        self, dt: date, region: str, maturity: int, measure: str, probs: np.ndarray,
    ):
        key = f"{region}_{dt}_{maturity}y_{measure}"
        path = self.cache_dir / "distributions" / f"{key}.npy"
        np.save(path, probs)

    def load_distributions(
        self, dt: date, region: str, maturity: int, measure: str,
    ) -> np.ndarray | None:
        key = f"{region}_{dt}_{maturity}y_{measure}"
        path = self.cache_dir / "distributions" / f"{key}.npy"
        if path.exists():
            return np.load(path)
        return None

    def has_distributions(self, dt: date, region: str, maturity: int, measure: str) -> bool:
        key = f"{region}_{dt}_{maturity}y_{measure}"
        return (self.cache_dir / "distributions" / f"{key}.npy").exists()

    # --- Markov params ---

    def save_markov_params(self, dt: date, region: str, params: MarkovParams):
        key = f"{region}_{dt}"
        path = self.cache_dir / "markov_params" / f"{key}.json"
        path.write_text(json.dumps(params.model_dump(), indent=2))

    def load_markov_params(self, dt: date, region: str) -> MarkovParams | None:
        key = f"{region}_{dt}"
        path = self.cache_dir / "markov_params" / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return MarkovParams(**data)
        return None

    def has_markov_params(self, dt: date, region: str) -> bool:
        return (self.cache_dir / "markov_params" / f"{region}_{dt}.json").exists()

    # --- Disaster probabilities ---

    def save_disaster_prob(self, result: DisasterProbability):
        key = f"{result.region}_{result.date}_d{result.threshold}"
        path = self.cache_dir / "disaster_probs" / f"{key}.json"
        path.write_text(json.dumps(result.model_dump(), default=str, indent=2))

    def load_disaster_prob(self, dt: date, region: str, threshold: float) -> DisasterProbability | None:
        key = f"{region}_{dt}_d{threshold}"
        path = self.cache_dir / "disaster_probs" / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
            data["date"] = date.fromisoformat(data["date"])
            return DisasterProbability(**data)
        return None

    def load_all_disaster_probs(self, region: str | None = None) -> list[DisasterProbability]:
        """Load all cached disaster probability results."""
        results = []
        for path in sorted((self.cache_dir / "disaster_probs").glob("*.json")):
            try:
                data = json.loads(path.read_text())
                data["date"] = date.fromisoformat(data["date"])
                dp = DisasterProbability(**data)
                if region is None or dp.region == region:
                    results.append(dp)
            except Exception as e:
                log.warning(f"Failed to load {path}: {e}")
        return results

    # --- Audit trail ---

    def log_audit(
        self,
        dt: date,
        region: str,
        step: str,
        status: str,
        details: str = "",
    ):
        """Log a pipeline step for audit trail.

        Parameters
        ----------
        dt : date
        region : str
        step : str
            E.g. "data_quality", "sabr_calibration", "gmm_convergence"
        status : str
            "ok", "warning", "error", "skipped"
        details : str
            Human-readable details.
        """
        record = {
            "date": str(dt),
            "region": region,
            "step": step,
            "status": status,
            "details": details,
        }
        self._audit_records.append(record)
        if status in ("warning", "error"):
            log.warning(f"AUDIT [{region} {dt}] {step}: {status} - {details}")

    def flush_audit_log(self):
        """Write accumulated audit records to CSV."""
        if not self._audit_records:
            return
        df = pd.DataFrame(self._audit_records)
        if self._audit_log_path.exists():
            existing = pd.read_csv(self._audit_log_path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(self._audit_log_path, index=False)
        self._audit_records.clear()
        log.info(f"Audit log written to {self._audit_log_path}")

    # --- Bulk export ---

    def export_results(
        self,
        output_path: Path | str,
        region: str | None = None,
        format: str = "csv",
    ):
        """Export all cached results to CSV or Excel.

        Parameters
        ----------
        output_path : Path
        region : str, optional
            Filter by region.
        format : str
            "csv" or "excel"
        """
        from inflation_disaster.analytics.disaster_probs import results_to_dataframe

        results = self.load_all_disaster_probs(region)
        if not results:
            log.warning("No cached results to export")
            return

        df = results_to_dataframe(results)
        output_path = Path(output_path)

        if format == "excel":
            df.to_excel(output_path, index=False, sheet_name="disaster_probs")
        else:
            df.to_csv(output_path, index=False)

        log.info(f"Exported {len(df)} results to {output_path}")
