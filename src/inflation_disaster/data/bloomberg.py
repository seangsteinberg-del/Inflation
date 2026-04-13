"""Bloomberg data fetcher for inflation options, swaps, and rates.

Fetches zero-coupon inflation caps/floors, year-on-year caps/floors,
and inflation swap rates via the Bloomberg API (blpapi).

Ticker conventions:
  US ZC inflation caps:  USCPI{maturity}F {strike} Curncy  (floors similar)
  EZ ZC inflation caps:  EUHICPX{maturity}F {strike} Curncy
  US inflation swaps:    USSWIT{maturity} Curncy
  EZ inflation swaps:    EUSWI{maturity} Curncy

Note: Exact Bloomberg ticker construction may vary by terminal configuration.
The fetch methods use BSRCH and overrides as needed.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from inflation_disaster.config import settings
from inflation_disaster.data.schemas import OptionSurface

log = logging.getLogger("inflation_disaster.data.bloomberg")


# ---------------------------------------------------------------------------
# Bloomberg ticker maps
# ---------------------------------------------------------------------------

# Zero-coupon cap/floor ticker templates
# These follow the pattern used by Bloomberg for inflation vol surfaces
_ZC_TICKER_MAP = {
    "US": {
        "cap": "USCP{maturity:02d}{strike_code} Curncy",
        "floor": "USFP{maturity:02d}{strike_code} Curncy",
        "swap": "USSWIT{maturity} Curncy",
    },
    "EZ": {
        "cap": "EUCP{maturity:02d}{strike_code} Curncy",
        "floor": "EUFP{maturity:02d}{strike_code} Curncy",
        "swap": "EUSWI{maturity} Curncy",
    },
}

# Year-on-year cap/floor tickers
_YOY_TICKER_MAP = {
    "US": {
        "cap": "USCPYY{maturity:02d}{strike_code} Curncy",
        "floor": "USFPYY{maturity:02d}{strike_code} Curncy",
    },
    "EZ": {
        "cap": "EUCPYY{maturity:02d}{strike_code} Curncy",
        "floor": "EUFPYY{maturity:02d}{strike_code} Curncy",
    },
}


def _strike_to_code(strike: float) -> str:
    """Convert strike price to Bloomberg ticker code.

    Examples: -2.0 -> 'N200', -0.5 -> 'N050', 0.0 -> '000',
              1.5 -> '150', 4.0 -> '400', 6.0 -> '600'
    """
    if strike < 0:
        return f"N{abs(strike * 100):03.0f}"
    return f"{strike * 100:03.0f}"


class BloombergFetcher:
    """Fetches inflation options data from Bloomberg.

    Uses blpapi for historical data requests. All fetched data is cached
    locally as parquet files to avoid repeated Bloomberg queries.

    Parameters
    ----------
    cache_dir : Path
        Directory for parquet cache files.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or settings.raw_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = None

    def _get_session(self):
        """Lazily initialize Bloomberg session."""
        if self._session is not None:
            return self._session

        import blpapi

        session_options = blpapi.SessionOptions()
        session_options.setServerHost("localhost")
        session_options.setServerPort(8194)

        session = blpapi.Session(session_options)
        if not session.start():
            raise ConnectionError("Failed to start Bloomberg session")
        if not session.openService("//blp/refdata"):
            raise ConnectionError("Failed to open //blp/refdata service")

        self._session = session
        log.info("Bloomberg session started")
        return session

    def _bdh(
        self,
        securities: list[str],
        fields: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Bloomberg historical data request (BDH).

        Returns DataFrame with MultiIndex columns (security, field).
        """
        import blpapi

        session = self._get_session()
        refdata = session.getService("//blp/refdata")
        request = refdata.createRequest("HistoricalDataRequest")

        for sec in securities:
            request.getElement("securities").appendValue(sec)
        for fld in fields:
            request.getElement("fields").appendValue(fld)

        request.set("startDate", start_date.strftime("%Y%m%d"))
        request.set("endDate", end_date.strftime("%Y%m%d"))
        request.set("periodicitySelection", "DAILY")

        session.sendRequest(request)

        results = []
        timeout_count = 0
        max_timeouts = 10
        while True:
            event = session.nextEvent(5000)

            if event.eventType() == blpapi.Event.TIMEOUT:
                timeout_count += 1
                if timeout_count >= max_timeouts:
                    log.warning(
                        f"Bloomberg request timed out after {max_timeouts} attempts"
                    )
                    break
                continue

            timeout_count = 0  # reset on any non-timeout event

            for msg in event:
                if msg.hasElement("securityData"):
                    sec_data = msg.getElement("securityData")
                    security = sec_data.getElementAsString("security")
                    field_data = sec_data.getElement("fieldData")

                    for i in range(field_data.numValues()):
                        row = field_data.getValueAsElement(i)
                        d = row.getElementAsDatetime("date")
                        row_date = date(d.year, d.month, d.day)
                        for fld in fields:
                            if row.hasElement(fld):
                                val = row.getElementAsFloat(fld)
                                results.append({
                                    "date": row_date,
                                    "security": security,
                                    "field": fld,
                                    "value": val,
                                })

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        return df.pivot_table(
            index="date", columns=["security", "field"], values="value"
        )

    def fetch_zc_caps_floors(
        self,
        region: Literal["US", "EZ"],
        start: date,
        end: date,
        maturities: list[int] | None = None,
        strikes: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Fetch zero-coupon inflation cap and floor prices.

        Parameters
        ----------
        region : "US" or "EZ"
        start, end : date range
        maturities : list of maturities in years, default [5, 10]
        strikes : array of strike prices in percent, default from config

        Returns
        -------
        DataFrame with columns: date, maturity, strike, cap_price, floor_price
        """
        if maturities is None:
            maturities = list(settings.maturities)
        if strikes is None:
            strikes = settings.strikes

        cache_key = f"zc_{region}_{start}_{end}.parquet"
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            log.info(f"Loading cached ZC data from {cache_path}")
            return pd.read_parquet(cache_path)

        ticker_map = _ZC_TICKER_MAP[region]
        all_tickers = []
        ticker_meta = []

        for mat in maturities:
            for strike in strikes:
                strike_code = _strike_to_code(strike)

                # Determine if this strike is a cap or floor
                # Caps: strike >= swap rate (roughly); floors: strike < swap rate
                # For full coverage, fetch both where available
                cap_ticker = ticker_map["cap"].format(
                    maturity=mat, strike_code=strike_code
                )
                floor_ticker = ticker_map["floor"].format(
                    maturity=mat, strike_code=strike_code
                )

                all_tickers.extend([cap_ticker, floor_ticker])
                ticker_meta.extend([
                    {"maturity": mat, "strike": strike, "type": "cap", "ticker": cap_ticker},
                    {"maturity": mat, "strike": strike, "type": "floor", "ticker": floor_ticker},
                ])

        log.info(
            f"Fetching {len(all_tickers)} ZC tickers for {region} "
            f"({len(maturities)} mats x {len(strikes)} strikes x 2 types)"
        )

        # Fetch in batches to avoid Bloomberg limits
        batch_size = 50
        all_data = []
        for i in range(0, len(all_tickers), batch_size):
            batch = all_tickers[i : i + batch_size]
            try:
                df = self._bdh(batch, ["PX_LAST"], start, end)
                all_data.append(df)
            except Exception as e:
                log.warning(f"Batch {i // batch_size} failed: {e}")

        if not all_data:
            log.error("No Bloomberg data retrieved")
            return pd.DataFrame()

        raw = pd.concat(all_data, axis=1)

        # Restructure into tidy format
        records = []
        for meta in ticker_meta:
            ticker = meta["ticker"]
            col = (ticker, "PX_LAST")
            if col not in raw.columns:
                continue
            series = raw[col].dropna()
            for dt, price in series.items():
                records.append({
                    "date": dt,
                    "maturity": meta["maturity"],
                    "strike": meta["strike"],
                    f"{meta['type']}_price": price,
                })

        result = pd.DataFrame(records)
        if not result.empty:
            # Merge caps and floors on same date/maturity/strike
            caps = result[result["cap_price"].notna()][
                ["date", "maturity", "strike", "cap_price"]
            ]
            floors = result[result["floor_price"].notna()][
                ["date", "maturity", "strike", "floor_price"]
            ]
            result = pd.merge(
                caps, floors, on=["date", "maturity", "strike"], how="outer"
            )
            result.to_parquet(cache_path)
            log.info(f"Cached ZC data to {cache_path}: {len(result)} rows")

        return result

    def fetch_yoy_caps_floors(
        self,
        region: Literal["US", "EZ"],
        start: date,
        end: date,
        maturities: list[int] | None = None,
        strikes: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Fetch year-on-year inflation cap and floor prices.

        Used for forward-starting option distributions (years 5-10).
        """
        if maturities is None:
            maturities = list(settings.yoy_maturities)
        if strikes is None:
            strikes = settings.strikes

        cache_key = f"yoy_{region}_{start}_{end}.parquet"
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            log.info(f"Loading cached YOY data from {cache_path}")
            return pd.read_parquet(cache_path)

        ticker_map = _YOY_TICKER_MAP[region]
        all_tickers = []
        ticker_meta = []

        for mat in maturities:
            for strike in strikes:
                strike_code = _strike_to_code(strike)
                cap_ticker = ticker_map["cap"].format(
                    maturity=mat, strike_code=strike_code
                )
                floor_ticker = ticker_map["floor"].format(
                    maturity=mat, strike_code=strike_code
                )
                all_tickers.extend([cap_ticker, floor_ticker])
                ticker_meta.extend([
                    {"maturity": mat, "strike": strike, "type": "cap", "ticker": cap_ticker},
                    {"maturity": mat, "strike": strike, "type": "floor", "ticker": floor_ticker},
                ])

        log.info(f"Fetching {len(all_tickers)} YOY tickers for {region}")

        batch_size = 50
        all_data = []
        for i in range(0, len(all_tickers), batch_size):
            batch = all_tickers[i : i + batch_size]
            try:
                df = self._bdh(batch, ["PX_LAST"], start, end)
                all_data.append(df)
            except Exception as e:
                log.warning(f"YOY batch {i // batch_size} failed: {e}")

        if not all_data:
            return pd.DataFrame()

        raw = pd.concat(all_data, axis=1)
        records = []
        for meta in ticker_meta:
            col = (meta["ticker"], "PX_LAST")
            if col not in raw.columns:
                continue
            series = raw[col].dropna()
            for dt, price in series.items():
                records.append({
                    "date": dt,
                    "maturity": meta["maturity"],
                    "strike": meta["strike"],
                    f"{meta['type']}_price": price,
                })

        result = pd.DataFrame(records)
        if not result.empty:
            caps = result[result["cap_price"].notna()][
                ["date", "maturity", "strike", "cap_price"]
            ]
            floors = result[result["floor_price"].notna()][
                ["date", "maturity", "strike", "floor_price"]
            ]
            result = pd.merge(
                caps, floors, on=["date", "maturity", "strike"], how="outer"
            )
            result.to_parquet(cache_path)
            log.info(f"Cached YOY data to {cache_path}: {len(result)} rows")

        return result

    def fetch_inflation_swaps(
        self,
        region: Literal["US", "EZ"],
        start: date,
        end: date,
        maturities: list[int] | None = None,
    ) -> pd.DataFrame:
        """Fetch inflation swap rates (breakeven inflation).

        Returns DataFrame with columns: date, maturity, swap_rate
        """
        if maturities is None:
            maturities = [1, 2, 3, 5, 7, 10]

        cache_key = f"swaps_{region}_{start}_{end}.parquet"
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            log.info(f"Loading cached swap data from {cache_path}")
            return pd.read_parquet(cache_path)

        ticker_map = _ZC_TICKER_MAP[region]
        tickers = []
        mat_map = {}
        for mat in maturities:
            ticker = ticker_map["swap"].format(maturity=mat)
            tickers.append(ticker)
            mat_map[ticker] = mat

        log.info(f"Fetching {len(tickers)} swap tickers for {region}")
        raw = self._bdh(tickers, ["PX_LAST"], start, end)

        records = []
        for ticker, mat in mat_map.items():
            col = (ticker, "PX_LAST")
            if col not in raw.columns:
                continue
            series = raw[col].dropna()
            for dt, rate in series.items():
                records.append({
                    "date": dt,
                    "maturity": mat,
                    "swap_rate": rate / 100.0,  # convert from percent
                })

        result = pd.DataFrame(records)
        if not result.empty:
            result.to_parquet(cache_path)
            log.info(f"Cached swap data to {cache_path}: {len(result)} rows")

        return result

    def fetch_nominal_rates(
        self,
        region: Literal["US", "EZ"],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch nominal government bond yields for discounting.

        US: Treasury yields; EZ: German Bund yields.
        """
        tickers = {
            "US": {"5y": "USGG5YR Index", "10y": "USGG10YR Index"},
            "EZ": {"5y": "GDBR5 Index", "10y": "GDBR10 Index"},
        }

        region_tickers = tickers[region]
        all_tickers = list(region_tickers.values())

        raw = self._bdh(all_tickers, ["PX_LAST"], start, end)

        records = []
        for label, ticker in region_tickers.items():
            mat = int(label.replace("y", ""))
            col = (ticker, "PX_LAST")
            if col not in raw.columns:
                continue
            series = raw[col].dropna()
            for dt, rate in series.items():
                records.append({
                    "date": dt,
                    "maturity": mat,
                    "nominal_rate": rate / 100.0,
                })

        return pd.DataFrame(records)

    def build_option_surface(
        self,
        zc_data: pd.DataFrame,
        swap_data: pd.DataFrame,
        target_date: date,
        region: Literal["US", "EZ"],
        maturity: int,
    ) -> OptionSurface | None:
        """Construct an OptionSurface for a specific date and maturity.

        Selects the closest available date within 5 business days.
        """
        # Find closest date
        available_dates = zc_data["date"].unique()
        date_diffs = np.abs(
            pd.to_datetime(available_dates) - pd.Timestamp(target_date)
        )
        min_idx = date_diffs.argmin()
        if date_diffs[min_idx].days > 7:
            log.warning(
                f"No data within 7 days of {target_date} for {region} {maturity}y"
            )
            return None

        actual_date_ts = pd.Timestamp(available_dates[min_idx])
        actual_date = actual_date_ts.date()

        # Filter to this date and maturity
        # Use pd.Timestamp for comparison to avoid datetime.date vs Timestamp mismatch
        mask = (zc_data["date"] == actual_date_ts) & (zc_data["maturity"] == maturity)
        subset = zc_data[mask].sort_values("strike")

        if subset.empty:
            return None

        # Get swap rate
        swap_mask = (swap_data["date"] == actual_date_ts) & (
            swap_data["maturity"] == maturity
        )
        swap_subset = swap_data[swap_mask]
        swap_rate = swap_subset["swap_rate"].iloc[0] if not swap_subset.empty else 0.02

        strikes = subset["strike"].values
        cap_prices = subset["cap_price"].fillna(0.0).values
        floor_prices = subset["floor_price"].fillna(0.0).values

        return OptionSurface(
            date=actual_date,
            region=region,
            maturity=maturity,
            strikes=strikes,
            cap_prices=cap_prices,
            floor_prices=floor_prices,
            swap_rate=swap_rate,
        )

    def close(self):
        """Close Bloomberg session."""
        if self._session is not None:
            self._session.stop()
            self._session = None
            log.info("Bloomberg session closed")
