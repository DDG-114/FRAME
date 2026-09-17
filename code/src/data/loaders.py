"""Dataset registry used by training, cache-building, and evaluation entry points."""
from __future__ import annotations

from collections.abc import Callable

from src.data.load_st_evcdp import load_st_evcdp
from src.data.load_urbanev import load_urbanev
from src.data.load_gs_market import load_gs_market, load_gs_market_2025
from src.data.load_gs_price import load_gs_price, load_gs_price_2025
from src.data.load_gefc2014 import load_gefc2014_load, load_gefc2014_load_task15
from src.data.load_gb_gsp_netload import load_gb_gsp_netload
from src.data.load_jiangsu_load import load_jiangsu_load
from src.data.load_jiangsu_price import load_jiangsu_price
from src.data.load_mmsp_fusionsf import load_mmsp_fusionsf
from src.data.load_dkasc import load_dkasc_pv, load_dkasc_total_pv
from src.data.load_renewable_generation import load_renewable_solar, load_renewable_wind
from src.data.load_sdwpf import load_sdwpf
from src.data.load_wotai_evcdp import load_wotai_evcdp
from src.data.load_wotai_actual_load import load_wotai_actual_load
from src.data.load_aemo_unified import (
    load_aemo_load,
    load_aemo_net_load,
    load_aemo_pv,
    load_aemo_wind,
)
from src.data.load_perform_ercot import (
    load_perform_load,
    load_perform_net_load,
    load_perform_pv,
    load_perform_wind,
)

DATASET_LOADERS: dict[str, Callable[[], dict]] = {
    "st_evcdp": load_st_evcdp,
    "urbanev": load_urbanev,
    "wotai_evcdp": load_wotai_evcdp,
    "wotai_actual_load": load_wotai_actual_load,
    "renewable_solar": load_renewable_solar,
    "renewable_wind": load_renewable_wind,
    "gs_market": load_gs_market,
    "gs_market_2025": load_gs_market_2025,
    "gs_price": load_gs_price,
    "gs_price_2025": load_gs_price_2025,
    "gefc2014_load": load_gefc2014_load,
    "gefc2014_load_task15": load_gefc2014_load_task15,
    "gb_gsp_netload": load_gb_gsp_netload,
    "jiangsu_load": load_jiangsu_load,
    "jiangsu_price": load_jiangsu_price,
    "sdwpf": load_sdwpf,
    "mmsp_fusionsf": load_mmsp_fusionsf,
    "dkasc_pv": load_dkasc_pv,
    "dkasc_total_pv": load_dkasc_total_pv,
    "aemo_load": load_aemo_load,
    "aemo_wind": load_aemo_wind,
    "aemo_pv": load_aemo_pv,
    "aemo_net_load": load_aemo_net_load,
    "perform_load": load_perform_load,
    "perform_wind": load_perform_wind,
    "perform_pv": load_perform_pv,
    "perform_net_load": load_perform_net_load,
}

DATASET_CHOICES = tuple(DATASET_LOADERS.keys())


def load_dataset(dataset: str) -> dict:
    try:
        return DATASET_LOADERS[dataset]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset={dataset!r}; expected one of {sorted(DATASET_LOADERS)}"
        ) from exc
