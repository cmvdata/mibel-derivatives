"""Tolling agreement pricer for a CCGT (Castejon I reference plant).

Specification (CONTEXT.md section "Tolling agreement sobre Castejon I"):

A tolling agreement gives the offtaker the right -- but not the
obligation -- to dispatch a combined-cycle gas turbine (CCGT) against the
power, gas and carbon markets, in exchange for a fixed capacity fee paid
to the plant owner. Its value to the offtaker is the present value of the
optimised gross spark-spread margin the plant can earn, net of that fee.

Reference asset: Iberdrola's Castejon I CCGT in Navarra. Public reference
parameters (Iberdrola Environmental Declaration 2024; Aurecon/AEMO and
NREL technical benchmarks) live in :class:`AssetParameters`.

Dispatch optimiser (programacion dinamica)
------------------------------------------
The hour-by-hour dispatch is solved by **dynamic programming over the
plant's operating state**. The Bellman state tracks the commitment status
(off / on), the power level, and the time-in-state counters needed to
enforce the temporal constraints:

* technical band Pmin <= P <= Pmax,
* minimum up-time (TMO) and minimum down-time (TMA),
* ramp rate (MW/min) between consecutive on-hours,
* state-dependent start-up costs (hot / warm / cold, by hours off),
* heat-rate degradation at part load (worse GJ/MWh near Pmin).

The optimiser is solved **per realised price path with full information**
(perfect foresight). The reported tolling value is the Monte Carlo mean,
over composed power paths, of the discounted optimal gross margin minus
the fixed fee. This perfect-foresight dispatch value is the standard
deterministic-equivalent valuation and is an **upper bound** on the
non-anticipative (Longstaff-Schwartz recourse) value that CONTEXT.md
lists as the further extension; the gap is the value of foresight. The
pricer is kept path-agnostic (it consumes a ``(n_paths, n_hours)`` array
of power prices) for the same reasons the swing pricer is -- it can be
validated on synthetic paths and the path model stays out of the pricing
unit tests. The Monte Carlo composition of Pieza 1 (spot model) and
Pieza 2 (Schwartz-Smith forward) supplies those power paths; gas (MIBGAS
PVB) and carbon (EUA primary auction) enter as deterministic forward
curves from ``data/curated``.

Spark spread (per MWh, at power level P)::

    spark = power - (HR(P)/3.6) * gas - (HR(P)/3.6) * co2_intensity * eua

where HR(P) is the GJ/MWh heat rate (divided by 3.6 to convert to
MWh_th/MWh_e so it multiplies the EUR/MWh gas and carbon prices), and
``co2_intensity`` is the natural-gas combustion factor in tCO2/MWh_th.

References: Schwartz & Smith (2000) for the two-factor forward model that
anchors the power paths; the unit-commitment dynamic program follows the
standard single-unit formulation (e.g. Tseng & Barz 2002, "Short-term
generation asset valuation: a real options approach").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Combustion energy of one MWh expressed in GJ (unit bridge between the
#: GJ/MWh heat rate and the EUR/MWh gas and carbon prices).
GJ_PER_MWH: float = 3.6

#: Default Monte Carlo path counts (CONTEXT.md "Parametros numericos").
DEFAULT_N_PATHS_REPORT: int = 50_000
DEFAULT_N_PATHS_DEV: int = 10_000


@dataclass(frozen=True)
class AssetParameters:
    """Public reference parameters of the Castejon I CCGT.

    Every value is a public or benchmarked estimate (see module docstring
    and CONTEXT.md for sources); none is contractual. Heat rates are in
    GJ/MWh of electrical output and converted internally to MWh_th/MWh_e.

    Attributes
    ----------
    pmax_mw, pmin_mw
        Gross design power and technical minimum (MW).
    heat_rate_full_gj_per_mwh, heat_rate_min_gj_per_mwh
        Heat rate at full load and at the technical minimum. The minimum
        is worse (degraded part-load efficiency); the curve is linear in
        power between the two anchors.
    startup_cost_{hot,warm,cold}_eur_per_mw
        Start-up cost per MW of capacity, by how long the unit has been
        off (hot/warm/cold). Multiplied by ``capacity_mw`` per start.
    min_uptime_h, min_downtime_h
        Minimum continuous run hours (TMO) and minimum off hours between
        runs (TMA).
    ramp_mw_per_min
        Sustained ramp rate while running. At 8 MW/min an hourly step can
        traverse 480 MW > the full band, so the constraint is slack at
        hourly resolution by default; it binds at finer resolution or for
        a slower unit.
    co2_intensity_t_per_mwh_th
        Natural-gas combustion emission factor (tCO2 per MWh thermal).
    hot_max_off_h, warm_max_off_h
        Off-duration thresholds (hours) separating hot/warm/cold starts.
    n_power_levels
        Number of discrete power levels in [Pmin, Pmax] used by the DP.
    """

    pmax_mw: float = 386.10
    pmin_mw: float = 120.0
    heat_rate_full_gj_per_mwh: float = 6.55
    heat_rate_min_gj_per_mwh: float = 7.5
    startup_cost_hot_eur_per_mw: float = 50.0
    startup_cost_warm_eur_per_mw: float = 80.0
    startup_cost_cold_eur_per_mw: float = 110.0
    min_uptime_h: int = 4
    min_downtime_h: int = 2
    ramp_mw_per_min: float = 8.0
    co2_intensity_t_per_mwh_th: float = 0.2016
    hot_max_off_h: int = 8
    warm_max_off_h: int = 48
    n_power_levels: int = 7

    def __post_init__(self) -> None:
        if not (0.0 < self.pmin_mw < self.pmax_mw):
            raise ValueError("require 0 < pmin_mw < pmax_mw")
        if self.heat_rate_min_gj_per_mwh < self.heat_rate_full_gj_per_mwh:
            raise ValueError(
                "heat_rate_min must be >= heat_rate_full (part load is less efficient)"
            )
        if self.min_uptime_h < 1 or self.min_downtime_h < 1:
            raise ValueError("min_uptime_h and min_downtime_h must be >= 1")
        if self.ramp_mw_per_min <= 0.0:
            raise ValueError("ramp_mw_per_min must be > 0")
        if not (self.hot_max_off_h >= 1 and self.warm_max_off_h > self.hot_max_off_h):
            raise ValueError("require 1 <= hot_max_off_h < warm_max_off_h")
        if self.n_power_levels < 2:
            raise ValueError("n_power_levels must be >= 2")

    @property
    def power_levels(self) -> np.ndarray:
        """Discrete dispatchable power levels in [Pmin, Pmax] (MW)."""
        return np.linspace(self.pmin_mw, self.pmax_mw, self.n_power_levels)

    @property
    def ramp_mw_per_hour(self) -> float:
        """Sustained ramp budget over one hour (MW)."""
        return self.ramp_mw_per_min * 60.0


@dataclass(frozen=True)
class TollingAgreement:
    """Terms of a tolling agreement over a CCGT.

    Attributes
    ----------
    asset
        The plant's physical/operating parameters.
    capacity_mw
        Contracted (toll) capacity in MW. Drives the start-up cost
        (``EUR/MW * capacity``) and the fixed fee. Defaults to the asset's
        ``pmax_mw`` when left as ``None``.
    fixed_fee_eur_per_mw_year
        Fixed capacity payment the offtaker pays the owner, per MW of
        ``capacity_mw`` per year. Prorated to the horizon by hours.
    variable_om_eur_per_mwh
        Variable O&M deducted from each dispatched MWh. Default 0.
    """

    asset: AssetParameters = field(default_factory=AssetParameters)
    capacity_mw: float | None = None
    fixed_fee_eur_per_mw_year: float = 0.0
    variable_om_eur_per_mwh: float = 0.0

    def __post_init__(self) -> None:
        if self.fixed_fee_eur_per_mw_year < 0.0:
            raise ValueError("fixed_fee_eur_per_mw_year must be >= 0")
        if self.variable_om_eur_per_mwh < 0.0:
            raise ValueError("variable_om_eur_per_mwh must be >= 0")

    @property
    def effective_capacity_mw(self) -> float:
        return self.asset.pmax_mw if self.capacity_mw is None else self.capacity_mw


@dataclass(frozen=True)
class DispatchResult:
    """Output of :func:`optimise_dispatch` for a batch of price paths.

    All arrays have leading dimension ``n_paths``. ``value`` is already
    net of start-up cost and variable O&M (gross spark-spread margin of
    the optimal schedule, in present-value EUR)."""

    value: np.ndarray            # (n_paths,) PV of optimal gross margin
    power_schedule: np.ndarray   # (n_paths, n_hours) MW dispatched per hour
    n_starts: np.ndarray         # (n_paths,) number of start-ups
    startup_cost: np.ndarray     # (n_paths,) total PV start-up cost
    running_hours: np.ndarray    # (n_paths,) hours the unit ran


@dataclass(frozen=True)
class TollingResult:
    """Valuation output of :func:`price_tolling`.

    Attributes
    ----------
    option_value
        Monte Carlo mean PV (EUR) of the optimised gross dispatch margin
        -- the value of the dispatch optionality before paying the fee.
    std_error
        Monte Carlo standard error of ``option_value``.
    fixed_fee_pv
        Present value of the fixed capacity payments over the horizon.
    net_value
        ``option_value - fixed_fee_pv``: the value to the offtaker.
    per_path_value
        Per-path PV gross margin, shape ``(n_paths,)``.
    mean_capacity_factor
        Mean dispatched energy / (capacity * hours).
    mean_n_starts, mean_running_hours, mean_startup_cost
        Mean dispatch diagnostics across paths.
    n_paths, n_hours
        Echoed problem dimensions.
    """

    option_value: float
    std_error: float
    fixed_fee_pv: float
    net_value: float
    per_path_value: np.ndarray
    mean_capacity_factor: float
    mean_n_starts: float
    mean_running_hours: float
    mean_startup_cost: float
    n_paths: int
    n_hours: int


# ---- Physics: heat rate, spark spread, hourly margin -----------------------


def heat_rate_gj_per_mwh(asset: AssetParameters, power_mw: np.ndarray | float) -> np.ndarray:
    """Heat rate (GJ/MWh) at the given power, linear between the Pmin and
    Pmax anchors and clamped outside the technical band.

    Degraded part-load efficiency means ``HR(Pmin) > HR(Pmax)``."""
    p = np.asarray(power_mw, dtype=float)
    return np.interp(
        p,
        [asset.pmin_mw, asset.pmax_mw],
        [asset.heat_rate_min_gj_per_mwh, asset.heat_rate_full_gj_per_mwh],
    )


def spark_spread_per_mwh(
    asset: AssetParameters,
    power_price: np.ndarray | float,
    gas_price: np.ndarray | float,
    eua_price: np.ndarray | float,
    *,
    power_mw: float | None = None,
) -> np.ndarray:
    """Clean spark spread in EUR/MWh at power ``power_mw`` (default Pmax).

    ``power`` - HR_e * ``gas`` - HR_e * co2_intensity * ``eua``, with the
    electrical heat rate ``HR_e = HR(power_mw)/3.6`` in MWh_th/MWh_e. This
    is the per-MWh form of :func:`hourly_gross_margin`."""
    p_mw = asset.pmax_mw if power_mw is None else power_mw
    hr_e = float(heat_rate_gj_per_mwh(asset, p_mw)) / GJ_PER_MWH
    power = np.asarray(power_price, dtype=float)
    gas = np.asarray(gas_price, dtype=float)
    eua = np.asarray(eua_price, dtype=float)
    return power - hr_e * gas - hr_e * asset.co2_intensity_t_per_mwh_th * eua


def hourly_gross_margin(
    asset: AssetParameters,
    power_price: np.ndarray | float,
    gas_price: np.ndarray | float,
    eua_price: np.ndarray | float,
    power_mw: np.ndarray | float,
    *,
    variable_om_eur_per_mwh: float = 0.0,
) -> np.ndarray:
    """Gross margin in EUR for running one hour at ``power_mw``.

    ``power_mw * (power - vom)`` minus fuel and carbon cost, where the
    thermal input is ``power_mw * HR(power_mw)/3.6`` MWh_th."""
    p_mw = np.asarray(power_mw, dtype=float)
    thermal = p_mw * heat_rate_gj_per_mwh(asset, p_mw) / GJ_PER_MWH
    power = np.asarray(power_price, dtype=float)
    gas = np.asarray(gas_price, dtype=float)
    eua = np.asarray(eua_price, dtype=float)
    fuel_carbon = thermal * (gas + asset.co2_intensity_t_per_mwh_th * eua)
    return p_mw * (power - variable_om_eur_per_mwh) - fuel_carbon


# ---- Dynamic-programming dispatch optimiser --------------------------------


def _start_class_per_mw(asset: AssetParameters, off_dur: np.ndarray) -> np.ndarray:
    """Vectorised hot/warm/cold start-up cost (EUR/MW) by hours-off."""
    cost = np.full(off_dur.shape, asset.startup_cost_cold_eur_per_mw, dtype=float)
    cost[off_dur <= asset.warm_max_off_h] = asset.startup_cost_warm_eur_per_mw
    cost[off_dur <= asset.hot_max_off_h] = asset.startup_cost_hot_eur_per_mw
    return cost


@dataclass(frozen=True)
class _StateSpace:
    """Pre-computed Bellman state space and transition table.

    States: ``D`` off-states indexed by off-duration 1..D (the last
    saturates as 'cold'); ``K*U`` on-states indexed by (power level k,
    up-time u in 1..U, saturating). A start always lands on the technical
    minimum (level 0) and is exempt from the operational ramp (the
    synchronisation ramp is priced into the start-up cost); the
    operational ramp constrains only on->on power moves; a stop is allowed
    only once min up-time is met and is ramp-exempt."""

    n_states: int
    off_count: int                 # D
    n_levels: int                  # K
    uptime_cap: int                # U
    is_off: np.ndarray             # (S,) bool
    off_dur_of: np.ndarray         # (S,) off-duration (0 for on-states)
    power_of: np.ndarray           # (S,) MW (0 if off)
    level_of: np.ndarray           # (S,) level index (-1 if off)
    cand_to: np.ndarray            # (S, Cmax) int, -1 padding
    cand_start_per_mw: np.ndarray  # (S, Cmax) float (0 if not a start)
    cand_mask: np.ndarray          # (S, Cmax) bool


def _build_state_space(asset: AssetParameters, ramp_mw_per_hour: float) -> _StateSpace:
    levels = asset.power_levels
    K = len(levels)
    U = int(asset.min_uptime_h)
    D = int(asset.warm_max_off_h) + 1  # off-durations 1..warm_max, +1 = cold saturate
    min_down = int(asset.min_downtime_h)

    def on_index(k: int, u: int) -> int:
        return D + k * U + (u - 1)

    n_states = D + K * U
    is_off = np.zeros(n_states, dtype=bool)
    off_dur_of = np.zeros(n_states, dtype=int)
    power_of = np.zeros(n_states, dtype=float)
    level_of = np.full(n_states, -1, dtype=int)
    is_off[:D] = True
    off_dur_of[:D] = np.arange(1, D + 1)
    for k in range(K):
        for u in range(1, U + 1):
            s = on_index(k, u)
            power_of[s] = levels[k]
            level_of[s] = k

    off_dur_arr = np.arange(1, D + 1)
    start_cost_by_dur = _start_class_per_mw(asset, off_dur_arr)  # (D,)

    cand: list[list[tuple[int, float]]] = [[] for _ in range(n_states)]
    # A start synchronises to the technical minimum Pmin; that synchronisation
    # ramp is priced into the start-up cost and is exempt from the operational
    # ramp limit, which constrains only on->on power moves below.
    for d in range(1, D + 1):  # off-state with off_dur = d
        s = d - 1
        stay = min(d + 1, D) - 1  # off_dur -> min(d+1, D)
        cand[s].append((stay, 0.0))
        if d >= min_down:
            cand[s].append((on_index(0, 1), float(start_cost_by_dur[d - 1])))

    for k in range(K):
        for u in range(1, U + 1):
            s = on_index(k, u)
            u_next = min(u + 1, U)
            for k2 in range(K):
                if abs(levels[k2] - levels[k]) <= ramp_mw_per_hour + 1e-9:
                    cand[s].append((on_index(k2, u_next), 0.0))
            if u >= asset.min_uptime_h:  # min up-time met -> may stop
                cand[s].append((0, 0.0))  # off_dur = 1

    cmax = max(len(c) for c in cand)
    cand_to = np.full((n_states, cmax), -1, dtype=int)
    cand_start = np.zeros((n_states, cmax), dtype=float)
    cand_mask = np.zeros((n_states, cmax), dtype=bool)
    for s, cs in enumerate(cand):
        for j, (to, sp) in enumerate(cs):
            cand_to[s, j] = to
            cand_start[s, j] = sp
            cand_mask[s, j] = True

    return _StateSpace(
        n_states=n_states,
        off_count=D,
        n_levels=K,
        uptime_cap=U,
        is_off=is_off,
        off_dur_of=off_dur_of,
        power_of=power_of,
        level_of=level_of,
        cand_to=cand_to,
        cand_start_per_mw=cand_start,
        cand_mask=cand_mask,
    )


def optimise_dispatch(
    power_paths: np.ndarray,
    gas_curve: np.ndarray,
    eua_curve: np.ndarray,
    asset: AssetParameters,
    *,
    discount_factors: np.ndarray | None = None,
    capacity_mw: float | None = None,
    variable_om_eur_per_mwh: float = 0.0,
    ramp_mw_per_min: float | None = None,
    initial_off_hours: int | None = None,
) -> DispatchResult:
    """Optimise the hour-by-hour dispatch of the unit for each price path.

    Solves the unit-commitment dynamic program backward in time with
    perfect foresight per path, then replays the optimal policy forward to
    recover the schedule and diagnostics.

    Parameters
    ----------
    power_paths
        Power prices in EUR/MWh, shape ``(n_paths, n_hours)`` (a 1D array
        is treated as a single path).
    gas_curve, eua_curve
        Deterministic gas (MIBGAS PVB, EUR/MWh) and carbon (EUA, EUR/t)
        forward curves, shape ``(n_hours,)``.
    asset
        Plant parameters.
    discount_factors
        Per-hour PV factors from the valuation date, shape ``(n_hours,)``.
        Default all ones (undiscounted).
    capacity_mw
        Capacity driving start-up cost (``EUR/MW * capacity``). Default
        ``asset.pmax_mw``.
    variable_om_eur_per_mwh
        Variable O&M deducted per dispatched MWh.
    ramp_mw_per_min
        Override the asset ramp rate (used to exercise the ramp limit).
    initial_off_hours
        Hours the unit has been off entering the horizon (sets the first
        start's hot/warm/cold class). Default: cold (``warm_max + 1``).

    Returns
    -------
    DispatchResult
    """
    paths = np.asarray(power_paths, dtype=float)
    if paths.ndim == 1:
        paths = paths[None, :]
    if paths.ndim != 2:
        raise ValueError(f"power_paths must be 1D or 2D, got shape {paths.shape}")
    n_paths, n_hours = paths.shape
    if n_paths < 1 or n_hours < 1:
        raise ValueError("power_paths must have at least one path and one hour")
    if not np.isfinite(paths).all():
        raise ValueError("power_paths contains non-finite values")

    gas = np.asarray(gas_curve, dtype=float)
    eua = np.asarray(eua_curve, dtype=float)
    if gas.shape != (n_hours,) or eua.shape != (n_hours,):
        raise ValueError(
            f"gas_curve and eua_curve must have shape ({n_hours},); "
            f"got {gas.shape} and {eua.shape}"
        )
    if not (np.isfinite(gas).all() and np.isfinite(eua).all()):
        raise ValueError("gas_curve / eua_curve contain non-finite values")

    if discount_factors is None:
        df = np.ones(n_hours, dtype=float)
    else:
        df = np.asarray(discount_factors, dtype=float)
        if df.shape != (n_hours,):
            raise ValueError(f"discount_factors must have shape ({n_hours},)")
        if not np.isfinite(df).all() or (df <= 0.0).any():
            raise ValueError("discount_factors must be finite and strictly positive")

    cap = asset.pmax_mw if capacity_mw is None else capacity_mw
    ramp_per_hour = (
        asset.ramp_mw_per_hour if ramp_mw_per_min is None else ramp_mw_per_min * 60.0
    )
    ss = _build_state_space(asset, ramp_per_hour)
    S = ss.n_states

    fuel_cost = gas + asset.co2_intensity_t_per_mwh_th * eua  # EUR / MWh_th, (n_hours,)
    levels = asset.power_levels
    thermal_by_level = levels * heat_rate_gj_per_mwh(asset, levels) / GJ_PER_MWH  # (K,)

    level_of = ss.level_of
    power_of = ss.power_of
    on_mask = level_of >= 0
    on_levels = level_of[on_mask]
    on_power = power_of[on_mask]
    on_thermal = thermal_by_level[on_levels]

    cand_to = ss.cand_to
    cand_mask = ss.cand_mask
    cand_start = ss.cand_start_per_mw

    # Backward DP: ``value`` holds V_{t+1}; ``choice[t]`` the chosen to-state.
    value = np.zeros((S, n_paths), dtype=float)
    choice = np.empty((n_hours, S, n_paths), dtype=np.int16)

    for t in range(n_hours - 1, -1, -1):
        power_t = paths[:, t]
        fuel_t = fuel_cost[t]
        df_t = df[t]
        df_next = df[t + 1] if t + 1 < n_hours else df[t]

        margin_state = np.zeros((S, n_paths), dtype=float)
        margin_state[on_mask] = (
            on_power[:, None] * (power_t[None, :] - variable_om_eur_per_mwh)
            - on_thermal[:, None] * fuel_t
        )

        new_value = np.empty((S, n_paths), dtype=float)
        for s in range(S):
            best_val: np.ndarray | None = None
            best_to = np.zeros(n_paths, dtype=np.int16)
            for j in range(cand_to.shape[1]):
                if not cand_mask[s, j]:
                    continue
                to = int(cand_to[s, j])
                cont = value[to] - df_next * cand_start[s, j] * cap
                if best_val is None:
                    best_val = cont.copy()
                    best_to[:] = to
                else:
                    take = cont > best_val
                    best_val = np.where(take, cont, best_val)
                    best_to = np.where(take, np.int16(to), best_to)
            assert best_val is not None
            new_value[s] = df_t * margin_state[s] + best_val
            choice[t, s] = best_to
        value = new_value

    # Initial transition from the pre-horizon off-state into hour 0.
    init_off = asset.warm_max_off_h + 1 if initial_off_hours is None else initial_off_hours
    init_off = max(1, int(init_off))
    init_state = min(init_off, ss.off_count) - 1
    best_val0: np.ndarray | None = None
    s0 = np.zeros(n_paths, dtype=np.int16)
    for j in range(cand_to.shape[1]):
        if not cand_mask[init_state, j]:
            continue
        to = int(cand_to[init_state, j])
        cont = value[to] - df[0] * cand_start[init_state, j] * cap
        if best_val0 is None:
            best_val0 = cont.copy()
            s0[:] = to
        else:
            take = cont > best_val0
            best_val0 = np.where(take, cont, best_val0)
            s0 = np.where(take, np.int16(to), s0)
    assert best_val0 is not None
    value_per_path = best_val0

    # Forward replay: recover schedule and diagnostics, with the exact
    # start class taken from the off-duration entering each start.
    rows = np.arange(n_paths)
    is_off = ss.is_off
    off_dur_of = ss.off_dur_of
    power_schedule = np.zeros((n_paths, n_hours), dtype=float)
    state = s0.astype(int)
    prev_off = np.ones(n_paths, dtype=bool)
    prev_off_dur = np.full(n_paths, init_off, dtype=int)
    n_starts = np.zeros(n_paths, dtype=int)
    startup_cost = np.zeros(n_paths, dtype=float)
    running_hours = np.zeros(n_paths, dtype=int)

    for t in range(n_hours):
        cur_off = is_off[state]
        started = prev_off & (~cur_off)
        if started.any():
            per_mw = _start_class_per_mw(asset, prev_off_dur)
            startup_cost[started] += df[t] * per_mw[started] * cap
            n_starts[started] += 1
        power_schedule[:, t] = power_of[state]
        running_hours += (~cur_off).astype(int)
        prev_off = cur_off
        prev_off_dur = np.where(cur_off, off_dur_of[state], prev_off_dur)
        if t + 1 < n_hours:
            state = choice[t, state, rows].astype(int)

    return DispatchResult(
        value=value_per_path,
        power_schedule=power_schedule,
        n_starts=n_starts,
        startup_cost=startup_cost,
        running_hours=running_hours,
    )


def price_tolling(
    agreement: TollingAgreement,
    power_paths: np.ndarray,
    gas_curve: np.ndarray,
    eua_curve: np.ndarray,
    *,
    discount_factors: np.ndarray | None = None,
    hours_per_year: float = 8760.0,
    chunk_size: int = 200,
) -> TollingResult:
    """Value a tolling agreement by Monte Carlo over optimised dispatch.

    Runs :func:`optimise_dispatch` over the composed power paths (gas and
    carbon fixed at their forward curves), averages the discounted optimal
    gross margin, and nets the present value of the fixed capacity fee.

    Parameters
    ----------
    agreement
        Contract terms (asset, capacity, fixed fee, variable O&M).
    power_paths
        Power-price paths, shape ``(n_paths, n_hours)``.
    gas_curve, eua_curve
        Deterministic gas (EUR/MWh) and carbon (EUR/t) curves, ``(n_hours,)``.
    discount_factors
        Per-hour PV factors, ``(n_hours,)``. Default all ones.
    hours_per_year
        Used to prorate the annual fixed fee to the horizon.
    chunk_size
        Paths processed per DP batch (bounds the policy-tensor memory).

    Returns
    -------
    TollingResult
    """
    paths = np.asarray(power_paths, dtype=float)
    if paths.ndim != 2:
        raise ValueError("power_paths must be 2D (n_paths, n_hours)")
    n_paths, n_hours = paths.shape
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    asset = agreement.asset
    cap = agreement.effective_capacity_mw

    if discount_factors is None:
        df = np.ones(n_hours, dtype=float)
    else:
        df = np.asarray(discount_factors, dtype=float)

    per_path = np.empty(n_paths, dtype=float)
    energy = np.empty(n_paths, dtype=float)
    n_starts = np.empty(n_paths, dtype=float)
    running = np.empty(n_paths, dtype=float)
    startup = np.empty(n_paths, dtype=float)

    for lo in range(0, n_paths, chunk_size):
        hi = min(lo + chunk_size, n_paths)
        res = optimise_dispatch(
            paths[lo:hi],
            gas_curve,
            eua_curve,
            asset,
            discount_factors=df,
            capacity_mw=cap,
            variable_om_eur_per_mwh=agreement.variable_om_eur_per_mwh,
        )
        per_path[lo:hi] = res.value
        energy[lo:hi] = res.power_schedule.sum(axis=1)
        n_starts[lo:hi] = res.n_starts
        running[lo:hi] = res.running_hours
        startup[lo:hi] = res.startup_cost

    option_value = float(per_path.mean())
    std_error = float(per_path.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else 0.0

    fee_total = agreement.fixed_fee_eur_per_mw_year * cap * (n_hours / hours_per_year)
    fixed_fee_pv = fee_total * float(df.mean())

    max_energy = cap * n_hours
    mean_capacity_factor = float(energy.mean() / max_energy) if max_energy > 0 else 0.0

    return TollingResult(
        option_value=option_value,
        std_error=std_error,
        fixed_fee_pv=fixed_fee_pv,
        net_value=option_value - fixed_fee_pv,
        per_path_value=per_path,
        mean_capacity_factor=mean_capacity_factor,
        mean_n_starts=float(n_starts.mean()),
        mean_running_hours=float(running.mean()),
        mean_startup_cost=float(startup.mean()),
        n_paths=n_paths,
        n_hours=n_hours,
    )
