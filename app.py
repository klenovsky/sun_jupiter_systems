#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit app: Solar System next to Jupiter and the Galilean moons.

Left panel:
    One Newtonian N-body integration shown in barycentric Solar-System coordinates.
Right panel:
    The same integration transformed to Jupiter-centered coordinates, with
    Io, Europa, Ganymede and Callisto visible around Jupiter.

Run locally:
    streamlit run app.py

Units:
    length = AU, time = Julian year, mass = solar mass.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
import tempfile
from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# =============================================================================
# Constants
# =============================================================================

G_MODEL = 4.0 * math.pi * math.pi  # AU^3 / (M_sun yr^2)
DAYS_PER_YEAR = 365.25
KM_PER_AU = 149_597_870.7
MSUN_KG = 1.98847e30
JUPITER_RADIUS_KM = 69_911.0
JUPITER_RADIUS_AU = JUPITER_RADIUS_KM / KM_PER_AU
SOFTENING_AU = 1.0e-8
PLOT_UIREVISION = "solar-jupiter-galilean-moons-v1"

GM_TO_MSUN = 1.0e9 / 6.67430e-11 / MSUN_KG  # convert km^3/s^2 to M_sun

# =============================================================================
# Data
# =============================================================================

@dataclass(frozen=True)
class Body:
    name: str
    cs_name: str
    mass_msun: float
    radius_km: float
    color: str

@dataclass(frozen=True)
class PlanetOrbit:
    body: Body
    a_au: float
    inc_deg: float
    phase_deg: float

@dataclass(frozen=True)
class MoonOrbit:
    body: Body
    a_km: float
    period_days: float
    inc_deg: float
    phase_deg: float

SUN = Body("Sun", "Slunce", 1.0, 696_340.0, "gold")
PLANETS: tuple[PlanetOrbit, ...] = (
    PlanetOrbit(Body("Mercury", "Merkur", 0.330103e24 / MSUN_KG, 2439.4, "dimgray"), 0.38709927, 7.00497902, 252.25032350),
    PlanetOrbit(Body("Venus", "Venuše", 4.86731e24 / MSUN_KG, 6051.8, "orange"), 0.72333566, 3.39467605, 181.97909950),
    PlanetOrbit(Body("Earth", "Země", 5.97217e24 / MSUN_KG, 6371.0084, "royalblue"), 1.00000261, -0.00001531, 100.46457166),
    PlanetOrbit(Body("Mars", "Mars", 0.641691e24 / MSUN_KG, 3389.50, "red"), 1.52371034, 1.84969142, -4.55343205),
    PlanetOrbit(Body("Jupiter", "Jupiter", 1898.125e24 / MSUN_KG, JUPITER_RADIUS_KM, "sienna"), 5.20288700, 1.30439695, 34.39644051),
    PlanetOrbit(Body("Saturn", "Saturn", 568.317e24 / MSUN_KG, 58232.0, "peru"), 9.53667594, 2.48599187, 49.95424423),
    PlanetOrbit(Body("Uranus", "Uran", 86.8099e24 / MSUN_KG, 25362.0, "cyan"), 19.18916464, 0.77263783, 313.23810451),
    PlanetOrbit(Body("Neptune", "Neptun", 102.4092e24 / MSUN_KG, 24622.0, "purple"), 30.06992276, 1.77004347, -55.12002969),
)

# JPL/JUP365 mean orbital elements and physical parameters for Galilean moons.
# Masses are derived from JPL GM values in km^3/s^2 via M = GM/G.
MOONS: tuple[MoonOrbit, ...] = (
    MoonOrbit(Body("Io", "Io", 5959.91547 * GM_TO_MSUN, 1821.49, "crimson"), 421_800.0, 1.762732, 0.0, 330.9),
    MoonOrbit(Body("Europa", "Europa", 3202.71210 * GM_TO_MSUN, 1560.80, "deepskyblue"), 671_100.0, 3.525463, 0.5, 345.4),
    MoonOrbit(Body("Ganymede", "Ganymed", 9887.83275 * GM_TO_MSUN, 2631.20, "seagreen"), 1_070_400.0, 7.155588, 0.2, 324.8),
    MoonOrbit(Body("Callisto", "Callisto", 7179.28340 * GM_TO_MSUN, 2410.30, "darkviolet"), 1_882_700.0, 16.690440, 0.3, 87.4),
)

BODIES: tuple[Body, ...] = (SUN,) + tuple(p.body for p in PLANETS) + tuple(m.body for m in MOONS)
SUN_IDX = 0
MERCURY_IDX = 1
JUPITER_IDX = 5
MOON_START_IDX = 1 + len(PLANETS)
MOON_INDICES = tuple(range(MOON_START_IDX, MOON_START_IDX + len(MOONS)))
PLANET_INDICES = tuple(range(0, 1 + len(PLANETS)))
MAX_PLANET_RADIUS = max(b.radius_km for b in BODIES[:1 + len(PLANETS)])
MAX_MOON_RADIUS = max(m.body.radius_km for m in MOONS)

# =============================================================================
# Language
# =============================================================================

LANG_OPTIONS = ("English", "Čeština")
TEXT = {
    "en": {
        "title": "Solar System and Jupiter's Galilean moons",
        "build": "Build: solar + Galilean moons v2 (Inner planets default)",
        "what": "What this app computes",
        "reset": "Reset to initial values",
        "global": "Global controls",
        "time_days": "Simulated time [days]",
        "frames": "Stored time frames",
        "rtol": "DOP853 relative tolerance log10",
        "show_planets": "Solar-System panel",
        "region": "Displayed Solar-System region",
        "to_jupiter": "To Jupiter",
        "all_planets": "All planets",
        "inner": "Inner planets",
        "jupiter_panel": "Jupiter panel",
        "jovian_halfwidth": "Jupiter-panel half-width [Jupiter radii]",
        "trail": "Trail length [frames]",
        "animation": "Animation controls",
        "max_anim": "Max Plotly animation frames",
        "duration": "Animation frame duration [ms]",
        "apply": "Apply and recompute",
        "apply_help": "Change settings, then press Apply and recompute. Visual marker sizes redraw immediately.",
        "visual": "Visual marker sizes",
        "sun_size": "Sun marker size [px]",
        "planet_size": "Largest planet marker size [px]",
        "moon_size": "Largest moon marker size [px]",
        "min_size": "Minimum marker size [px]",
        "show_names": "Show body names",
        "spinner": "Integrating the Newtonian system...",
        "left_title": "Solar System: barycentric Newtonian solution",
        "right_title": "Jupiter frame: same solution, Jupiter fixed",
        "fig_title": "One Newtonian solution shown in two coordinate systems",
        "play": "Play",
        "pause": "Pause",
        "export": "Export",
        "gif_frames": "GIF frames",
        "gif_fps": "GIF frame rate [fps]",
        "generate_gif": "Generate downloadable GIF",
        "download_gif": "Download GIF",
        "download_protocol": "Download TXT protocol",
        "protocol": "Simulation protocol",
        "metrics": "Diagnostics",
        "energy_drift": "Relative Newtonian energy drift",
        "min_sep": "Minimum separation [AU]",
        "note": "The Galilean-moon panel is Jupiter-centered: r' = r - r_Jupiter. This is a coordinate transformation of the same Newtonian solution, not a different force law.",
        "jupiter_scale_note": "Right-panel distances are shown in Jupiter radii R_J. The four Galilean moons are integrated together with the Sun and planets.",
    },
    "cs": {
        "title": "Sluneční soustava a Galileovy měsíce Jupiteru",
        "build": "Build: solar + Galileovy měsíce v2 (výchozí vnitřní planety)",
        "what": "Co aplikace počítá",
        "reset": "Obnovit výchozí hodnoty",
        "global": "Globální ovládání",
        "time_days": "Simulovaný čas [dny]",
        "frames": "Uložené časové snímky",
        "rtol": "Relativní tolerance DOP853 log10",
        "show_planets": "Panel Sluneční soustavy",
        "region": "Zobrazená oblast Sluneční soustavy",
        "to_jupiter": "Po Jupiter",
        "all_planets": "Všechny planety",
        "inner": "Vnitřní planety",
        "jupiter_panel": "Panel Jupiteru",
        "jovian_halfwidth": "Poloviční šířka panelu Jupiteru [poloměry Jupiteru]",
        "trail": "Délka stopy [snímky]",
        "animation": "Ovládání animace",
        "max_anim": "Maximální počet snímků animace Plotly",
        "duration": "Délka snímku animace [ms]",
        "apply": "Použít a přepočítat",
        "apply_help": "Změňte nastavení a poté stiskněte Použít a přepočítat. Vizuální velikosti značek se překreslí hned.",
        "visual": "Vizuální velikosti značek",
        "sun_size": "Velikost značky Slunce [px]",
        "planet_size": "Velikost značky největší planety [px]",
        "moon_size": "Velikost značky největšího měsíce [px]",
        "min_size": "Minimální velikost značky [px]",
        "show_names": "Zobrazit názvy těles",
        "spinner": "Integruji Newtonovský systém...",
        "left_title": "Sluneční soustava: barycentrické Newtonovské řešení",
        "right_title": "Soustava Jupiteru: stejné řešení, Jupiter fixován",
        "fig_title": "Jedno Newtonovské řešení ve dvou souřadných soustavách",
        "play": "Spustit",
        "pause": "Pozastavit",
        "export": "Export",
        "gif_frames": "Počet snímků GIFu",
        "gif_fps": "Snímková frekvence GIFu [fps]",
        "generate_gif": "Vygenerovat stažitelný GIF",
        "download_gif": "Stáhnout GIF",
        "download_protocol": "Stáhnout TXT protokol",
        "protocol": "Protokol simulace",
        "metrics": "Diagnostika",
        "energy_drift": "Relativní drift Newtonovské energie",
        "min_sep": "Minimální separace [AU]",
        "note": "Panel Galileových měsíců je centrovaný na Jupiter: r' = r - r_Jupiter. Jde o souřadnicovou transformaci téhož Newtonovského řešení, ne o jiný silový zákon.",
        "jupiter_scale_note": "V pravém panelu jsou vzdálenosti v poloměrech Jupiteru R_J. Čtyři Galileovy měsíce se integrují společně se Sluncem a planetami.",
    },
}


def lang_code() -> str:
    return "cs" if st.session_state.get("language") == "Čeština" else "en"


def tr(key: str, **kwargs) -> str:
    text = TEXT[lang_code()][key]
    return text.format(**kwargs) if kwargs else text


def body_name(idx: int) -> str:
    body = BODIES[idx]
    return body.cs_name if lang_code() == "cs" else body.name

# =============================================================================
# Mechanics
# =============================================================================

def rotation_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)), dtype=float)


def barycentric(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mtot = float(np.sum(masses))
    rcm = np.sum(pos * masses[:, None], axis=0) / mtot
    vcm = np.sum(vel * masses[:, None], axis=0) / mtot
    return pos - rcm, vel - vcm


def build_initial_conditions(moon_plane_tilt_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(BODIES)
    masses = np.array([b.mass_msun for b in BODIES], dtype=float)
    pos = np.zeros((n, 3), dtype=float)
    vel = np.zeros((n, 3), dtype=float)

    # Sun at origin first; planets on simplified circular heliocentric orbits.
    for i, orbit in enumerate(PLANETS, start=1):
        r = orbit.a_au
        phase = math.radians(orbit.phase_deg)
        inc = math.radians(orbit.inc_deg)
        local_pos = np.array((r * math.cos(phase), r * math.sin(phase), 0.0))
        speed = math.sqrt(G_MODEL * (masses[SUN_IDX] + masses[i]) / r)
        local_vel = np.array((-speed * math.sin(phase), speed * math.cos(phase), 0.0))
        rot = rotation_x(inc)
        pos[i] = rot @ local_pos
        vel[i] = rot @ local_vel

    # Galilean moons start relative to Jupiter.  We use JPL mean semimajor axes,
    # periods and phases to build clean didactic initial conditions.
    jpos = pos[JUPITER_IDX].copy()
    jvel = vel[JUPITER_IDX].copy()
    moon_plane_tilt = math.radians(moon_plane_tilt_deg)
    for k, moon in enumerate(MOONS):
        idx = MOON_START_IDX + k
        a = moon.a_km / KM_PER_AU
        phase = math.radians(moon.phase_deg)
        inc = math.radians(moon.inc_deg) + moon_plane_tilt
        # Period-based circular speed keeps the displayed angular period close
        # to the JPL mean element.  This is clearer for education than forcing
        # exact two-body circular speed while also ignoring all ephemeris terms.
        speed = 2.0 * math.pi * a / (moon.period_days / DAYS_PER_YEAR)
        local_pos = np.array((a * math.cos(phase), a * math.sin(phase), 0.0))
        local_vel = np.array((-speed * math.sin(phase), speed * math.cos(phase), 0.0))
        rot = rotation_x(inc)
        pos[idx] = jpos + rot @ local_pos
        vel[idx] = jvel + rot @ local_vel

    pos, vel = barycentric(pos, vel, masses)
    return pos, vel, masses


def acceleration_vectorized(pos: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Vectorized Newtonian N-body acceleration."""
    dr = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", dr, dr) + SOFTENING_AU**2
    np.fill_diagonal(r2, np.inf)
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    return -G_MODEL * np.sum(dr * inv_r3[:, :, None] * masses[None, :, None], axis=1)


def rhs(_t: float, y: np.ndarray, masses: np.ndarray) -> np.ndarray:
    n = len(masses)
    pos = y[: 3 * n].reshape((n, 3))
    vel = y[3 * n :].reshape((n, 3))
    acc = acceleration_vectorized(pos, masses)
    return np.concatenate((vel.reshape(-1), acc.reshape(-1)))


def energy(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray) -> float:
    kin = 0.5 * float(np.sum(masses[:, None] * vel * vel))
    pot = 0.0
    n = len(masses)
    for i in range(n):
        for j in range(i + 1, n):
            r = math.sqrt(float(np.dot(pos[i] - pos[j], pos[i] - pos[j])) + SOFTENING_AU**2)
            pot += -G_MODEL * masses[i] * masses[j] / r
    return kin + pot


@st.cache_data(show_spinner=False)
def simulate_cached(total_days: float, n_frames: int, log10_rtol: float, moon_plane_tilt_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    pos0, vel0, masses = build_initial_conditions(moon_plane_tilt_deg)
    n = len(masses)
    y0 = np.concatenate((pos0.reshape(-1), vel0.reshape(-1)))
    t_end = total_days / DAYS_PER_YEAR
    times = np.linspace(0.0, t_end, int(n_frames))
    rtol = 10.0 ** float(log10_rtol)
    atol = min(1e-12, rtol * 1e-3)
    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, masses),
        t_span=(0.0, t_end),
        y0=y0,
        method="DOP853",
        t_eval=times,
        rtol=rtol,
        atol=atol,
        vectorized=False,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    y = sol.y.T
    frames = y[:, : 3 * n].reshape((len(times), n, 3))
    vframes = y[:, 3 * n :].reshape((len(times), n, 3))
    e0 = energy(frames[0], vframes[0], masses)
    e1 = energy(frames[-1], vframes[-1], masses)
    diagnostics = {
        "energy_drift": float((e1 - e0) / max(abs(e0), 1e-300)),
        "nfev": int(sol.nfev),
    }
    return times, frames, vframes, masses, diagnostics

# =============================================================================
# Plot helpers
# =============================================================================

def visible_planet_indices(region: str) -> list[int]:
    if region == "Inner planets":
        return list(range(SUN_IDX, JUPITER_IDX))  # Sun through Mars
    if region == "To Jupiter":
        return list(range(SUN_IDX, JUPITER_IDX + 1))
    return list(PLANET_INDICES)


def marker_size_for_indices(indices: Sequence[int], sun_px: float, planet_px: float, moon_px: float, min_px: float) -> list[float]:
    sizes = []
    for idx in indices:
        b = BODIES[idx]
        if idx == SUN_IDX:
            sizes.append(float(sun_px))
        elif idx in MOON_INDICES:
            norm = max(b.radius_km / MAX_MOON_RADIUS, 1e-9)
            sizes.append(float(min_px + (moon_px - min_px) * norm ** 0.45))
        else:
            norm = max(b.radius_km / MAX_PLANET_RADIUS, 1e-9)
            sizes.append(float(min_px + (planet_px - min_px) * norm ** 0.35))
    return sizes


def progressive_slice(frame_idx: int, trail_frames: int) -> slice:
    start = max(0, int(frame_idx) - int(trail_frames) + 1)
    return slice(start, int(frame_idx) + 1)


def left_axis_range(region: str) -> tuple[float, float]:
    if region == "Inner planets":
        half = 1.9
    elif region == "To Jupiter":
        half = 6.5
    else:
        half = 33.0
    return -half, half


def make_figure(
    times: np.ndarray,
    frames: np.ndarray,
    frame_idx: int,
    region: str,
    jupiter_halfwidth_rj: float,
    trail_frames: int,
    max_animation_frames: int,
    animation_duration_ms: int,
    show_names: bool,
    sun_px: float,
    planet_px: float,
    moon_px: float,
    min_px: float,
) -> go.Figure:
    planet_indices = visible_planet_indices(region)
    moon_panel_indices = [JUPITER_IDX] + list(MOON_INDICES)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(tr("left_title"), tr("right_title")),
        horizontal_spacing=0.02,
    )
    frame_idx = int(np.clip(frame_idx, 0, len(times) - 1))
    sl = progressive_slice(frame_idx, trail_frames)

    def add_panel(indices: Sequence[int], panel: int, centered_on_jupiter: bool):
        sizes = marker_size_for_indices(indices, sun_px, planet_px, moon_px, min_px)
        names = [body_name(i) for i in indices]
        colors = [BODIES[i].color for i in indices]
        for idx in indices:
            xyz = frames[sl, idx, :].copy()
            if centered_on_jupiter:
                xyz = (xyz - frames[sl, JUPITER_IDX, :]) / JUPITER_RADIUS_AU
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                    mode="lines",
                    line=dict(color=BODIES[idx].color, width=2),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=panel,
            )
        pts = frames[frame_idx, indices, :].copy()
        if centered_on_jupiter:
            pts = (pts - frames[frame_idx, JUPITER_IDX, :]) / JUPITER_RADIUS_AU
        text = names if show_names else ["" for _ in names]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers+text" if show_names else "markers",
                marker=dict(size=sizes, color=colors, opacity=0.96, sizemode="diameter"),
                text=text,
                textposition="top center",
                hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=panel,
        )

    add_panel(planet_indices, 1, centered_on_jupiter=False)
    add_panel(moon_panel_indices, 2, centered_on_jupiter=True)

    lmin, lmax = left_axis_range(region)
    rmin, rmax = -float(jupiter_halfwidth_rj), float(jupiter_halfwidth_rj)
    scene_left = dict(
        xaxis=dict(title="x [AU]", range=[lmin, lmax], autorange=False),
        yaxis=dict(title="y [AU]", range=[lmin, lmax], autorange=False),
        zaxis=dict(title="z [AU]", range=[lmin, lmax], autorange=False),
        aspectmode="cube",
        uirevision=PLOT_UIREVISION,
    )
    scene_right = dict(
        xaxis=dict(title="x [R_J]", range=[rmin, rmax], autorange=False),
        yaxis=dict(title="y [R_J]", range=[rmin, rmax], autorange=False),
        zaxis=dict(title="z [R_J]", range=[rmin, rmax], autorange=False),
        aspectmode="cube",
        uirevision=PLOT_UIREVISION,
    )
    fig.update_layout(
        scene=scene_left,
        scene2=scene_right,
        height=760,
        margin=dict(l=5, r=5, t=70, b=5),
        title=f"{tr('fig_title')} — t = {times[frame_idx] * DAYS_PER_YEAR:.1f} days",
        uirevision=PLOT_UIREVISION,
    )

    # Browser-side Plotly animation.  Future trails are not drawn in each frame.
    n_total = len(times)
    if n_total <= max_animation_frames:
        selected = list(range(n_total))
    else:
        selected = sorted(set(np.linspace(0, n_total - 1, int(max_animation_frames)).astype(int).tolist()))

    traces_per_panel = len(planet_indices) + 1
    traces_per_panel_2 = len(moon_panel_indices) + 1
    total_traces = traces_per_panel + traces_per_panel_2
    anim_frames = []
    for fidx in selected:
        data = []
        slf = progressive_slice(fidx, trail_frames)
        for idx in planet_indices:
            xyz = frames[slf, idx, :]
            data.append(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2]))
        pts = frames[fidx, planet_indices, :]
        data.append(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2]))
        for idx in moon_panel_indices:
            xyz = (frames[slf, idx, :] - frames[slf, JUPITER_IDX, :]) / JUPITER_RADIUS_AU
            data.append(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2]))
        pts = (frames[fidx, moon_panel_indices, :] - frames[fidx, JUPITER_IDX, :]) / JUPITER_RADIUS_AU
        data.append(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2]))
        anim_frames.append(go.Frame(data=data, traces=list(range(total_traces)), name=str(fidx)))
    fig.frames = anim_frames
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=1.08,
                buttons=[
                    dict(label=tr("play"), method="animate", args=[None, {"frame": {"duration": int(animation_duration_ms), "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}]),
                    dict(label=tr("pause"), method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.1,
                y=0.01,
                len=0.82,
                steps=[
                    dict(method="animate", label=f"{times[i] * DAYS_PER_YEAR:.0f}", args=[[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}])
                    for i in selected
                ],
            )
        ],
    )
    return fig


def render_gif(
    times: np.ndarray,
    frames: np.ndarray,
    region: str,
    jupiter_halfwidth_rj: float,
    trail_frames: int,
    gif_frames: int,
    gif_fps: int,
    show_names: bool,
) -> bytes:
    planet_indices = visible_planet_indices(region)
    moon_indices = [JUPITER_IDX] + list(MOON_INDICES)
    selected = np.unique(np.linspace(0, len(times) - 1, int(max(2, min(gif_frames, len(times))))).astype(int))
    lmin, lmax = left_axis_range(region)
    rmin, rmax = -float(jupiter_halfwidth_rj), float(jupiter_halfwidth_rj)
    fig = plt.figure(figsize=(12.5, 6.2), dpi=110)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for ax, title, lim, unit in ((ax1, tr("left_title"), (lmin, lmax), "AU"), (ax2, tr("right_title"), (rmin, rmax), "R_J")):
        ax.set_title(title)
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_zlim(*lim)
        ax.set_xlabel(f"x [{unit}]"); ax.set_ylabel(f"y [{unit}]"); ax.set_zlabel(f"z [{unit}]")
        ax.view_init(elev=22, azim=42)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    artists = []
    lines1 = [] ; marks1 = [] ; labels1 = []
    for idx in planet_indices:
        color = BODIES[idx].color
        (ln,) = ax1.plot([], [], [], color=color, lw=1.2, alpha=0.85)
        (mk,) = ax1.plot([], [], [], marker="o", linestyle="None", color=color, markersize=4.5)
        txt = ax1.text(0, 0, 0, body_name(idx), fontsize=7, color="black")
        lines1.append(ln); marks1.append(mk); labels1.append(txt)
    lines2 = [] ; marks2 = [] ; labels2 = []
    for idx in moon_indices:
        color = BODIES[idx].color
        ms = 7.0 if idx == JUPITER_IDX else 5.0
        (ln,) = ax2.plot([], [], [], color=color, lw=1.2, alpha=0.85)
        (mk,) = ax2.plot([], [], [], marker="o", linestyle="None", color=color, markersize=ms)
        txt = ax2.text(0, 0, 0, body_name(idx), fontsize=7, color="black")
        lines2.append(ln); marks2.append(mk); labels2.append(txt)
    time_text = fig.text(0.5, 0.965, "", ha="center", va="top", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(k):
        f = int(selected[k])
        sl = progressive_slice(f, trail_frames)
        for a, idx in enumerate(planet_indices):
            xyz = frames[sl, idx, :]
            lines1[a].set_data_3d(xyz[:, 0], xyz[:, 1], xyz[:, 2])
            p = frames[f, idx, :]
            marks1[a].set_data_3d([p[0]], [p[1]], [p[2]])
            labels1[a].set_visible(show_names)
            labels1[a].set_position((p[0] + 0.02, p[1] + 0.02)); labels1[a].set_3d_properties(p[2] + 0.02, zdir="z")
        for a, idx in enumerate(moon_indices):
            xyz = (frames[sl, idx, :] - frames[sl, JUPITER_IDX, :]) / JUPITER_RADIUS_AU
            lines2[a].set_data_3d(xyz[:, 0], xyz[:, 1], xyz[:, 2])
            p = (frames[f, idx, :] - frames[f, JUPITER_IDX, :]) / JUPITER_RADIUS_AU
            marks2[a].set_data_3d([p[0]], [p[1]], [p[2]])
            labels2[a].set_visible(show_names)
            labels2[a].set_position((p[0] + 0.4, p[1] + 0.4)); labels2[a].set_3d_properties(p[2] + 0.4, zdir="z")
        time_text.set_text(f"{tr('fig_title')} — t = {times[f] * DAYS_PER_YEAR:.1f} days")
        return lines1 + marks1 + labels1 + lines2 + marks2 + labels2 + [time_text]

    anim = FuncAnimation(fig, update, frames=len(selected), interval=1000.0 / max(1, int(gif_fps)), blit=False)
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=True) as tmp:
        anim.save(tmp.name, writer=PillowWriter(fps=max(1, int(gif_fps))))
        tmp.seek(0)
        data = tmp.read()
    plt.close(fig)
    return data


def make_protocol(times: np.ndarray, frames: np.ndarray, vframes: np.ndarray, masses: np.ndarray, diagnostics: dict[str, float]) -> str:
    lines = []
    lines.append("Solar System + Galilean moons Newtonian simulation protocol")
    lines.append(f"Simulated time: {times[-1] * DAYS_PER_YEAR:.6g} days")
    lines.append(f"Frames: {len(times)}")
    lines.append(f"G = {G_MODEL:.16g} AU^3/(M_sun yr^2)")
    lines.append(f"Softening = {SOFTENING_AU:.3e} AU")
    lines.append(f"Function evaluations: {diagnostics.get('nfev', 0)}")
    lines.append(f"Relative energy drift: {diagnostics.get('energy_drift', 0.0):.8e}")
    lines.append("")
    lines.append("Initial and final positions are in barycentric AU.")
    for idx, b in enumerate(BODIES):
        p0 = frames[0, idx]; v0 = vframes[0, idx]
        p1 = frames[-1, idx]; v1 = vframes[-1, idx]
        lines.append(f"{b.name}:")
        lines.append(f"  mass [M_sun] = {masses[idx]:.16e}")
        lines.append(f"  r0 [AU] = ({p0[0]:.16e}, {p0[1]:.16e}, {p0[2]:.16e})")
        lines.append(f"  v0 [AU/yr] = ({v0[0]:.16e}, {v0[1]:.16e}, {v0[2]:.16e})")
        lines.append(f"  r_final [AU] = ({p1[0]:.16e}, {p1[1]:.16e}, {p1[2]:.16e})")
        lines.append(f"  v_final [AU/yr] = ({v1[0]:.16e}, {v1[1]:.16e}, {v1[2]:.16e})")
    return "\n".join(lines) + "\n"

# =============================================================================
# Session state and UI
# =============================================================================

DEFAULTS = {
    "language": "English",
    "total_days": 60.0,
    "n_frames": 360,
    "log10_rtol": -8.0,
    "moon_plane_tilt_deg": 0.0,
    "region": "Inner planets",
    "jupiter_halfwidth_rj": 32.0,
    "trail_frames": 160,
    "max_anim_frames": 180,
    "anim_duration_ms": 45,
    "sun_px": 8.0,
    "planet_px": 13.0,
    "moon_px": 10.0,
    "min_px": 4.0,
    "show_names": True,
    "gif_frames": 90,
    "gif_fps": 12,
}


def init_state():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)


def reset_keep_language():
    lang = st.session_state.get("language", DEFAULTS["language"])
    for k, v in DEFAULTS.items():
        if k != "language":
            st.session_state[k] = v
    st.session_state["language"] = lang

st.set_page_config(page_title="Solar System + Galilean moons", layout="wide")
init_state()
st.sidebar.selectbox("Language / Jazyk", LANG_OPTIONS, key="language")
st.title(tr("title"))
st.caption(tr("build"))
st.sidebar.caption(tr("build"))

st.sidebar.header("Reset")
st.sidebar.button(tr("reset"), use_container_width=True, on_click=reset_keep_language)

with st.expander(tr("what"), expanded=False):
    if lang_code() == "cs":
        st.markdown(
            """
Aplikace integruje jeden společný Newtonovský model obsahující Slunce, osm planet a čtyři Galileovy měsíce Jupiteru: Io, Europu, Ganymeda a Callisto.
Levý panel ukazuje barycentrické souřadnice celé Sluneční soustavy. Pravý panel ukazuje stejné vypočtené polohy převedené do soustavy, kde je Jupiter v počátku.
            """
        )
        st.latex(r"\ddot{\mathbf r}_i=-\sum_{j\ne i}Gm_j\frac{\mathbf r_i-\mathbf r_j}{(|\mathbf r_i-\mathbf r_j|^2+\epsilon^2)^{3/2}}");
        st.latex(r"\mathbf r'_i(t)=\frac{\mathbf r_i(t)-\mathbf r_{\rm Jupiter}(t)}{R_J}")
        st.markdown(
            """
Pravý panel tedy není jiný fyzikální model. Je to jen změna vztažné soustavy: Jupiter je zvolen jako střed. To dobře ukazuje, proč je přirozené mluvit o Jupiterově soustavě měsíců, i když celý systém stále obíhá kolem barycentra Sluneční soustavy.

Počáteční podmínky planet jsou zjednodušené kruhové dráhy z průměrných vzdáleností. Počáteční podmínky Galileových měsíců používají střední prvky JPL/JUP365: hlavní poloosy, fáze, sklony a periody. Hmotnosti měsíců jsou odvozené z hodnot JPL \(GM\). Nejde o přesnou efemeridu JPL Horizons, ale o výukovou numerickou vizualizaci.

Numerická integrace používá `scipy.integrate.solve_ivp` s metodou DOP853 a vektorované NumPy vyhodnocení Newtonovských zrychlení. Slidery `Simulated time`, `Stored time frames` a `DOP853 tolerance` mění numerický výpočet. Vizuální velikosti markerů mění pouze vykreslení.
            """
        )
    else:
        st.markdown(
            """
The app integrates one common Newtonian model containing the Sun, eight planets and Jupiter's four Galilean moons: Io, Europa, Ganymede and Callisto.
The left panel shows barycentric Solar-System coordinates. The right panel shows the same computed positions transformed to a Jupiter-centered coordinate system.
            """
        )
        st.latex(r"\ddot{\mathbf r}_i=-\sum_{j\ne i}Gm_j\frac{\mathbf r_i-\mathbf r_j}{(|\mathbf r_i-\mathbf r_j|^2+\epsilon^2)^{3/2}}");
        st.latex(r"\mathbf r'_i(t)=\frac{\mathbf r_i(t)-\mathbf r_{\rm Jupiter}(t)}{R_J}")
        st.markdown(
            """
The right panel is not a different physical model. It is only a change of reference frame: Jupiter is chosen as the center. This illustrates why it is natural to talk about Jupiter's system of moons, even though the whole system still moves in the Solar-System barycentric frame.

Planet initial conditions are simplified circular orbits based on mean distances. The Galilean-moon initial conditions use JPL/JUP365 mean elements: semimajor axes, phases, inclinations and periods. Moon masses are derived from JPL \(GM\) values. This is not a high-precision JPL Horizons ephemeris; it is an educational numerical visualization.

The numerical integration uses `scipy.integrate.solve_ivp` with the DOP853 method and vectorized NumPy evaluation of Newtonian accelerations. `Simulated time`, `Stored time frames` and `DOP853 tolerance` change the numerical computation. Marker-size sliders only redraw the plot.
            """
        )

with st.sidebar.form("controls"):
    st.header(tr("global"))
    st.slider(tr("time_days"), 5.0, 730.0, step=5.0, key="total_days")
    st.slider(tr("frames"), 80, 1200, step=20, key="n_frames")
    st.slider(tr("rtol"), -11.0, -5.0, step=0.5, key="log10_rtol")
    st.slider("Galilean-moon plane tilt [deg]", -10.0, 10.0, step=0.5, key="moon_plane_tilt_deg")
    st.header(tr("show_planets"))
    st.selectbox(tr("region"), ("Inner planets", "To Jupiter", "All planets"), key="region", format_func=lambda x: {"Inner planets": tr("inner"), "To Jupiter": tr("to_jupiter"), "All planets": tr("all_planets")}[x])
    st.header(tr("jupiter_panel"))
    st.slider(tr("jovian_halfwidth"), 8.0, 80.0, step=1.0, key="jupiter_halfwidth_rj")
    st.slider(tr("trail"), 5, 600, step=5, key="trail_frames")
    st.header(tr("animation"))
    st.slider(tr("max_anim"), 40, 400, step=10, key="max_anim_frames")
    st.slider(tr("duration"), 10, 150, step=5, key="anim_duration_ms")
    submitted = st.form_submit_button(tr("apply"), use_container_width=True, help=tr("apply_help"))

st.sidebar.header(tr("visual"))
st.sidebar.slider(tr("sun_size"), 2.0, 24.0, step=0.5, key="sun_px")
st.sidebar.slider(tr("planet_size"), 4.0, 28.0, step=0.5, key="planet_px")
st.sidebar.slider(tr("moon_size"), 4.0, 24.0, step=0.5, key="moon_px")
st.sidebar.slider(tr("min_size"), 1.0, 10.0, step=0.5, key="min_px")
st.sidebar.checkbox(tr("show_names"), key="show_names")

with st.spinner(tr("spinner")):
    times, frames, vframes, masses, diag = simulate_cached(
        float(st.session_state.total_days),
        int(st.session_state.n_frames),
        float(st.session_state.log10_rtol),
        float(st.session_state.moon_plane_tilt_deg),
    )

fig = make_figure(
    times, frames, 0,
    st.session_state.region,
    float(st.session_state.jupiter_halfwidth_rj),
    int(st.session_state.trail_frames),
    int(st.session_state.max_anim_frames),
    int(st.session_state.anim_duration_ms),
    bool(st.session_state.show_names),
    float(st.session_state.sun_px),
    float(st.session_state.planet_px),
    float(st.session_state.moon_px),
    float(st.session_state.min_px),
)
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
st.info(tr("note"))
st.caption(tr("jupiter_scale_note"))

st.subheader(tr("metrics"))
c1, c2, c3 = st.columns(3)
c1.metric(tr("energy_drift"), f"{diag['energy_drift']:.3e}")
c2.metric("DOP853 function evaluations", f"{diag['nfev']:,}")
c3.metric("Stored frames", f"{len(times):,}")

st.subheader(tr("export"))
g1, g2 = st.columns(2)
with g1:
    st.slider(tr("gif_frames"), 20, 180, step=10, key="gif_frames")
with g2:
    st.slider(tr("gif_fps"), 4, 24, step=1, key="gif_fps")

if st.button(tr("generate_gif"), use_container_width=True):
    with st.spinner("Rendering GIF..." if lang_code() == "en" else "Vykresluji GIF..."):
        gif_data = render_gif(
            times, frames,
            st.session_state.region,
            float(st.session_state.jupiter_halfwidth_rj),
            int(st.session_state.trail_frames),
            int(st.session_state.gif_frames),
            int(st.session_state.gif_fps),
            bool(st.session_state.show_names),
        )
    st.download_button(tr("download_gif"), gif_data, file_name="solar_jupiter_galilean_moons.gif", mime="image/gif", use_container_width=True)

protocol = make_protocol(times, frames, vframes, masses, diag)
st.download_button(tr("download_protocol"), protocol, file_name="solar_jupiter_galilean_moons_protocol.txt", mime="text/plain", use_container_width=True)

st.subheader("Bodies" if lang_code() == "en" else "Tělesa")
rows = []
for i, b in enumerate(BODIES):
    rows.append({"body": body_name(i), "mass [M_sun]": masses[i], "radius [km]": b.radius_km})
st.dataframe(rows, use_container_width=True, hide_index=True)
