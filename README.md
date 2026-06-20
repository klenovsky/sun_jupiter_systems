# Solar System + Galilean moons Streamlit app

This Streamlit web app visualizes one Newtonian integration containing the Sun, eight planets, Jupiter, and Jupiter's four Galilean moons: Io, Europa, Ganymede, and Callisto.

The left panel shows the Solar System in barycentric coordinates. The right panel shows the same computed solution in a Jupiter-centered frame, so the Galilean moons are visible around Jupiter.

The app is educational. It is not a high-precision JPL Horizons ephemeris.

## Files

```text
app.py
requirements.txt
README.md
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
3. In Streamlit Community Cloud, create a new app and set `app.py` as the entrypoint.
4. Deploy.

## Data and model

- Units: AU, Julian year, solar mass.
- The Newtonian N-body acceleration is evaluated with vectorized NumPy broadcasting.
- The time integration uses `scipy.integrate.solve_ivp` with method `DOP853`.
- Planet initial conditions are simplified circular orbits.
- Galilean-moon semimajor axes, phases, inclinations and periods are based on JPL/JUP365 mean elements.
- Galilean-moon masses are derived from JPL satellite `GM` values; radii are used for marker scaling.

The right panel uses the transformation

```text
r'_i(t) = (r_i(t) - r_Jupiter(t)) / R_J
```

where `R_J` is Jupiter's mean radius. This is a reference-frame transformation, not a new force law.

## Export

The app can export an animated GIF and a text simulation protocol.

## Default view

The left Solar-System panel now starts with `Inner planets` as the default displayed region. The user can still switch to `To Jupiter` or `All planets` in the sidebar.

Výchozí zobrazená oblast levého panelu Sluneční soustavy je nyní `Inner planets` / vnitřní planety. Uživatel může v sidebaru stále přepnout na `To Jupiter` nebo `All planets`.

## Recurrence / revival-like indicator

This version adds a visible recurrence diagnostic below the main 3D animation.  In the Jupiter-centered frame the app computes

```text
D_J(t) = RMS distance between the current relative configuration and the initial relative configuration
```

in units of Jupiter radii.  Small local minima of this curve mark classical recurrence-like moments: the selected group of bodies is again close to its initial visual arrangement in Jupiter's frame.

This is not a quantum revival.  It is a classical near-return of orbital phases/configuration in a system made of several nearly periodic motions.  The user can choose whether the diagnostic uses the Galilean moons only, the Sun plus the inner planets and Jupiter, or all bodies.
