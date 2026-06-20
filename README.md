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
