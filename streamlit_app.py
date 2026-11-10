"""Streamlit front end for the Ringed Transit Simulator.

Run locally:   streamlit run streamlit_app.py
Streamlit Cloud: set "Main file path" to streamlit_app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

import ringed_transit as rt

st.set_page_config(page_title="Ringed Transit Simulator", layout="wide")
st.title("Ringed Transit Simulator")

tab_web, tab_py = st.tabs(["Interactive simulator", "Python model"])

# ------------------------------------------------ the full browser simulator
with tab_web:
    html = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
    components.html(html, height=1800, scrolling=True)

# ------------------------------------------------ ringed_transit.light_curve
with tab_py:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Orbit")
        P = st.slider("Period P [days]", 0.5, 50.0, 6.3, 0.1)
        a = st.slider("Semi-major axis a / R*", 2.0, 100.0, 16.0, 0.5)
        inc = st.slider("Inclination [deg]", 80.0, 90.0, 89.0, 0.05)
        rp = st.slider("Planet radius rp / R*", 0.01, 0.3, 0.10, 0.005)
    with c2:
        st.subheader("Ring")
        r_in, r_out = st.slider("Ring radii [planet radii]", 1.0, 4.0, (1.3, 2.2), 0.05)
        tau = st.slider("Optical depth tau", 0.0, 5.0, 0.8, 0.05)
        ring_incl = st.slider("Ring tilt [deg]", 0.0, 90.0, 72.0, 1.0)
        ring_pa = st.slider("Ring position angle [deg]", -90.0, 90.0, 25.0, 1.0)
    with c3:
        st.subheader("Star")
        u1 = st.slider("Limb darkening u1", 0.0, 1.0, 0.4, 0.01)
        u2 = st.slider("Limb darkening u2", -0.5, 1.0, 0.25, 0.01)
        n_grid = st.select_slider("Grid resolution", [100, 200, 300, 400], 200)

    @st.cache_data
    def curves(P, a, inc, rp, r_in, r_out, tau, ring_incl, ring_pa, u1, u2, n_grid):
        b = a * np.cos(np.radians(inc))
        half = P / np.pi * np.arcsin(min(1.0, (1 + rp * r_out) / a)) * 1.3
        t = np.linspace(-half, half, 1500)
        kw = dict(P=P, t0=0.0, a=a, inc=inc, rp=rp, u1=u1, u2=u2,
                  r_in=r_in, r_out=r_out, ring_incl=ring_incl,
                  ring_pa=ring_pa, n_grid=n_grid)
        ringed = rt.light_curve(t, tau=tau, **kw)
        bare = rt.light_curve(t, tau=0.0, **kw)
        return t, ringed, bare, b

    if r_out <= r_in:
        st.warning("Outer ring radius must be larger than the inner radius.")
        st.stop()

    t, ringed, bare, b = curves(P, a, inc, rp, r_in, r_out, tau,
                                ring_incl, ring_pa, u1, u2, n_grid)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw=dict(height_ratios=[3, 1]))
    th = t * 24
    ax1.plot(th, bare, color="0.6", lw=1.5, label="planet only")
    ax1.plot(th, ringed, color="C0", lw=1.5, label="ringed planet")
    ax1.set_ylabel("normalised flux")
    ax1.legend()
    ax2.plot(th, (ringed - bare) * 1e6, color="C3")
    ax2.set_ylabel("ring signal [ppm]")
    ax2.set_xlabel("time from mid-transit [hours]")
    fig.tight_layout()
    st.pyplot(fig)

    m1, m2, m3 = st.columns(3)
    m1.metric("Impact parameter b", f"{b:.2f}")
    m2.metric("Transit depth", f"{(1 - ringed.min()) * 100:.3f} %")
    m3.metric("Max ring signal", f"{np.abs(ringed - bare).max() * 1e6:.0f} ppm")
