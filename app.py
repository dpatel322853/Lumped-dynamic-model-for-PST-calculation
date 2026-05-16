
"""
Streamlit PST Calculator App
Low Temperature Protection of CS Piping downstream of LP Ethylene Superheater

Deployable on Streamlit Community Cloud.
"""

import io
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="PST Calculator | DPSIM",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main {background-color: #0b1120; color: #e5e7eb;}
    .stApp {background: linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);}
    h1, h2, h3 {color: #f8fafc;}
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    }
    .small-note {
        color: #94a3b8;
        font-size: 0.90rem;
    }
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 20px 0 5px 0;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Model data class
# -----------------------------
@dataclass
class PSTCase:
    case_name: str = "Base Case"

    # Temperature data, degC
    T_eth_in: float = -40.0
    T_eth_out_initial: float = 30.0
    TALL: float = 0.0
    T_CS_limit: float = -29.0
    T_ambient: float = 30.0

    # Ethylene process data
    m_eth_normal: float = 1.0
    Cp_eth: float = 2200.0
    M_eth_out_control_volume: float = 50.0

    # Superheater metal thermal inertia
    M_hex_metal: float = 2000.0
    Cp_hex_metal: float = 500.0
    UA_eth_to_hex: float = 8000.0

    # Heating failure model
    heating_failure_mode: str = "duty_loss"
    Q_normal: float = 0.0
    residual_Q_fraction: float = 0.0
    residual_Q_tau: float = 10.0
    T_methanol_failed: float = 30.0
    UA_methanol_to_hex: float = 0.0

    # CS pipe data
    M_pipe_metal: float = 800.0
    Cp_pipe_metal: float = 500.0
    M_pipe_fluid: float = 50.0
    UA_pipe_fluid_wall: float = 4000.0
    UA_pipe_ambient: float = 0.0
    transport_delay_s: float = 0.0

    # SIF timing
    TT_tau: float = 5.0
    logic_delay: float = 1.0
    solenoid_delay: float = 1.0
    XV_stroke_time: float = 10.0
    XV_leakage_fraction: float = 0.0

    # Simulation settings
    t_final: float = 600.0
    max_step: float = 0.25
    required_margin_fraction: float = 0.10

    def clean(self):
        for k, v in list(asdict(self).items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                setattr(self, k, PSTCase().__dict__[k])
        self.heating_failure_mode = str(self.heating_failure_mode).strip()
        if self.Q_normal == 0.0:
            self.Q_normal = max(0.0, self.m_eth_normal * self.Cp_eth * (self.T_eth_out_initial - self.T_eth_in))
        return self


# -----------------------------
# Calculation functions
# -----------------------------
def xv_opening_fraction(t: float, t_trip: Optional[float], case: PSTCase) -> float:
    if t_trip is None:
        return 1.0

    t_start_close = t_trip + case.logic_delay + case.solenoid_delay

    if t < t_start_close:
        opening = 1.0
    elif t_start_close <= t <= t_start_close + case.XV_stroke_time:
        opening = 0.0 if case.XV_stroke_time <= 0 else 1.0 - (t - t_start_close) / case.XV_stroke_time
    else:
        opening = 0.0

    return max(case.XV_leakage_fraction, min(1.0, opening))


def residual_heating_duty(t: float, case: PSTCase) -> float:
    if case.heating_failure_mode == "residual_heating":
        return case.Q_normal * case.residual_Q_fraction * math.exp(-max(t, 0.0) / max(case.residual_Q_tau, 1e-6))
    return 0.0


def methanol_heat_to_hex(T_hex: float, t: float, case: PSTCase) -> float:
    if case.heating_failure_mode == "cold_methanol_circulation":
        return case.UA_methanol_to_hex * (case.T_methanol_failed - T_hex)
    return residual_heating_duty(t, case)


def dynamic_model(t: float, y: np.ndarray, case: PSTCase, t_trip: Optional[float]) -> List[float]:
    T_hex, T_eth_out, T_pipe_fluid, T_pipe_wall, T_TT = y

    opening = xv_opening_fraction(t, t_trip, case)
    m_eth = case.m_eth_normal * opening

    Q_eth_hex = case.UA_eth_to_hex * (case.T_eth_in - T_hex)
    Q_methanol = methanol_heat_to_hex(T_hex, t, case)

    dT_hex_dt = (Q_eth_hex + Q_methanol) / max(case.M_hex_metal * case.Cp_hex_metal, 1e-9)

    dT_eth_out_dt = (
        m_eth * case.Cp_eth * (case.T_eth_in - T_eth_out)
        + case.UA_eth_to_hex * (T_hex - T_eth_out)
    ) / max(case.M_eth_out_control_volume * case.Cp_eth, 1e-9)

    dT_TT_dt = (T_eth_out - T_TT) / max(case.TT_tau, 1e-9)

    effective_M_pipe_fluid = case.M_pipe_fluid + max(0.0, case.transport_delay_s) * max(case.m_eth_normal, 1e-9)

    dT_pipe_fluid_dt = (
        m_eth * case.Cp_eth * (T_eth_out - T_pipe_fluid)
        + case.UA_pipe_fluid_wall * (T_pipe_wall - T_pipe_fluid)
    ) / max(effective_M_pipe_fluid * case.Cp_eth, 1e-9)

    dT_pipe_wall_dt = (
        case.UA_pipe_fluid_wall * (T_pipe_fluid - T_pipe_wall)
        + case.UA_pipe_ambient * (case.T_ambient - T_pipe_wall)
    ) / max(case.M_pipe_metal * case.Cp_pipe_metal, 1e-9)

    return [dT_hex_dt, dT_eth_out_dt, dT_pipe_fluid_dt, dT_pipe_wall_dt, dT_TT_dt]


def event_tall(case: PSTCase):
    def _event(t, y):
        return y[4] - case.TALL
    _event.terminal = True
    _event.direction = -1
    return _event


def event_cs_limit(case: PSTCase):
    def _event(t, y):
        return y[3] - case.T_CS_limit
    _event.terminal = True
    _event.direction = -1
    return _event


def simulate_case(case: PSTCase) -> Dict:
    case.clean()
    y0 = np.array([
        case.T_eth_out_initial,
        case.T_eth_out_initial,
        case.T_eth_out_initial,
        case.T_eth_out_initial,
        case.T_eth_out_initial,
    ], dtype=float)

    sol_trip = solve_ivp(
        lambda t, y: dynamic_model(t, y, case, None),
        (0.0, case.t_final),
        y0,
        events=event_tall(case),
        max_step=case.max_step,
        rtol=1e-7,
        atol=1e-8,
        dense_output=True,
    )
    t_trip = float(sol_trip.t_events[0][0]) if len(sol_trip.t_events[0]) else None

    sol = solve_ivp(
        lambda t, y: dynamic_model(t, y, case, t_trip),
        (0.0, case.t_final),
        y0,
        events=event_cs_limit(case),
        max_step=case.max_step,
        rtol=1e-7,
        atol=1e-8,
        dense_output=False,
    )

    t = sol.t
    xv = np.array([100.0 * xv_opening_fraction(float(tt), t_trip, case) for tt in t])
    t_mdmt = float(sol.t_events[0][0]) if len(sol.t_events[0]) else None

    ts = pd.DataFrame({
        "time_s": t,
        "T_hex_metal_C": sol.y[0],
        "T_eth_out_actual_C": sol.y[1],
        "T_TT_measured_C": sol.y[4],
        "T_CS_pipe_fluid_C": sol.y[2],
        "T_CS_pipe_wall_C": sol.y[3],
        "XV_opening_pct": xv,
    })

    required_after_tall = case.logic_delay + case.solenoid_delay + case.XV_stroke_time
    required_total_sif = case.TT_tau + required_after_tall
    available_after_tall = None if (t_trip is None or t_mdmt is None) else t_mdmt - t_trip

    pass_without_margin = None
    pass_with_margin = None
    if available_after_tall is not None:
        pass_without_margin = required_after_tall < available_after_tall
        pass_with_margin = required_after_tall * (1 + case.required_margin_fraction) < available_after_tall

    summary = {
        "Case Name": case.case_name,
        "TALL Time, s": t_trip,
        "CS Limit Time / PST, s": t_mdmt,
        "Available Time After TALL, s": available_after_tall,
        "Required Time After TALL, s": required_after_tall,
        "Required Total SIF Time, s": required_total_sif,
        "Pass Without Margin": pass_without_margin,
        "Pass With Margin": pass_with_margin,
        "Minimum CS Pipe Wall Temp, °C": float(ts["T_CS_pipe_wall_C"].min()),
        "Minimum CS Pipe Fluid Temp, °C": float(ts["T_CS_pipe_fluid_C"].min()),
        "Minimum Superheater Outlet Temp, °C": float(ts["T_eth_out_actual_C"].min()),
        "Final XV Opening, %": float(ts["XV_opening_pct"].iloc[-1]),
    }

    return {"summary": summary, "timeseries": ts, "case": case}


def make_temperature_plot(ts: pd.DataFrame, case: PSTCase, summary: Dict):
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(ts["time_s"], ts["T_eth_out_actual_C"], label="Superheater outlet actual")
    ax.plot(ts["time_s"], ts["T_TT_measured_C"], label="TT measured temperature")
    ax.plot(ts["time_s"], ts["T_CS_pipe_fluid_C"], label="CS pipe fluid")
    ax.plot(ts["time_s"], ts["T_CS_pipe_wall_C"], label="CS pipe wall")
    ax.axhline(case.TALL, linestyle="--", label=f"TALL = {case.TALL:g} °C")
    ax.axhline(case.T_CS_limit, linestyle="--", label=f"CS limit = {case.T_CS_limit:g} °C")

    if summary["TALL Time, s"] is not None:
        ax.axvline(summary["TALL Time, s"], linestyle=":", label="TALL generated")
    if summary["CS Limit Time / PST, s"] is not None:
        ax.axvline(summary["CS Limit Time / PST, s"], linestyle=":", label="CS limit reached")

    ax.set_xlabel("Time, s")
    ax.set_ylabel("Temperature, °C")
    ax.set_title(f"PST Temperature Response - {case.case_name}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def make_xv_plot(ts: pd.DataFrame, case: PSTCase, summary: Dict):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(ts["time_s"], ts["XV_opening_pct"], label="XV opening")
    if summary["TALL Time, s"] is not None:
        tall = summary["TALL Time, s"]
        ax.axvline(tall, linestyle=":", label="TALL generated")
        ax.axvline(tall + case.logic_delay + case.solenoid_delay, linestyle=":", label="XV starts closing")
    ax.set_xlabel("Time, s")
    ax.set_ylabel("XV opening, %")
    ax.set_ylim(-5, 105)
    ax.set_title("XV Closure Profile")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def dataframe_to_excel_bytes(summary_df: pd.DataFrame, ts: pd.DataFrame, inputs: Dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([inputs]).to_excel(writer, sheet_name="inputs", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        ts.to_excel(writer, sheet_name="timeseries", index=False)
    output.seek(0)
    return output.getvalue()


def create_pdf_bytes(summary_df: pd.DataFrame, temp_png: bytes, xv_png: bytes) -> bytes:
    if not REPORTLAB_AVAILABLE:
        return b""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=14*mm, leftMargin=14*mm, topMargin=12*mm, bottomMargin=12*mm)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("PST Calculation Report", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Low temperature protection of CS piping downstream of LP Ethylene Superheater.", styles["BodyText"]))
    story.append(Spacer(1, 10))

    data = [list(summary_df.columns)] + summary_df.astype(str).values.tolist()
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))

    for png in [temp_png, xv_png]:
        img_buf = io.BytesIO(png)
        story.append(Image(img_buf, width=180*mm, height=90*mm))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Engineering note: This is a first-pass lumped dynamic model. Validate against HYSYS Dynamics / approved project data for formal SIS documentation.", styles["Italic"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------
# UI
# -----------------------------
st.title("🛡️ PST Calculator for CS Piping Low-Temperature Protection")
st.markdown(
    """
    <div class="small-note">
    Scenario: LP Ethylene Superheater methanol heating-medium failure. The app calculates TALL time, PST to CS temperature limit, available response time, and visualizes temperature/XV response.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Model basis / caution", expanded=False):
    st.markdown(
        """
        - This is a **first-pass lumped dynamic model**, suitable for screening and sensitivity checks.
        - For formal SIS/PST validation, benchmark against **Aspen HYSYS Dynamics** or an approved dynamic simulation.
        - For conservative calculations, use maximum ethylene flow, minimum inlet temperature, maximum TT lag, maximum XV stroke time, and no ambient heat gain unless justified.
        """
    )

st.sidebar.header("Input Data")

with st.sidebar:
    case_name = st.text_input("Case name", "Base Case")

    st.subheader("Temperatures")
    T_eth_in = st.number_input("Ethylene inlet temperature after failure, °C", value=-40.0, step=1.0)
    T_eth_out_initial = st.number_input("Initial superheater outlet temperature, °C", value=30.0, step=1.0)
    TALL = st.number_input("TALL trip setpoint, °C", value=0.0, step=1.0)
    T_CS_limit = st.number_input("CS pipe limiting temperature, °C", value=-29.0, step=1.0)
    T_ambient = st.number_input("Ambient temperature, °C", value=30.0, step=1.0)

    st.subheader("Ethylene / Superheater")
    m_eth_normal = st.number_input("Ethylene mass flow, kg/s", min_value=0.001, value=1.0, step=0.1)
    Cp_eth = st.number_input("Ethylene Cp, J/kg.K", min_value=1.0, value=2200.0, step=100.0)
    M_eth_out_control_volume = st.number_input("Outlet fluid inventory, kg", min_value=0.001, value=50.0, step=5.0)
    M_hex_metal = st.number_input("Superheater metal mass, kg", min_value=0.001, value=2000.0, step=100.0)
    Cp_hex_metal = st.number_input("Superheater metal Cp, J/kg.K", min_value=1.0, value=500.0, step=50.0)
    UA_eth_to_hex = st.number_input("UA ethylene-to-metal, W/K", min_value=0.001, value=8000.0, step=500.0)

    st.subheader("Heating Failure Mode")
    mode_label = st.selectbox(
        "Failure mode",
        ["Complete duty loss", "Residual heating decay", "Cold methanol circulation"],
        index=0,
    )
    mode_map = {
        "Complete duty loss": "duty_loss",
        "Residual heating decay": "residual_heating",
        "Cold methanol circulation": "cold_methanol_circulation",
    }
    heating_failure_mode = mode_map[mode_label]
    Q_normal = st.number_input("Normal duty, W (0 = auto estimate)", min_value=0.0, value=0.0, step=10000.0)
    residual_Q_fraction = 0.0
    residual_Q_tau = 10.0
    T_methanol_failed = 30.0
    UA_methanol_to_hex = 0.0
    if heating_failure_mode == "residual_heating":
        residual_Q_fraction = st.number_input("Residual duty fraction", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
        residual_Q_tau = st.number_input("Residual duty decay tau, s", min_value=0.001, value=30.0, step=5.0)
    elif heating_failure_mode == "cold_methanol_circulation":
        T_methanol_failed = st.number_input("Failed methanol temperature, °C", value=0.0, step=1.0)
        UA_methanol_to_hex = st.number_input("UA methanol-to-metal, W/K", min_value=0.0, value=2000.0, step=500.0)

    st.subheader("CS Pipe")
    M_pipe_metal = st.number_input("CS pipe metal mass, kg", min_value=0.001, value=800.0, step=50.0)
    Cp_pipe_metal = st.number_input("CS pipe metal Cp, J/kg.K", min_value=1.0, value=500.0, step=50.0)
    M_pipe_fluid = st.number_input("CS pipe fluid holdup, kg", min_value=0.001, value=50.0, step=5.0)
    UA_pipe_fluid_wall = st.number_input("UA pipe fluid-to-wall, W/K", min_value=0.001, value=4000.0, step=500.0)
    UA_pipe_ambient = st.number_input("UA pipe-to-ambient, W/K", min_value=0.0, value=0.0, step=100.0)
    transport_delay_s = st.number_input("Transport delay to CS pipe, s", min_value=0.0, value=0.0, step=1.0)

    st.subheader("SIF / XV Timing")
    TT_tau = st.number_input("TT/thermowell lag, s", min_value=0.001, value=5.0, step=1.0)
    logic_delay = st.number_input("Logic solver delay, s", min_value=0.0, value=1.0, step=0.5)
    solenoid_delay = st.number_input("Solenoid delay, s", min_value=0.0, value=1.0, step=0.5)
    XV_stroke_time = st.number_input("XV stroke time, s", min_value=0.0, value=10.0, step=1.0)
    XV_leakage_fraction = st.number_input("XV leakage fraction after closure", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
    required_margin_fraction = st.number_input("Required margin fraction", min_value=0.0, max_value=1.0, value=0.10, step=0.05)

    st.subheader("Simulation")
    t_final = st.number_input("Final simulation time, s", min_value=1.0, value=600.0, step=60.0)
    max_step = st.number_input("Max integration step, s", min_value=0.001, value=0.25, step=0.05)

    run_button = st.button("Run PST Calculation", type="primary", use_container_width=True)

case = PSTCase(
    case_name=case_name,
    T_eth_in=T_eth_in,
    T_eth_out_initial=T_eth_out_initial,
    TALL=TALL,
    T_CS_limit=T_CS_limit,
    T_ambient=T_ambient,
    m_eth_normal=m_eth_normal,
    Cp_eth=Cp_eth,
    M_eth_out_control_volume=M_eth_out_control_volume,
    M_hex_metal=M_hex_metal,
    Cp_hex_metal=Cp_hex_metal,
    UA_eth_to_hex=UA_eth_to_hex,
    heating_failure_mode=heating_failure_mode,
    Q_normal=Q_normal,
    residual_Q_fraction=residual_Q_fraction,
    residual_Q_tau=residual_Q_tau,
    T_methanol_failed=T_methanol_failed,
    UA_methanol_to_hex=UA_methanol_to_hex,
    M_pipe_metal=M_pipe_metal,
    Cp_pipe_metal=Cp_pipe_metal,
    M_pipe_fluid=M_pipe_fluid,
    UA_pipe_fluid_wall=UA_pipe_fluid_wall,
    UA_pipe_ambient=UA_pipe_ambient,
    transport_delay_s=transport_delay_s,
    TT_tau=TT_tau,
    logic_delay=logic_delay,
    solenoid_delay=solenoid_delay,
    XV_stroke_time=XV_stroke_time,
    XV_leakage_fraction=XV_leakage_fraction,
    t_final=t_final,
    max_step=max_step,
    required_margin_fraction=required_margin_fraction,
)

if run_button:
    with st.spinner("Running dynamic PST calculation..."):
        result = simulate_case(case)
        st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]
    summary = result["summary"]
    ts = result["timeseries"]
    case = result["case"]

    st.subheader("Calculation Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TALL Time", "Not reached" if summary["TALL Time, s"] is None else f"{summary['TALL Time, s']:.2f} s")
    c2.metric("PST / CS Limit Time", "Not reached" if summary["CS Limit Time / PST, s"] is None else f"{summary['CS Limit Time / PST, s']:.2f} s")
    c3.metric("Available After TALL", "N/A" if summary["Available Time After TALL, s"] is None else f"{summary['Available Time After TALL, s']:.2f} s")
    c4.metric("Required After TALL", f"{summary['Required Time After TALL, s']:.2f} s")

    if summary["Pass With Margin"] is True:
        st.success("PASS: Required response time is less than available time after TALL with selected margin.")
    elif summary["Pass With Margin"] is False:
        st.error("FAIL: Required response time exceeds available time after TALL with selected margin.")
    else:
        st.info("CS limit was not reached within the simulation duration, or TALL was not reached. Review plots and extend simulation time if required.")

    summary_df = pd.DataFrame([summary])
    st.dataframe(summary_df, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Temperature Plot", "XV Plot", "Time Series"])

    with tab1:
        fig_temp = make_temperature_plot(ts, case, summary)
        st.pyplot(fig_temp, use_container_width=True)

    with tab2:
        fig_xv = make_xv_plot(ts, case, summary)
        st.pyplot(fig_xv, use_container_width=True)

    with tab3:
        st.dataframe(ts, use_container_width=True, height=420)

    temp_png = fig_to_png_bytes(fig_temp)
    xv_png = fig_to_png_bytes(fig_xv)
    excel_bytes = dataframe_to_excel_bytes(summary_df, ts, asdict(case))
    csv_bytes = ts.to_csv(index=False).encode("utf-8")
    pdf_bytes = create_pdf_bytes(summary_df, temp_png, xv_png)

    st.subheader("Downloads")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Download Excel Report", excel_bytes, file_name="pst_calculation_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    d2.download_button("Download Time Series CSV", csv_bytes, file_name="pst_timeseries.csv", mime="text/csv", use_container_width=True)
    d3.download_button("Download Temperature Plot", temp_png, file_name="pst_temperature_plot.png", mime="image/png", use_container_width=True)
    if pdf_bytes:
        d4.download_button("Download PDF Report", pdf_bytes, file_name="pst_report.pdf", mime="application/pdf", use_container_width=True)
    else:
        d4.info("PDF unavailable")
else:
    st.info("Enter inputs in the sidebar and click **Run PST Calculation**.")

st.markdown('<div class="footer">Created by Dhawal Patel | DPSIM</div>', unsafe_allow_html=True)
