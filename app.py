"""
DPSIM - Merged PST Calculator App
Low Temperature Protection of CS Piping downstream of LP Ethylene Superheater

Merged version:
1) Existing lumped dynamic model supplied by Dhawal Patel
2) Screening / plug-flow PST calculator
3) Sensitivity analysis
4) Excel / CSV / PNG / PDF report export

Deployable on Streamlit Community Cloud / Render / Hugging Face Spaces.
"""

import io
import math
from dataclasses import dataclass, asdict
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

# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(
    page_title="PST Calculator | DPSIM",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Styling
# =============================================================================
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%); color: #e5e7eb;}
    h1, h2, h3 {color: #f8fafc;}
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    }
    .small-note {color: #94a3b8; font-size: 0.90rem;}
    .footer {text-align: center; color: #94a3b8; padding: 20px 0 5px 0; font-size: 0.9rem;}
    .pass-box {padding: 0.9rem; border-radius: 0.8rem; background: #064e3b; color: #d1fae5; border: 1px solid #10b981;}
    .fail-box {padding: 0.9rem; border-radius: 0.8rem; background: #450a0a; color: #fee2e2; border: 1px solid #ef4444;}
    .warn-box {padding: 0.9rem; border-radius: 0.8rem; background: #451a03; color: #ffedd5; border: 1px solid #f97316;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Model data class - supplied dynamic model basis, cleaned and retained
# =============================================================================
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
    m_eth_normal: float = 31600.0 / 3600.0
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

    # Screening / hydraulic inputs
    m_eth_kg_h: float = 31600.0
    rho_eth: float = 18.0
    ss_length_to_tt_m: float = 10.0
    ss_pipe_id_m: float = 0.154
    tt_to_xv_m: float = 1.0
    xv_to_cs_m: float = 1.0
    cs_pipe_id_m: float = 0.154
    hx_cooling_tau_s: float = 60.0
    ss_pipe_lag_s: float = 5.0
    required_margin_s: float = 5.0

    # Simulation settings
    t_final: float = 600.0
    max_step: float = 0.25
    required_margin_fraction: float = 0.10

    def clean(self):
        default = PSTCase()
        for k, v in list(asdict(self).items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                setattr(self, k, getattr(default, k))
        self.heating_failure_mode = str(self.heating_failure_mode).strip()
        if self.Q_normal == 0.0:
            self.Q_normal = max(0.0, self.m_eth_normal * self.Cp_eth * (self.T_eth_out_initial - self.T_eth_in))
        return self

# =============================================================================
# General utility functions
# =============================================================================
def pipe_area(d_m: float) -> float:
    return math.pi * d_m**2 / 4.0


def velocity_from_mass_flow(mdot_kg_h: float, rho_kg_m3: float, id_m: float) -> float:
    mdot_kg_s = mdot_kg_h / 3600.0
    return mdot_kg_s / max(rho_kg_m3 * pipe_area(id_m), 1e-12)


def time_to_temp_exp_decay(t_target: float, t_initial: float, t_final: float, tau_s: float) -> float:
    if tau_s <= 0:
        raise ValueError("Cooling time constant must be greater than zero.")
    if not (t_initial > t_target > t_final):
        raise ValueError("For cooling case, temperature relationship must be Initial > Target > Final.")
    ratio = (t_target - t_final) / (t_initial - t_final)
    return -tau_s * math.log(ratio)


def exp_decay_temperature(t_s, t_initial: float, t_final: float, tau_s: float):
    return t_final + (t_initial - t_final) * np.exp(-np.asarray(t_s) / max(tau_s, 1e-9))


def fmt_seconds(x):
    if x is None or pd.isna(x):
        return "Not reached"
    return f"{float(x):.2f} s"

# =============================================================================
# Dynamic model functions - retained from supplied earlier deployed app
# =============================================================================
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

# =============================================================================
# Screening calculator functions
# =============================================================================
def screening_calculation(case: PSTCase) -> Dict:
    v_ss = velocity_from_mass_flow(case.m_eth_kg_h, case.rho_eth, case.ss_pipe_id_m)
    v_cs = velocity_from_mass_flow(case.m_eth_kg_h, case.rho_eth, case.cs_pipe_id_m)
    t_transport_to_tt = case.ss_length_to_tt_m / max(v_ss, 1e-12)
    t_tt_to_xv = case.tt_to_xv_m / max(v_ss, 1e-12)
    t_xv_to_cs = case.xv_to_cs_m / max(v_cs, 1e-12)

    t_hx_to_tall = time_to_temp_exp_decay(case.TALL, case.T_eth_out_initial, case.T_eth_in, case.hx_cooling_tau_s)
    t_hx_to_cs = time_to_temp_exp_decay(case.T_CS_limit, case.T_eth_out_initial, case.T_eth_in, case.hx_cooling_tau_s)

    t_tt_process_tall = t_hx_to_tall + t_transport_to_tt + case.ss_pipe_lag_s
    t_trip_demand = t_tt_process_tall + case.TT_tau
    t_effective_isolation = t_trip_demand + case.logic_delay + case.solenoid_delay + case.XV_stroke_time
    t_cs_limit_arrival = t_hx_to_cs + t_transport_to_tt + t_tt_to_xv + t_xv_to_cs
    margin = t_cs_limit_arrival - t_effective_isolation
    pass_flag = margin >= case.required_margin_s

    summary = {
        "Ethylene Flowrate, kg/h": case.m_eth_kg_h,
        "Ethylene Flowrate, kg/s": case.m_eth_kg_h / 3600.0,
        "SS Velocity, m/s": v_ss,
        "CS Velocity, m/s": v_cs,
        "Transport to TT, s": t_transport_to_tt,
        "TT to XV Transport, s": t_tt_to_xv,
        "XV to CS Transport, s": t_xv_to_cs,
        "HX Outlet Reaches TALL, s": t_hx_to_tall,
        "TT Process Reaches TALL, s": t_tt_process_tall,
        "Trip Demand, s": t_trip_demand,
        "Effective Isolation, s": t_effective_isolation,
        "CS Limit Arrival if Not Isolated, s": t_cs_limit_arrival,
        "Screening PST Margin, s": margin,
        "Required Margin, s": case.required_margin_s,
        "Screening Pass": pass_flag,
    }
    return summary

# =============================================================================
# Plot and export functions
# =============================================================================
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
    ax.set_title(f"Dynamic PST Temperature Response - {case.case_name}")
    ax.grid(True, alpha=0.3)
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
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def make_screening_plot(case: PSTCase, screening: Dict):
    t_end = max(case.t_final, screening["CS Limit Arrival if Not Isolated, s"] * 1.25, 60)
    t = np.linspace(0, t_end, int(min(max(t_end, 60), 2000)) + 1)
    hx = exp_decay_temperature(t, case.T_eth_out_initial, case.T_eth_in, case.hx_cooling_tau_s)
    delay_process = screening["Transport to TT, s"] + case.ss_pipe_lag_s
    tt_process = np.where(t < delay_process, case.T_eth_out_initial, exp_decay_temperature(t - delay_process, case.T_eth_out_initial, case.T_eth_in, case.hx_cooling_tau_s))
    delay_measured = delay_process + case.TT_tau
    tt_measured = np.where(t < delay_measured, case.T_eth_out_initial, exp_decay_temperature(t - delay_measured, case.T_eth_out_initial, case.T_eth_in, case.hx_cooling_tau_s))

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.plot(t, hx, label="HX outlet")
    ax.plot(t, tt_process, label="TT process")
    ax.plot(t, tt_measured, label="TT measured")
    ax.axhline(case.TALL, linestyle="--", label=f"TALL = {case.TALL:g} °C")
    ax.axhline(case.T_CS_limit, linestyle="--", label=f"CS limit = {case.T_CS_limit:g} °C")
    ax.axvline(screening["Effective Isolation, s"], linestyle=":", label="Effective isolation")
    ax.axvline(screening["CS Limit Arrival if Not Isolated, s"], linestyle=":", label="CS limit arrival")
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Temperature, °C")
    ax.set_title("Screening PST Temperature Response")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def dataframe_to_excel_bytes(inputs: Dict, dynamic_summary_df: pd.DataFrame, screening_df: pd.DataFrame, ts: pd.DataFrame, sensitivity_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([inputs]).to_excel(writer, sheet_name="inputs", index=False)
        dynamic_summary_df.to_excel(writer, sheet_name="dynamic_summary", index=False)
        screening_df.to_excel(writer, sheet_name="screening_summary", index=False)
        sensitivity_df.to_excel(writer, sheet_name="sensitivity", index=False)
        ts.to_excel(writer, sheet_name="timeseries", index=False)
    output.seek(0)
    return output.getvalue()


def create_pdf_bytes(dynamic_summary_df: pd.DataFrame, screening_df: pd.DataFrame, temp_png: bytes, xv_png: bytes, screening_png: bytes) -> bytes:
    if not REPORTLAB_AVAILABLE:
        return b""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=14*mm, leftMargin=14*mm, topMargin=12*mm, bottomMargin=12*mm)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("DPSIM PST Calculation Report", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Low temperature protection of CS piping downstream of LP Ethylene Superheater.", styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Dynamic Model Summary", styles["Heading2"]))
    data = [list(dynamic_summary_df.columns)] + dynamic_summary_df.astype(str).values.tolist()
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Screening Summary", styles["Heading2"]))
    data2 = [list(screening_df.columns)] + screening_df.astype(str).values.tolist()
    tbl2 = Table(data2, repeatRows=1)
    tbl2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]))
    story.append(tbl2)
    story.append(Spacer(1, 10))

    for title, png in [("Dynamic Temperature Plot", temp_png), ("XV Closure Plot", xv_png), ("Screening Plot", screening_png)]:
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Image(io.BytesIO(png), width=180*mm, height=90*mm))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Engineering note: This is a screening / first-pass lumped dynamic model. Validate against HYSYS Dynamics / approved project data for formal SIS documentation.", styles["Italic"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# =============================================================================
# UI
# =============================================================================
st.title("🛡️ DPSIM - Merged PST Calculator")
st.markdown(
    """
    <div class="small-note">
    Scenario: LP Ethylene Superheater methanol heating-medium failure. This app merges the earlier lumped dynamic PST calculator with a quick screening / plug-flow PST module.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Model basis / caution", expanded=False):
    st.markdown(
        """
        - **Dynamic model:** first-pass lumped dynamic model using superheater metal/fluid thermal inertia, CS pipe wall/fluid thermal inertia and XV closure profile.
        - **Screening model:** exponential superheater cooldown + plug-flow residence time + TT/SIS/XV delay.
        - For formal SIS/PST validation, benchmark against **Aspen HYSYS Dynamics** or an approved dynamic simulation.
        - Use maximum ethylene flow, minimum inlet temperature, maximum TT lag, maximum XV stroke time and conservative heat gain assumptions unless justified.
        """
    )

st.sidebar.header("Input Data")

with st.sidebar:
    case_name = st.text_input("Case name", "Base Case")

    st.subheader("Temperatures")
    T_eth_in = st.number_input("Ethylene inlet temperature after failure, °C", value=-40.0, step=1.0)
    T_eth_out_initial = st.number_input("Initial superheater outlet temperature, °C", value=30.0, step=1.0)
    TALL = st.number_input("TALL trip setpoint, °C", value=0.0, step=1.0)
    T_CS_limit = st.number_input("CS pipe limiting temperature / MDMT basis, °C", value=-29.0, step=1.0)
    T_ambient = st.number_input("Ambient temperature, °C", value=30.0, step=1.0)

    st.subheader("Ethylene / Superheater")
    m_eth_kg_h = st.number_input("Ethylene mass flow, kg/h", min_value=0.001, value=31600.0, step=100.0)
    m_eth_normal = m_eth_kg_h / 3600.0
    Cp_eth = st.number_input("Ethylene Cp, J/kg.K", min_value=1.0, value=2200.0, step=100.0)
    rho_eth = st.number_input("Ethylene density, kg/m³", min_value=0.01, value=18.0, step=0.5)
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

    st.subheader("Piping / Screening Inputs")
    ss_length_to_tt_m = st.number_input("SS pipe length from SH outlet to TT, m", min_value=0.0, value=10.0, step=0.5)
    ss_pipe_id_m = st.number_input("SS pipe internal diameter, m", min_value=0.001, value=0.154, step=0.001, format="%.3f")
    tt_to_xv_m = st.number_input("Distance from TT to XV, m", min_value=0.0, value=1.0, step=0.5)
    xv_to_cs_m = st.number_input("Distance from XV seat to first CS point, m", min_value=0.0, value=1.0, step=0.5)
    cs_pipe_id_m = st.number_input("CS pipe internal diameter, m", min_value=0.001, value=0.154, step=0.001, format="%.3f")
    hx_cooling_tau_s = st.number_input("Screening HX cooling time constant, s", min_value=0.001, value=60.0, step=1.0)
    ss_pipe_lag_s = st.number_input("Screening SS pipe lag allowance, s", min_value=0.0, value=5.0, step=0.5)
    required_margin_s = st.number_input("Required screening margin, s", min_value=0.0, value=5.0, step=0.5)

    st.subheader("CS Pipe Dynamic Model")
    M_pipe_metal = st.number_input("CS pipe metal mass, kg", min_value=0.001, value=800.0, step=50.0)
    Cp_pipe_metal = st.number_input("CS pipe metal Cp, J/kg.K", min_value=1.0, value=500.0, step=50.0)
    M_pipe_fluid = st.number_input("CS pipe fluid holdup, kg", min_value=0.001, value=50.0, step=5.0)
    UA_pipe_fluid_wall = st.number_input("UA pipe fluid-to-wall, W/K", min_value=0.001, value=4000.0, step=500.0)
    UA_pipe_ambient = st.number_input("UA pipe-to-ambient, W/K", min_value=0.0, value=0.0, step=100.0)
    transport_delay_s = st.number_input("Dynamic transport delay to CS pipe, s", min_value=0.0, value=0.0, step=1.0)

    st.subheader("SIF / XV Timing")
    TT_tau = st.number_input("TT/thermowell lag, s", min_value=0.001, value=5.0, step=1.0)
    logic_delay = st.number_input("Logic solver delay, s", min_value=0.0, value=1.0, step=0.5)
    solenoid_delay = st.number_input("Solenoid delay, s", min_value=0.0, value=1.0, step=0.5)
    XV_stroke_time = st.number_input("XV stroke time, s", min_value=0.0, value=10.0, step=1.0)
    XV_leakage_fraction = st.number_input("XV leakage fraction after closure", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
    required_margin_fraction = st.number_input("Required dynamic margin fraction", min_value=0.0, max_value=1.0, value=0.10, step=0.05)

    st.subheader("Simulation")
    t_final = st.number_input("Final simulation time, s", min_value=1.0, value=600.0, step=60.0)
    max_step = st.number_input("Max integration step, s", min_value=0.001, value=0.25, step=0.05)

    run_button = st.button("Run Merged PST Calculation", type="primary", use_container_width=True)

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
    m_eth_kg_h=m_eth_kg_h,
    rho_eth=rho_eth,
    ss_length_to_tt_m=ss_length_to_tt_m,
    ss_pipe_id_m=ss_pipe_id_m,
    tt_to_xv_m=tt_to_xv_m,
    xv_to_cs_m=xv_to_cs_m,
    cs_pipe_id_m=cs_pipe_id_m,
    hx_cooling_tau_s=hx_cooling_tau_s,
    ss_pipe_lag_s=ss_pipe_lag_s,
    required_margin_s=required_margin_s,
    t_final=t_final,
    max_step=max_step,
    required_margin_fraction=required_margin_fraction,
)

if run_button:
    with st.spinner("Running dynamic and screening PST calculations..."):
        dynamic_result = simulate_case(case)
        try:
            screening_result = screening_calculation(case)
        except Exception as e:
            screening_result = {"Screening Error": str(e)}
        st.session_state["dynamic_result"] = dynamic_result
        st.session_state["screening_result"] = screening_result

if "dynamic_result" in st.session_state:
    result = st.session_state["dynamic_result"]
    screening = st.session_state.get("screening_result", {})
    summary = result["summary"]
    ts = result["timeseries"]
    case = result["case"]

    st.subheader("Calculation Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dynamic TALL Time", fmt_seconds(summary["TALL Time, s"]))
    c2.metric("Dynamic PST / CS Limit", fmt_seconds(summary["CS Limit Time / PST, s"]))
    c3.metric("Available After TALL", "N/A" if summary["Available Time After TALL, s"] is None else f"{summary['Available Time After TALL, s']:.2f} s")
    c4.metric("Required After TALL", f"{summary['Required Time After TALL, s']:.2f} s")

    if summary["Pass With Margin"] is True:
        st.markdown("<div class='pass-box'><b>Dynamic Model PASS:</b> Required response time is less than available time after TALL with selected margin.</div>", unsafe_allow_html=True)
    elif summary["Pass With Margin"] is False:
        st.markdown("<div class='fail-box'><b>Dynamic Model FAIL:</b> Required response time exceeds available time after TALL with selected margin.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='warn-box'><b>Dynamic Model Review:</b> CS limit or TALL was not reached within simulation duration. Review plots or extend simulation time.</div>", unsafe_allow_html=True)

    if "Screening Error" not in screening:
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Screening Trip Demand", fmt_seconds(screening["Trip Demand, s"]))
        c6.metric("Screening Isolation", fmt_seconds(screening["Effective Isolation, s"]))
        c7.metric("Screening CS Arrival", fmt_seconds(screening["CS Limit Arrival if Not Isolated, s"]))
        c8.metric("Screening Margin", fmt_seconds(screening["Screening PST Margin, s"]))
        if screening["Screening Pass"]:
            st.markdown("<div class='pass-box'><b>Screening Model PASS:</b> Screening PST margin meets the required margin.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='fail-box'><b>Screening Model FAIL / REVIEW:</b> Screening PST margin is below required margin.</div>", unsafe_allow_html=True)
    else:
        st.warning(f"Screening calculation was not completed: {screening['Screening Error']}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Dynamic Temperature Plot",
        "XV Plot",
        "Screening PST",
        "Sensitivity",
        "Time Series",
        "Report / Downloads",
    ])

    summary_df = pd.DataFrame([summary])
    screening_df = pd.DataFrame([screening]) if screening else pd.DataFrame()

    with tab1:
        fig_temp = make_temperature_plot(ts, case, summary)
        st.pyplot(fig_temp, use_container_width=True)
        st.dataframe(summary_df, use_container_width=True)

    with tab2:
        fig_xv = make_xv_plot(ts, case, summary)
        st.pyplot(fig_xv, use_container_width=True)

    with tab3:
        if "Screening Error" not in screening:
            fig_screen = make_screening_plot(case, screening)
            st.pyplot(fig_screen, use_container_width=True)
            st.dataframe(screening_df, use_container_width=True, hide_index=True)
        else:
            st.warning(screening["Screening Error"])
            fig_screen = None

    with tab4:
        st.markdown("### Screening sensitivity: HX cooling time constant vs XV stroke time")
        tau_values = np.linspace(10, 240, 8)
        xv_values = np.linspace(2, 30, 8)
        rows = []
        for tau in tau_values:
            row = {"HX tau, s": round(float(tau), 2)}
            for xv in xv_values:
                case_i = PSTCase(**asdict(case))
                case_i.hx_cooling_tau_s = float(tau)
                case_i.XV_stroke_time = float(xv)
                try:
                    res_i = screening_calculation(case_i)
                    row[f"XV {xv:.1f}s"] = round(res_i["Screening PST Margin, s"], 2)
                except Exception:
                    row[f"XV {xv:.1f}s"] = np.nan
            rows.append(row)
        sensitivity_df = pd.DataFrame(rows)
        st.dataframe(sensitivity_df, use_container_width=True, hide_index=True)

        fig_sens, ax_sens = plt.subplots(figsize=(10.5, 5))
        for xv in xv_values:
            ax_sens.plot(sensitivity_df["HX tau, s"], sensitivity_df[f"XV {xv:.1f}s"], marker="o", label=f"XV {xv:.1f}s")
        ax_sens.axhline(case.required_margin_s, linestyle="--", label="Required margin")
        ax_sens.set_xlabel("HX cooling time constant, s")
        ax_sens.set_ylabel("Screening PST margin, s")
        ax_sens.set_title("Sensitivity of PST Margin")
        ax_sens.grid(True, alpha=0.3)
        ax_sens.legend(ncol=2, fontsize=8)
        fig_sens.tight_layout()
        st.pyplot(fig_sens, use_container_width=True)

    with tab5:
        st.dataframe(ts, use_container_width=True, height=420)

    with tab6:
        temp_png = fig_to_png_bytes(fig_temp)
        xv_png = fig_to_png_bytes(fig_xv)
        if "Screening Error" not in screening:
            screening_png = fig_to_png_bytes(fig_screen)
        else:
            fig_blank, ax_blank = plt.subplots(figsize=(8, 3))
            ax_blank.text(0.5, 0.5, "Screening plot unavailable", ha="center", va="center")
            ax_blank.axis("off")
            screening_png = fig_to_png_bytes(fig_blank)

        excel_bytes = dataframe_to_excel_bytes(asdict(case), summary_df, screening_df, ts, sensitivity_df)
        csv_bytes = ts.to_csv(index=False).encode("utf-8")
        pdf_bytes = create_pdf_bytes(summary_df, screening_df, temp_png, xv_png, screening_png)

        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Download Excel Report", excel_bytes, file_name="dpsim_merged_pst_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        d2.download_button("Download Time Series CSV", csv_bytes, file_name="dpsim_pst_timeseries.csv", mime="text/csv", use_container_width=True)
        d3.download_button("Download Temperature Plot", temp_png, file_name="dpsim_pst_temperature_plot.png", mime="image/png", use_container_width=True)
        if pdf_bytes:
            d4.download_button("Download PDF Report", pdf_bytes, file_name="dpsim_pst_report.pdf", mime="application/pdf", use_container_width=True)
        else:
            d4.info("PDF unavailable")

        st.markdown("### Report-ready statement")
        st.info(
            f"For case '{case.case_name}', the dynamic model predicted TALL at {fmt_seconds(summary['TALL Time, s'])} and CS limit/PST at {fmt_seconds(summary['CS Limit Time / PST, s'])}. "
            f"The available time after TALL is {'N/A' if summary['Available Time After TALL, s'] is None else f'{summary['Available Time After TALL, s']:.2f} s'}, compared with required response time after TALL of {summary['Required Time After TALL, s']:.2f} s. "
            "The result shall be validated against approved project data, vendor TT/XV response data and rigorous dynamic simulation before formal SIS documentation."
        )
else:
    st.info("Enter inputs in the sidebar and click **Run Merged PST Calculation**.")

st.markdown('<div class="footer">Created by Dhawal Patel | DPSIM</div>', unsafe_allow_html=True)
