import io
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import math
import time
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from backend import (
    # LSTM
    build_fleet_ranking, explain_machine, get_model_performance,
    get_val_scatter_data, get_engine_rul_plot, get_test_scatter_data,
    get_ig_occlusion_data, get_comparison_df, get_max_cycle,
    get_test_performance, get_fleet_sensor_summary,
    get_engine_peak_snapshot, scaler,
    _feature_names_export, _healthy_mean_export, _healthy_std_export,
    # CNN
    CNN_AVAILABLE,
    build_fleet_ranking_cnn, explain_machine_cnn, get_cnn_activation_max,
    get_model_performance_cnn, get_test_performance_cnn,
    get_cnn_val_scatter_data, get_cnn_test_scatter_data,
    get_cnn_engine_rul_plot, get_lrp_ig_data_cnn,
)

st.set_page_config(layout="wide", page_icon="⚙️")

# ── Cached XAI helpers (ttl=10s — recompute at most once every 10 seconds) ───
@st.cache_data(ttl=10)
def _cached_ig_occlusion(machine_id, cycle_number):
    return get_ig_occlusion_data(machine_id, cycle_number)

@st.cache_data(ttl=10)
def _cached_lrp_ig_cnn(machine_id, cycle_number):
    return get_lrp_ig_data_cnn(machine_id, cycle_number)

# ── Session state defaults ────────────────────────────────────────────────────
_t = time.time()
for _k, _v in [
    ("current_cycle",    50),
    ("last_update",      _t),
    ("start_time",       _t),
    ("live_mode",        True),
    ("hold_live",        False),
    ("prev_halt",        False),
    ("light_mode",       False),
    ("prev_high_engines", set()),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.prev_halt and not st.session_state.hold_live:
    halted_at = st.session_state.current_cycle
    st.session_state.start_time = time.time() - (halted_at - 50) * 10

elapsed    = time.time() - st.session_state.start_time
live_cycle = 50 + int(elapsed // 10)
MAX_CYCLE  = get_max_cycle()

if st.session_state.live_mode and not st.session_state.hold_live:
    st.session_state.current_cycle = min(live_cycle, MAX_CYCLE)

if not st.session_state.hold_live:
    st_autorefresh(interval=10000, key="live_refresh")

# ── Plotly template (respects light/dark mode toggle) ─────────────────────────
_PT = "plotly_white" if st.session_state.light_mode else "plotly_dark"

# ── Sidebar: model selector ────────────────────────────────────────────────────
st.sidebar.title("⚙️ Predictive Maintenance")
selected_model = st.sidebar.radio(
    "🤖 Select Model",
    ["LSTM Model", "CNN Model"],
    key="model_selector",
)
if selected_model == "CNN Model" and not CNN_AVAILABLE:
    st.sidebar.warning(
        "CNN model not available. Train the CNN in the notebook first and save "
        "the weights to `data/cnn_model.pth`."
    )

# ── Early fleet computation for sidebar health badges ─────────────────────────
if selected_model == "LSTM Model":
    _badge_fleet = build_fleet_ranking(st.session_state.current_cycle)
elif CNN_AVAILABLE:
    _badge_fleet = build_fleet_ranking_cnn(st.session_state.current_cycle)
else:
    _badge_fleet = pd.DataFrame()

if len(_badge_fleet) > 0:
    _nb_high = int((_badge_fleet["Risk_Label"] == "HIGH").sum())
    _nb_med  = int((_badge_fleet["Risk_Label"] == "MEDIUM").sum())
    _nb_low  = int((_badge_fleet["Risk_Label"] == "LOW").sum())

    # ── Detect engines that just entered the HIGH-risk zone ────────────────
    # Only build the alert data here; the components.html() call is at the
    # very bottom of the script so the Streamlit widget tree stays stable
    # regardless of whether there are new HIGH engines this rerun.
    _cur_high = set(_badge_fleet[_badge_fleet["Risk_Label"] == "HIGH"]["Machine_ID"].tolist())
    _new_high = _cur_high - st.session_state.prev_high_engines
    _alert_entries_js = ""
    _show_alert = False
    if _new_high:
        _show_alert = True
        _entries_html = ""
        for _eid in sorted(_new_high):
            _row = _badge_fleet[_badge_fleet["Machine_ID"] == _eid].iloc[0]
            _rul = round(float(_row["Predicted_RUL"]))
            _entries_html += (
                "<div style='margin-bottom:12px;padding-bottom:12px;"
                "border-bottom:1px solid rgba(255,255,255,0.2);'>"
                f"<div style='font-size:17px;font-weight:700;margin-bottom:4px;'>"
                f"🚨 Engine {_eid} entered HIGH RISK zone</div>"
                f"<div style='font-size:13px;opacity:0.85;'>"
                f"Predicted RUL: <strong>{_rul} cycles</strong></div>"
                "</div>"
            )
        _alert_entries_js = _entries_html.replace("\\", "\\\\").replace("'", "\\'")
    st.session_state.prev_high_engines = _cur_high
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""<div style="padding:10px 4px 4px;">
<p style="margin:0 0 8px;font-size:10px;letter-spacing:1.2px;
color:rgba(255,255,255,0.4);text-transform:uppercase;font-weight:600;">
Fleet Status — Cycle {st.session_state.current_cycle}</p>
<div style="display:flex;flex-wrap:wrap;gap:5px;">
<span style="background:#e63946;color:#fff;padding:4px 12px;border-radius:20px;
font-size:12px;font-weight:700;">🔴 {_nb_high} Critical</span>
<span style="background:#3a7bd5;color:#fff;padding:4px 12px;border-radius:20px;
font-size:12px;font-weight:700;">🔵 {_nb_med} Moderate</span>
<span style="background:#22c55e;color:#fff;padding:4px 12px;border-radius:20px;
font-size:12px;font-weight:700;">🟢 {_nb_low} Normal</span>
</div></div>""",
        unsafe_allow_html=True,
    )

# ── Sidebar: Dark / Light mode toggle ─────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.toggle("☀️ Light Mode", key="light_mode")

# ── Sidebar: Glossary ─────────────────────────────────────────────────────────
with st.sidebar.expander("📖 Glossary"):
    _GLOSSARY = [
        ("RUL",        "Remaining Useful Life — estimated cycles before engine failure"),
        ("HIGH risk",  "Predicted RUL < 40 cycles — immediate maintenance required"),
        ("MEDIUM risk","40 ≤ Predicted RUL < 80 cycles — schedule maintenance soon"),
        ("LOW risk",   "Predicted RUL ≥ 80 cycles — engine operating normally"),
    ]
    for _term, _defn in _GLOSSARY:
        st.markdown(f"**{_term}** — {_defn}")

# ── Anti-flicker CSS — always injected (fixes blinking on Streamlit Cloud) ────
st.markdown("""<style>
[data-stale="true"] { opacity: 1 !important; visibility: visible !important; }
[data-testid="stSkeleton"] { display: none !important; }
</style>""", unsafe_allow_html=True)

# ── Light mode CSS injection ───────────────────────────────────────────────────
if st.session_state.light_mode:
    st.markdown("""<style>
.stApp { background-color: #f4f6fa !important; }
section[data-testid="stSidebar"] > div { background-color: #e8eaf0 !important; }
.stMarkdown p, .stMarkdown li, label { color: #1a1a2e !important; }
h1, h2, h3, h4 { color: #1a1a2e !important; }
.stTabs [data-baseweb="tab"] { background-color: #e0e4ef !important; color: #1a1a2e !important; }
</style>""", unsafe_allow_html=True)

# ── Main title + live controls ─────────────────────────────────────────────────
st.title("⚙️ Predictive Maintenance Dashboard")

_ctrl1, _ctrl2, _ctrl3 = st.columns([1, 1, 1])
with _ctrl1:
    st.toggle("🟢 Live Mode", key="live_mode")
with _ctrl2:
    st.metric("Current Cycle", st.session_state.current_cycle)
with _ctrl3:
    st.toggle("⏸ Halt", key="hold_live")

if st.session_state.live_mode:
    st.caption("⏸ Halted" if st.session_state.hold_live else "🟢 Live")
else:
    st.caption("📊 Historical")

st.session_state.prev_halt = st.session_state.hold_live

# ── Shared helpers ─────────────────────────────────────────────────────────────
_RISK_COLORS = {"HIGH": "#e63946", "MEDIUM": "#3a7bd5", "LOW": "#22c55e"}


def _tooltip(text, tip):
    """Inline HTML tooltip — hover the term to see the definition."""
    return (
        f'<span title="{tip}" style="border-bottom:1px dotted rgba(255,255,255,0.45);'
        f'cursor:help;padding-bottom:1px;">{text} <sup style="font-size:9px;'
        f'opacity:0.6;">ℹ</sup></span>'
    )


def highlight_rows_fleet(row):
    """Pandas Styler — color full row by Risk_Label."""
    risk = row.get("Risk_Label", "")
    if risk == "HIGH":
        bg = "background-color:rgba(230,57,70,0.28)"      # red
    elif risk == "MEDIUM":
        bg = "background-color:rgba(58,123,213,0.28)"     # clear royal blue
    elif risk == "LOW":
        bg = "background-color:rgba(34,197,94,0.22)"      # clear emerald green
    else:
        bg = ""
    return [bg] * len(row)


# keep legacy alias used in home tab
highlight_rows_home = highlight_rows_fleet


def _cycle_slider(tab_key):
    """Return cycle number from slider or current cycle."""
    min_cycle = 50
    max_cycle = st.session_state.current_cycle
    if max_cycle <= min_cycle:
        st.info("📡 Streaming data… waiting for enough cycles")
        return max_cycle
    selected = st.slider(
        "Explore Past Cycles",
        min_value=min_cycle,
        max_value=max_cycle,
        value=max_cycle,
        key=tab_key,
        disabled=st.session_state.live_mode,
    )
    return selected


def _render_risk_rul_cards(risk, rul, prefix=""):
    rc = _RISK_COLORS.get(risk, "#888")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""<div style="border:2px solid {rc};border-radius:10px;padding:14px;
text-align:center;background:rgba(255,255,255,0.03);">
<p style="margin:0;color:rgba(255,255,255,0.5);font-size:12px;">Predicted Risk</p>
<h2 style="margin:4px 0 0;color:{rc};font-weight:700;">{risk}</h2></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div style="border:2px solid #888;border-radius:10px;padding:14px;
text-align:center;background:rgba(255,255,255,0.03);">
<p style="margin:0;color:rgba(255,255,255,0.5);font-size:12px;">Predicted RUL</p>
<h2 style="margin:4px 0 0;color:#eee;font-weight:700;">{rul:.1f} cycles</h2></div>""",
            unsafe_allow_html=True,
        )
    st.markdown("")


def _render_ig_heatmap_bar(ig_arr, feature_names_list, cycles, sensor_values,
                            healthy_vals, healthy_stds, hm_key, bar_key,
                            colorscale="RdYlGn", zmid=0):
    """Render (heatmap | bar) side-by-side. ig_arr: (50,24) raw."""
    z = ig_arr.T  # → (24, 50)
    nf = min(z.shape[0], sensor_values.shape[1], len(feature_names_list), len(healthy_vals))
    nc = min(z.shape[1], sensor_values.shape[0], len(cycles))
    z      = z[:nf, :nc]
    fnames = feature_names_list[:nf]
    hvals  = healthy_vals[:nf]
    hstds  = healthy_stds[:nf]
    cyc    = cycles[:nc]
    svals  = sensor_values[:nc, :]

    z_abs      = np.abs(z)
    z_col_norm = z_abs / (z_abs.sum(axis=0, keepdims=True) + 1e-8)
    cum_frac   = z_abs.sum(axis=1) / (z_abs.sum() + 1e-8)

    hover_text = []
    for i in range(nf):
        row_h = []
        for j in range(nc):
            row_h.append(
                f"<b>{fnames[i]}</b><br>"
                f"Cycle: {cyc[j]}<br>"
                f"IG Attribution: {z[i, j]:.4g} "
                f"<i>(positive=RUL up; negative=RUL down)</i><br>"
                f"Sensor importance this cycle: {z_col_norm[i, j]*100:.1f}%<br>"
                f"Current value: {svals[j, i]:.6g}<br>"
                f"Healthy baseline: {hvals[i]:.6g}"
            )
        hover_text.append(row_h)

    col_hm, col_bar = st.columns([3, 2])
    with col_hm:
        fig_h = go.Figure(data=go.Heatmap(
            z=z, x=cyc, y=fnames,
            text=hover_text, hoverinfo="text",
            colorscale=colorscale,
            zmid=zmid if zmid is not None else None,
            colorbar=dict(title="IG Attribution"),
        ))
        fig_h.update_layout(
            template=_PT, height=650,
            title="IG Attribution Heatmap (signed · red=degrades RUL · green=improves RUL)",
        )
        st.plotly_chart(fig_h, use_container_width=True, key=hm_key)

    with col_bar:
        sort_idx     = np.argsort(cum_frac)
        sorted_names = [fnames[i] for i in sort_idx]
        sorted_frac  = cum_frac[sort_idx]
        fig_bar = go.Figure(go.Bar(
            x=sorted_frac, y=sorted_names, orientation="h",
            marker_color="salmon",
            hovertemplate="%{y}<br>Importance share: %{x:.2%}<extra></extra>",
        ))
        fig_bar.update_layout(
            template=_PT, height=650,
            title="Cumulative Sensor Importance<br>(fractions sum to 1)",
            xaxis_title="Share of Total Importance",
            xaxis_tickformat=".0%",
            yaxis_title="",
            margin=dict(l=10, r=10, t=60, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True, key=bar_key)


def _render_sensor_deepdive(fleet_df, selected_engine, result_cache, cycle_number,
                             explain_fn, prefix="lstm"):
    """Sensor Deep Dive section, shared between LSTM and CNN tabs."""
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("## 🔬 Sensor Deep Dive")
    st.caption(
        "Pick any engine and compare its last-50-cycle sensor readings against the healthy "
        "population baseline. The dashed line is the healthy mean; the green band is ±1 std "
        "(68% of healthy engine readings fall inside it — values outside suggest degradation)."
    )

    _dd_col1, _dd_col2 = st.columns([1, 3])
    with _dd_col1:
        engine_list = fleet_df["Machine_ID"].tolist()
        default_idx = (engine_list.index(selected_engine)
                       if selected_engine in engine_list else 0)
        dd_engine = st.selectbox(
            "Engine to inspect",
            engine_list,
            index=default_idx,
            key=f"{prefix}_dd_engine_select",
        )

    dd_res = result_cache.get(dd_engine)
    if dd_res is None:
        dd_res, _ = explain_fn(dd_engine, cycle_number, fleet_df, top_k=3)
    if dd_res is None:
        st.warning("Not enough data for this engine.")
        return

    # Support both LSTM (returns "ig") and CNN (returns "lrp")
    _attr_key = "ig" if "ig" in dd_res else "lrp"
    dd_z     = np.array(dd_res[_attr_key]).T
    dd_fcols = dd_res["feature_names"]
    dd_hvals = dd_res["healthy_values"]
    dd_hstds = dd_res["healthy_std"]
    dd_svals = dd_res["sensor_values"]
    dd_cycs  = dd_res["cycles"]

    dd_nf = min(dd_z.shape[0], dd_svals.shape[1], len(dd_fcols), len(dd_hvals))
    dd_nc = min(dd_z.shape[1], dd_svals.shape[0], len(dd_cycs))
    dd_z     = dd_z[:dd_nf, :dd_nc]
    dd_fcols = dd_fcols[:dd_nf]
    dd_hvals = dd_hvals[:dd_nf]
    dd_hstds = dd_hstds[:dd_nf]
    dd_svals = dd_svals[:dd_nc, :]
    dd_cycs  = dd_cycs[:dd_nc]

    dd_z_abs    = np.abs(dd_z)
    dd_cum_frac = dd_z_abs.sum(axis=1) / (dd_z_abs.sum() + 1e-8)
    top3_idx    = np.argsort(dd_cum_frac)[::-1][:3]
    top3_default = [dd_fcols[i] for i in top3_idx]
    _attr_label  = "IG" if _attr_key == "ig" else "LRP"

    with _dd_col2:
        selected_sensors = st.multiselect(
            f"Select sensors to inspect (default = top 3 by {_attr_label} importance)",
            options=list(dd_fcols),
            default=[s for s in top3_default if s in dd_fcols],
            key=f"{prefix}_sensor_deepdive_select",
        )

    show_all = st.checkbox("Show full report (all sensors)", key=f"{prefix}_sensor_deepdive_all")
    sensors_to_plot = list(dd_fcols) if show_all else selected_sensors

    if sensors_to_plot:
        n_cols = 2
        n_rows = math.ceil(len(sensors_to_plot) / n_cols)
        v_spacing = min(0.12, 0.9 / max(n_rows - 1, 1))
        fig_dd = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=sensors_to_plot,
            vertical_spacing=v_spacing,
            horizontal_spacing=0.10,
        )
        for idx, sname in enumerate(sensors_to_plot):
            if sname not in list(dd_fcols):
                continue
            fi     = list(dd_fcols).index(sname)
            vals   = dd_svals[:dd_nc, fi]
            h_mean = dd_hvals[fi]
            h_std  = dd_hstds[fi]
            row_n  = idx // n_cols + 1
            col_n  = idx % n_cols + 1

            fig_dd.add_trace(go.Scatter(
                x=list(dd_cycs), y=vals.tolist(),
                mode="lines", name="Current",
                line=dict(color="royalblue", width=2),
                showlegend=(idx == 0), legendgroup="current",
            ), row=row_n, col=col_n)
            fig_dd.add_trace(go.Scatter(
                x=[dd_cycs[0], dd_cycs[-1]], y=[h_mean, h_mean],
                mode="lines", name="Healthy mean",
                line=dict(color="lime", dash="dash", width=1.5),
                showlegend=(idx == 0), legendgroup="healthy",
            ), row=row_n, col=col_n)
            fig_dd.add_trace(go.Scatter(
                x=list(dd_cycs) + list(dd_cycs[::-1]),
                y=[h_mean + h_std] * dd_nc + [h_mean - h_std] * dd_nc,
                fill="toself", fillcolor="rgba(0,200,0,0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Healthy ±1 std",
                showlegend=(idx == 0), legendgroup="std",
            ), row=row_n, col=col_n)

        fig_dd.update_layout(
            template=_PT,
            height=max(400, n_rows * 280),
            title_text=f"Engine {dd_engine} — Sensor Values vs Healthy Baseline",
            margin=dict(t=60, b=60),
        )
        st.plotly_chart(fig_dd, use_container_width=True, key=f"{prefix}_sensor_deepdive_chart")
    else:
        st.info("Select at least one sensor above.")


def _render_maintenance_controls(fleet_df, prefix="lstm"):
    """Maintenance Done button + history table."""
    maintenance_engine = st.selectbox(
        "Select Engine for Maintenance",
        fleet_df["Machine_ID"].unique(),
        key=f"{prefix}_maintenance_engine",
    )
    if st.button("🔧 Maintenance Done", key=f"{prefix}_maintenance_btn"):
        if f"maintenance_history_{maintenance_engine}" not in st.session_state:
            st.session_state[f"maintenance_history_{maintenance_engine}"] = []
        st.session_state[f"maintenance_history_{maintenance_engine}"].append(
            st.session_state.current_cycle
        )
        st.success(f"Maintenance applied to Engine {maintenance_engine}")

    all_records = []
    for key in st.session_state:
        if key.startswith("maintenance_history_"):
            mid = int(key.split("_")[-1])
            for i, c in enumerate(st.session_state[key]):
                all_records.append({"Machine_ID": mid, "Maintenance Event": i + 1, "Cycle": c})
    if all_records:
        st.markdown("### 🛠 Maintenance History")
        st.dataframe(pd.DataFrame(all_records), use_container_width=True)


def _metric_card(col, label, value, icon, border_color, bg_color, sublabel=""):
    """Render a styled metric card inside a column."""
    col.markdown(
        f"""<div style="border-left:5px solid {border_color};border-radius:10px;
padding:16px 18px;background:{bg_color};margin:4px 0 12px 0;">
<p style="margin:0 0 4px;font-size:11px;color:rgba(255,255,255,0.55);
letter-spacing:0.5px;text-transform:uppercase;">{icon}&nbsp; {label}</p>
<h2 style="margin:0;color:#ffffff;font-weight:700;font-size:1.9rem;">{value}</h2>
{"<p style='margin:4px 0 0;font-size:11px;color:rgba(255,255,255,0.4);'>"+sublabel+"</p>" if sublabel else ""}
</div>""",
        unsafe_allow_html=True,
    )


# ── Engineer Excel export builder ─────────────────────────────────────────────
def _build_engineer_excel(cycle_number, use_cnn=False):
    """Build a 5-sheet Excel workbook for the maintenance engineer report.

    Sheets
    ------
    1. Fleet Overview      — Machine_ID, Cycle, Predicted_RUL, Risk_Label, Predicted_Failure_Cycle
    2. Mean Sensor Values  — 50-cycle window average per engine
    3. Last Cycle Values   — raw sensor values at the most-recent observed cycle (no Peak_Cycle column)
    4. Sensor Anomalies    — sensors deviating > 2σ from the healthy baseline
    5. Healthy Baseline    — Sensor, Healthy_Mean, Healthy_Std
    """
    buf = io.BytesIO()

    # ── Sheet 1: Fleet Overview ───────────────────────────────────────────────
    _fl = build_fleet_ranking_cnn(cycle_number) if use_cnn else build_fleet_ranking(cycle_number)
    overview_df = _fl[["Machine_ID", "Predicted_RUL", "Risk_Label"]].copy()
    overview_df = overview_df.sort_values("Predicted_RUL").reset_index(drop=True)
    overview_df.insert(0, "Cycle", cycle_number)
    overview_df["Predicted_Failure_Cycle"] = (
        overview_df["Predicted_RUL"] + cycle_number
    ).round(0).astype(int)

    # ── Sheet 2: Mean Sensor Values ───────────────────────────────────────────
    mean_df = get_fleet_sensor_summary(cycle_number, use_cnn=use_cnn)

    # ── Sheet 3: Last Cycle Values (drop Peak_Cycle) ──────────────────────────
    peak_df = get_engine_peak_snapshot(cycle_number)
    last_cycle_df = peak_df.drop(columns=["Peak_Cycle"], errors="ignore")

    # ── Sheet 4: Sensor Anomalies (>2σ from healthy) ──────────────────────────
    fn = _feature_names_export
    hm = _healthy_mean_export
    hs = _healthy_std_export

    anomaly_rows = []
    for _, row in mean_df.iterrows():
        mid = row["Machine_ID"]
        for i, fname in enumerate(fn):
            col_name = f"mean_{fname}"
            if col_name in row.index and hs[i] > 0:
                val = row[col_name]
                z   = abs(val - hm[i]) / hs[i]
                if z > 2.0:
                    anomaly_rows.append({
                        "Machine_ID":   mid,
                        "Sensor":       fname,
                        "Current_Mean": round(float(val),    3),
                        "Healthy_Mean": round(float(hm[i]),  3),
                        "Healthy_Std":  round(float(hs[i]),  3),
                        "Z_Score":      round(float(z),      2),
                        "Status":       "⚠ Anomaly" if z > 3 else "⚡ Warning",
                    })

    anomaly_df = (
        pd.DataFrame(anomaly_rows)
        if anomaly_rows
        else pd.DataFrame(columns=[
            "Machine_ID", "Sensor", "Current_Mean", "Healthy_Mean",
            "Healthy_Std", "Z_Score", "Status",
        ])
    )

    # ── Sheet 5: Healthy Baseline ─────────────────────────────────────────────
    baseline_df = pd.DataFrame({
        "Sensor":       fn,
        "Healthy_Mean": np.round(hm, 4),
        "Healthy_Std":  np.round(hs, 4),
    })

    # ── Write workbook ────────────────────────────────────────────────────────
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        overview_df.to_excel(   writer, sheet_name="Fleet Overview",      index=False)
        mean_df.to_excel(       writer, sheet_name="Mean Sensor Values",   index=False)
        last_cycle_df.to_excel( writer, sheet_name="Last Cycle Values",    index=False)
        anomaly_df.to_excel(    writer, sheet_name="Sensor Anomalies",     index=False)
        baseline_df.to_excel(   writer, sheet_name="Healthy Baseline",     index=False)

    buf.seek(0)
    return buf


# ── Health progress bar (replaces gauge) ──────────────────────────────────────
def _render_health_bar(health_score, prefix):
    bar_color = (
        "#2a9d8f" if health_score > 66
        else ("#e9c46a" if health_score > 33 else "#e63946")
    )
    bar_icon  = "✅" if health_score > 66 else ("⚠️" if health_score > 33 else "🔴")
    st.markdown(
        f"""<div style="background:rgba(255,255,255,0.05);border-radius:12px;
padding:20px 24px;margin:16px 0 20px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <span style="color:rgba(255,255,255,0.75);font-size:14px;font-weight:600;">
      {bar_icon} Fleet Health Score</span>
    <span style="color:{bar_color};font-weight:700;font-size:2.2rem;line-height:1;">
      {health_score:.1f}%</span>
  </div>
  <div style="background:rgba(255,255,255,0.12);border-radius:8px;height:22px;overflow:hidden;">
    <div style="background:linear-gradient(90deg,{bar_color},{bar_color}bb);
         width:{health_score}%;height:100%;border-radius:8px;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:7px;">
    <span style="color:#e63946;font-size:10px;opacity:0.7;">Critical (0%)</span>
    <span style="color:#e9c46a;font-size:10px;opacity:0.7;">Moderate (33%)</span>
    <span style="color:#2a9d8f;font-size:10px;opacity:0.7;">Healthy (100%)</span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


# ── Home tab ──────────────────────────────────────────────────────────────────
def _render_home_tab(fleet_df, prefix="lstm", curve_fn=None):
    """Shared Home tab body."""
    if curve_fn is None:
        curve_fn = get_engine_rul_plot if prefix == "lstm" else get_cnn_engine_rul_plot

    # ── Animated KPIs with delta vs prev cycle ────────────────────────────────
    st.markdown("## 📊 Fleet Summary")
    prev_cycle = max(50, st.session_state.current_cycle - 1)
    if prefix == "lstm":
        prev_fleet = build_fleet_ranking(prev_cycle)
    else:
        prev_fleet = build_fleet_ranking_cnn(prev_cycle) if CNN_AVAILABLE else pd.DataFrame()

    curr_high    = int((fleet_df["Risk_Label"] == "HIGH").sum())
    prev_high    = int((prev_fleet["Risk_Label"] == "HIGH").sum()) if len(prev_fleet) > 0 else curr_high
    delta_high   = curr_high - prev_high

    curr_avg_rul = int(fleet_df["Predicted_RUL"].mean())
    prev_avg_rul = int(prev_fleet["Predicted_RUL"].mean()) if len(prev_fleet) > 0 else curr_avg_rul
    delta_rul    = curr_avg_rul - prev_avg_rul

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Running Engines", len(fleet_df))
    col2.metric(
        "High Risk Engines", curr_high,
        delta=delta_high if delta_high != 0 else None,
        delta_color="inverse",
        help="↑ more critical engines since last cycle is bad (shown red)",
    )
    col3.metric(
        "Average RUL", f"{curr_avg_rul} cycles",
        delta=f"{delta_rul:+d} cycles" if delta_rul != 0 else None,
        help="Change in fleet average Remaining Useful Life since last cycle",
    )

    # ── Top 10 critical bar chart ─────────────────────────────────────────────
    st.markdown("## Most Critical Engines")
    top10 = fleet_df.sort_values("Predicted_RUL").head(10)
    # Color by risk level — hex codes are more reliable across templates
    colors = [
        "#e63946" if r == "HIGH" else "#3a7bd5" if r == "MEDIUM" else "#22c55e"
        for r in top10["Risk_Label"]
    ]
    fig_bar = go.Figure(go.Bar(
        x=top10["Machine_ID"].astype(str),
        y=top10["Predicted_RUL"],
        marker_color=colors,
        hovertemplate="Engine %{x}<br>RUL: %{y:.0f} cycles<extra></extra>",
    ))
    fig_bar.update_layout(
        title="Top 10 Critical Engines",
        xaxis_title="Engine ID", yaxis_title="Predicted RUL",
        template=_PT, showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True, key=f"{prefix}_home_bar")

    # ── Emergency cards ───────────────────────────────────────────────────────
    high_engines = fleet_df[fleet_df["Risk_Label"] == "HIGH"]
    if not high_engines.empty:
        st.markdown("### 🚨 EMERGENCY — High Risk Engines Require Immediate Attention")
        card_cols = st.columns(min(len(high_engines), 4))
        for idx, (_, eng_row) in enumerate(high_engines.iterrows()):
            with card_cols[idx % 4]:
                st.markdown(
                    f"""<div style="border:2px solid #cc0000;border-radius:10px;
padding:12px;background:rgba(180,0,0,0.18);text-align:center;margin-bottom:8px;">
<p style="margin:0;font-size:11px;color:#ff6b6b;font-weight:600;letter-spacing:1px;">
⚠ EMERGENCY</p>
<h3 style="margin:4px 0;color:#ff4444;">Engine {int(eng_row['Machine_ID'])}</h3>
<p style="margin:0;font-size:13px;color:#ffaaaa;">
RUL: <b>{eng_row['Predicted_RUL']:.0f} cycles</b></p>
<p style="margin:2px 0 0;font-size:11px;color:#ff6b6b;">HIGH RISK</p>
</div>""",
                    unsafe_allow_html=True,
                )

    # ── Color-coded fleet table ──────────────────────────────────────────────
    styled_table = top10.style.apply(highlight_rows_fleet, axis=1)
    st.dataframe(styled_table, use_container_width=True, hide_index=True)


# ── Operations Manager tab ─────────────────────────────────────────────────────
def _render_ops_tab(fleet_df, prefix="lstm"):
    """Shared Operations Manager tab body."""
    st.markdown("## 📈 Fleet Overview")

    # KPI cards row
    col1, col2, col3 = st.columns(3)
    _metric_card(col1, "Active Engines",    str(len(fleet_df)),
                 "🏭", "#2a9d8f", "rgba(42,157,143,0.12)")
    _metric_card(col2, "Average RUL",
                 f"{int(fleet_df['Predicted_RUL'].mean())} cycles",
                 "⏱️", "#457b9d", "rgba(69,123,157,0.12)")
    _metric_card(col3, "High Risk Engines",
                 str(int((fleet_df["Risk_Label"] == "HIGH").sum())),
                 "🔴", "#e63946", "rgba(230,57,70,0.12)")

    # ── Fleet table with color-coded rows ────────────────────────────────────
    st.markdown("## 🚨 Fleet Engine Status")
    st.markdown(
        'Color key:&nbsp;'
        '<span style="background:#e63946;color:#fff;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;">HIGH</span>&nbsp;'
        '<span style="background:#3a7bd5;color:#fff;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;">MEDIUM</span>&nbsp;'
        '<span style="background:#22c55e;color:#fff;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;">LOW</span>',
        unsafe_allow_html=True,
    )
    n_all = len(fleet_df)
    num_engines = st.slider(
        "Engines to display", 1, n_all, n_all, key=f"{prefix}_ops_top_k"
    )
    top_engines = fleet_df.sort_values("Predicted_RUL").head(num_engines).copy()
    rul_max_ops = fleet_df["Predicted_RUL"].max()
    top_engines["RUL Progress"] = (
        top_engines["Predicted_RUL"] / max(rul_max_ops, 1) * 100
    ).round(1)

    styled_ops = top_engines.style.apply(highlight_rows_fleet, axis=1)
    st.dataframe(
        styled_ops,
        column_config={
            "RUL Progress": st.column_config.ProgressColumn(
                "RUL Remaining", min_value=0, max_value=100, format="%.0f%%"
            ),
            "Risk_Label": st.column_config.TextColumn("Risk"),
        },
        use_container_width=True,
        hide_index=True,
    )

    # ── Export: Manager ───────────────────────────────────────────────────────
    _export_mgr = (
        fleet_df[["Machine_ID", "Predicted_RUL", "Risk_Label"]]
        .sort_values("Predicted_RUL")
        .copy()
    )
    _export_mgr.insert(0, "Cycle", st.session_state.current_cycle)
    st.download_button(
        "📥 Export Fleet Report (CSV)",
        _export_mgr.to_csv(index=False),
        file_name=f"fleet_report_cycle_{st.session_state.current_cycle}.csv",
        mime="text/csv",
        key=f"{prefix}_ops_export",
        help="Downloads Machine ID, Predicted RUL and Risk Level for all engines",
    )

    # ── Emergency cards ───────────────────────────────────────────────────────
    high_engines_ops = fleet_df[fleet_df["Risk_Label"] == "HIGH"]
    if not high_engines_ops.empty:
        st.markdown("### 🚨 EMERGENCY — High Risk Engines Require Immediate Attention")
        card_cols_ops = st.columns(min(len(high_engines_ops), 4))
        for idx, (_, eng_row) in enumerate(high_engines_ops.iterrows()):
            with card_cols_ops[idx % 4]:
                st.markdown(
                    f"""<div style="border:2px solid #cc0000;border-radius:10px;
padding:12px;background:rgba(180,0,0,0.18);text-align:center;margin-bottom:8px;">
<p style="margin:0;font-size:11px;color:#ff6b6b;font-weight:600;letter-spacing:1px;">
⚠ EMERGENCY</p>
<h3 style="margin:4px 0;color:#ff4444;">Engine {int(eng_row['Machine_ID'])}</h3>
<p style="margin:0;font-size:13px;color:#ffaaaa;">
RUL: <b>{eng_row['Predicted_RUL']:.0f} cycles</b></p>
<p style="margin:2px 0 0;font-size:11px;color:#ff6b6b;">HIGH RISK</p>
</div>""",
                    unsafe_allow_html=True,
                )

    # ── Fleet Risk Distribution ───────────────────────────────────────────────
    st.markdown("## 📊 Fleet Risk Distribution")
    risk_counts  = fleet_df["Risk_Label"].value_counts()
    n_high_ops   = int(risk_counts.get("HIGH", 0))
    health_score = round(100 * (1 - n_high_ops / max(len(fleet_df), 1)), 1)

    # Progress bar (replaces gauge)
    _render_health_bar(health_score, prefix)

    # Pie chart
    labels = ["HIGH", "MEDIUM", "LOW"]
    values = [risk_counts.get(l, 0) for l in labels]
    colors = ["rgb(230,57,70)", "rgb(69,123,157)", "rgb(42,157,143)"]
    fig_pie = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.4,
        marker=dict(colors=colors),
        textinfo="label+percent+value",
        textfont=dict(size=14),
    )])
    fig_pie.update_layout(
        template=_PT, height=520,
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_pie, use_container_width=True, key=f"{prefix}_ops_pie")

    # ── Collapsible XAI Insights (ops manager) ───────────────────────────────
    with st.expander("🔬 Advanced: XAI Insights & Model Explanation"):
        st.markdown(
            f"### {_tooltip('What drives these risk predictions?', 'Explainability methods reveal which sensor signals push the RUL prediction up or down')}",
            unsafe_allow_html=True,
        )
        st.markdown(
            """The model assigns risk categories based on predicted **RUL**:
- 🔴 **HIGH** — RUL < 40 cycles (immediate maintenance)
- 🔵 **MEDIUM** — 40 ≤ RUL < 80 cycles (schedule maintenance)
- 🟢 **LOW** — RUL ≥ 80 cycles (normal operation)"""
        )
        st.markdown("---")
        st.markdown("#### Fleet Risk Summary")
        _risk_counts_detail = fleet_df["Risk_Label"].value_counts().reset_index()
        _risk_counts_detail.columns = ["Risk Level", "Engine Count"]
        _risk_counts_detail["% of Fleet"] = (
            _risk_counts_detail["Engine Count"] / len(fleet_df) * 100
        ).round(1).astype(str) + "%"
        st.dataframe(_risk_counts_detail, use_container_width=True, hide_index=True)
        st.info(
            "💡 For detailed sensor-level XAI analysis, switch to the "
            "**🔧 Maintenance Engineer** tab."
        )


# ── Model Performance tabs ─────────────────────────────────────────────────────
def _render_perf_tab_lstm(fleet_df):
    """Model Performance tab — LSTM."""
    perf_tab1, perf_tab2, perf_tab3 = st.tabs([
        "Model Performance", "Test Dataset", "Explainability Evaluation"
    ])

    with perf_tab1:
        st.markdown("## 📊 Model Performance Metrics")
        perf = get_model_performance()

        st.markdown("### 🎯 Regression — All Risk Windows (validation)")
        c1, c2 = st.columns(2)
        _metric_card(c1, "RMSE — All Levels (L+M+H)", f"{perf['rmse_all']:.2f} cycles",
                     "📉", "#2a9d8f", "rgba(42,157,143,0.12)", "Lower is better")
        _metric_card(c2, "MAE — All Levels (L+M+H)",  f"{perf['mae_all']:.2f} cycles",
                     "📏", "#2a9d8f", "rgba(42,157,143,0.12)", "Lower is better")

        st.markdown("### 🔴 Critical Engines — HIGH Risk Only")
        c5, c6 = st.columns(2)
        _metric_card(c5, "RMSE — HIGH Risk", f"{perf['rmse_high']:.2f} cycles",
                     "📉", "#e63946", "rgba(230,57,70,0.10)", "Most urgent engines")
        _metric_card(c6, "MAE  — HIGH Risk", f"{perf['mae_high']:.2f} cycles",
                     "📏", "#e63946", "rgba(230,57,70,0.10)", "Most urgent engines")

        st.markdown("### 🏷️ Risk Classification")
        c_acc, _ = st.columns([1, 2])
        _metric_card(c_acc, "Overall Accuracy", f"{perf['accuracy']*100:.2f}%",
                     "✅", "#457b9d", "rgba(69,123,157,0.12)", "LOW / MEDIUM / HIGH")
        st.markdown("### 📋 Classification Report")
        st.dataframe(perf["classification_df"], use_container_width=True, hide_index=True)

        cm = perf["confusion_matrix"]
        labels = ["Low", "Medium", "High"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=labels, y=labels, colorscale="Blues",
            text=cm, texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig_cm.update_layout(template=_PT, xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig_cm, use_container_width=True, key="lstm_perf_cm")

        data = get_val_scatter_data()
        st.markdown("### Model Accuracy — All Windows (validation)")
        _true_sc = np.clip(data["true"], 0, 125)
        _pred_sc = np.clip(data["pred"], 0, 125)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=_true_sc, y=_pred_sc, mode="markers", name="Predictions",
            marker=dict(size=5, opacity=0.5, color=_true_sc, colorscale="RdYlGn",
                        showscale=True, cmin=0, cmax=125,
                        colorbar=dict(title="True RUL", orientation="v",
                                      x=1.02, xanchor="left",
                                      thickness=15, len=0.9)),
        ))
        fig_sc.add_trace(go.Scatter(x=[0, 125], y=[0, 125],
                                    mode="lines", name="Ideal",
                                    line=dict(color="white", dash="dash")))
        fig_sc.add_vrect(x0=0, x1=40, fillcolor="red", opacity=0.08, layer="below", line_width=0)
        fig_sc.add_hrect(y0=0, y1=40, fillcolor="red", opacity=0.08, layer="below", line_width=0)
        fig_sc.add_vline(x=80, line_dash="dot", line_color="orange", opacity=0.4,
                         annotation_text="RUL=80", annotation_position="top right")
        fig_sc.update_layout(
            template=_PT,
            xaxis_title="True RUL (capped at 125)", yaxis_title="Predicted RUL (capped at 125)",
            xaxis=dict(range=[0, 130]),
            yaxis=dict(range=[0, 130]),
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
            margin=dict(b=80),
        )
        st.plotly_chart(fig_sc, use_container_width=True, key="lstm_perf_scatter")

        st.markdown("### RUL Prediction Trajectory for Selected Engine")
        engine_id = st.selectbox("Select Engine", fleet_df["Machine_ID"].unique(),
                                 key="lstm_perf_engine_select")
        curve = get_engine_rul_plot(engine_id)
        fig_tr = go.Figure()
        fig_tr.add_trace(go.Scatter(x=curve["cycles"], y=curve["true"],
                                    name="True RUL", line=dict(color="blue")))
        fig_tr.add_trace(go.Scatter(x=curve["cycles"], y=curve["pred"],
                                    name="Predicted RUL", line=dict(color="orange")))
        fig_tr.add_hline(y=40, line_dash="dash", line_color="red")
        fig_tr.update_layout(template=_PT)
        st.plotly_chart(fig_tr, use_container_width=True, key="lstm_perf_traj")

    with perf_tab2:
        st.markdown("## 🧪 Test Dataset Performance")
        test_perf = get_test_performance()
        tc1, tc2 = st.columns(2)
        _metric_card(tc1, "RMSE — All Engines", f"{test_perf['rmse']:.2f} cycles",
                     "📉", "#2a9d8f", "rgba(42,157,143,0.12)", "Unseen test set")
        _metric_card(tc2, "MAE — All Engines",  f"{test_perf['mae']:.2f} cycles",
                     "📏", "#2a9d8f", "rgba(42,157,143,0.12)", "Unseen test set")
        st.markdown("### 🔴 HIGH Risk Engines Only")
        tc3, tc4 = st.columns(2)
        _metric_card(tc3, "RMSE — HIGH Risk", f"{test_perf['rmse_high']:.2f} cycles",
                     "📉", "#e63946", "rgba(230,57,70,0.10)", "Critical zone")
        _metric_card(tc4, "MAE — HIGH Risk",  f"{test_perf['mae_high']:.2f} cycles",
                     "📏", "#e63946", "rgba(230,57,70,0.10)", "Critical zone")
        st.markdown("### 🏷️ Risk Classification")
        c_acc2, _ = st.columns([1, 2])
        _metric_card(c_acc2, "Overall Accuracy", f"{test_perf['accuracy']*100:.2f}%",
                     "✅", "#457b9d", "rgba(69,123,157,0.12)", "Test set")
        st.markdown("### 📋 Classification Report")
        st.dataframe(test_perf["classification_df"], use_container_width=True, hide_index=True)

        test_cm = test_perf["confusion_matrix"]
        cm_labels = ["Low", "Medium", "High"]
        fig_tcm = go.Figure(data=go.Heatmap(
            z=test_cm, x=cm_labels, y=cm_labels, colorscale="Blues",
            text=test_cm, texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig_tcm.update_layout(
            template=_PT,
            xaxis_title="Predicted", yaxis_title="Actual",
            title="Confusion Matrix — Test Set",
        )
        st.plotly_chart(fig_tcm, use_container_width=True, key="lstm_test_cm")

        comparison_df_lstm = get_comparison_df()
        st.markdown("### 📋 Prediction Comparison Sample")
        st.dataframe(comparison_df_lstm.sort_values("True_RUL").head(50),
                     use_container_width=True, hide_index=True)

        data_t = get_test_scatter_data()
        st.markdown("### 🧪 Test Set — True vs Predicted RUL (all engines)")
        _true_tsc = np.clip(data_t["true"], 0, 125)
        _pred_tsc = np.clip(data_t["pred"], 0, 125)
        fig_tsc = go.Figure()
        fig_tsc.add_trace(go.Scatter(
            x=_true_tsc, y=_pred_tsc, mode="markers", name="Predictions",
            marker=dict(size=5, opacity=0.5, color=_true_tsc, colorscale="RdYlGn",
                        showscale=True, cmin=0, cmax=125,
                        colorbar=dict(title="True RUL", orientation="v",
                                      x=1.02, xanchor="left",
                                      thickness=15, len=0.9)),
        ))
        fig_tsc.add_trace(go.Scatter(x=[0, 125], y=[0, 125],
                                     mode="lines", name="Ideal",
                                     line=dict(color="white", dash="dash")))
        fig_tsc.add_vrect(x0=0, x1=40, fillcolor="red", opacity=0.08, layer="below", line_width=0)
        fig_tsc.add_hrect(y0=0, y1=40, fillcolor="red", opacity=0.08, layer="below", line_width=0)
        fig_tsc.add_vline(x=80, line_dash="dot", line_color="orange", opacity=0.4,
                          annotation_text="RUL=80", annotation_position="top right")
        fig_tsc.update_layout(
            template=_PT,
            xaxis_title="True RUL (capped at 125)", yaxis_title="Predicted RUL (capped at 125)",
            xaxis=dict(range=[0, 130]),
            yaxis=dict(range=[0, 130]),
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
            margin=dict(b=80),
        )
        st.plotly_chart(fig_tsc, use_container_width=True, key="lstm_test_scatter")

    with perf_tab3:
        st.markdown("### 🧠 Feature Importance Agreement — IG vs Occlusion")
        st.caption(
            "Both Integrated Gradients and Occlusion Sensitivity are applied to the LSTM. "
            "High Spearman correlation means the two methods agree on which sensors matter most."
        )
        machine_id_xai = st.selectbox(
            "Select Engine for Explainability",
            fleet_df["Machine_ID"].unique(),
            key="lstm_xai_engine",
        )
        xai_data = _cached_ig_occlusion(machine_id_xai, st.session_state.current_cycle)
        if not xai_data or len(xai_data.get("ig", [])) == 0 or len(xai_data.get("occ", [])) == 0:
            st.warning("Explainability data not available for this engine at the current cycle.")
        else:
            fig_xai = go.Figure()
            fig_xai.add_trace(go.Scatter(
                x=xai_data["ig"], y=xai_data["occ"],
                mode="markers", name="Sensors",
                marker=dict(size=8, opacity=0.7, color="#f4a261"),
                text=list(range(len(xai_data["ig"]))),
                hovertemplate="Sensor %{text}<br>IG: %{x:.4f}<br>Occlusion: %{y:.4f}<extra></extra>",
            ))
            fig_xai.update_layout(
                template=_PT,
                xaxis_title="Integrated Gradients Importance",
                yaxis_title="Occlusion Importance",
                title="IG vs Occlusion — Per-sensor Feature Importance",
            )
            st.plotly_chart(fig_xai, use_container_width=True, key="lstm_xai_scatter")
            corr_val = xai_data["corr"]
            st.metric(
                "Spearman Correlation (IG vs Occlusion)",
                f"{corr_val:.3f}",
                help="1.0 = perfect agreement · 0.0 = no agreement",
            )
            if corr_val >= 0.7:
                st.success("✅ Strong agreement — both methods highlight the same sensors.")
            elif corr_val >= 0.4:
                st.warning("⚠️ Moderate agreement — methods partially disagree.")
            else:
                st.error("❌ Low agreement — interpret explanations with caution.")


def _render_perf_tab_cnn(fleet_df):
    """Model Performance tab — CNN."""
    perf_tab1, perf_tab2, perf_tab3 = st.tabs([
        "Model Performance", "Test Dataset", "Explainability Evaluation"
    ])

    with perf_tab1:
        st.markdown("## 📊 CNN Model Performance Metrics")
        perf = get_model_performance_cnn()
        if perf is None:
            st.warning("CNN model not available.")
            return

        st.markdown("### Regression Performance (validation set)")
        c1, c2 = st.columns(2)
        _metric_card(c1, "RMSE — All Risk Levels (L+M+H)", f"{perf['rmse_all']:.2f} cycles",
                     "📉", "#2a9d8f", "rgba(42,157,143,0.12)", "Validation set")
        _metric_card(c2, "MAE — All Risk Levels (L+M+H)", f"{perf['mae_all']:.2f} cycles",
                     "📏", "#2a9d8f", "rgba(42,157,143,0.12)", "Validation set")
        st.markdown("#### Critical Engine Accuracy (HIGH risk windows only)")
        c5, c6 = st.columns(2)
        _metric_card(c5, "RMSE — HIGH Risk Only", f"{perf['rmse_high']:.2f} cycles",
                     "🔴", "#e63946", "rgba(230,57,70,0.12)", "Validation set")
        _metric_card(c6, "MAE — HIGH Risk Only", f"{perf['mae_high']:.2f} cycles",
                     "🔴", "#e63946", "rgba(230,57,70,0.12)", "Validation set")
        st.markdown("### Classification Performance")
        c_acc, _ = st.columns([1, 2])
        _metric_card(c_acc, "Overall Accuracy", f"{perf['accuracy']*100:.2f}%",
                     "✅", "#457b9d", "rgba(69,123,157,0.12)", "Validation set")
        st.markdown("### 📋 Classification Report")
        st.dataframe(perf["classification_df"], use_container_width=True, hide_index=True)

        cm = perf["confusion_matrix"]
        labels = ["Low", "Medium", "High"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=labels, y=labels, colorscale="Blues",
            text=cm, texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig_cm.update_layout(template=_PT, xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig_cm, use_container_width=True, key="cnn_perf_cm")

        data = get_cnn_val_scatter_data()
        if len(data["true"]) > 0:
            st.markdown("### Model Accuracy — All Windows (validation)")
            _true_sc = np.clip(data["true"], 0, 125)
            _pred_sc = np.clip(data["pred"], 0, 125)
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=_true_sc, y=_pred_sc, mode="markers", name="Predictions",
                marker=dict(size=5, opacity=0.5, color=_true_sc, colorscale="RdYlGn",
                            showscale=True, cmin=0, cmax=125,
                            colorbar=dict(title="True RUL", orientation="v",
                                          x=1.02, xanchor="left",
                                          thickness=15, len=0.9)),
            ))
            fig_sc.add_trace(go.Scatter(x=[0, 125], y=[0, 125],
                                        mode="lines", name="Ideal",
                                        line=dict(color="white", dash="dash")))
            fig_sc.add_vrect(x0=0, x1=40, fillcolor="red", opacity=0.08, layer="below", line_width=0)
            fig_sc.add_hrect(y0=0, y1=40, fillcolor="red", opacity=0.08, layer="below", line_width=0)
            fig_sc.add_vline(x=80, line_dash="dot", line_color="orange", opacity=0.4,
                             annotation_text="RUL=80", annotation_position="top right")
            fig_sc.update_layout(
                template=_PT,
                xaxis_title="True RUL (capped at 125)", yaxis_title="Predicted RUL (capped at 125)",
                xaxis=dict(range=[0, 130]),
                yaxis=dict(range=[0, 130]),
                legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
                margin=dict(r=80),
            )
            st.plotly_chart(fig_sc, use_container_width=True, key="cnn_perf_scatter")

        st.markdown("### RUL Prediction Trajectory for Selected Engine")
        engine_id = st.selectbox("Select Engine", fleet_df["Machine_ID"].unique(),
                                 key="cnn_perf_engine_select")
        curve = get_cnn_engine_rul_plot(engine_id)
        fig_tr = go.Figure()
        fig_tr.add_trace(go.Scatter(x=curve["cycles"], y=curve["true"],
                                    name="True RUL", line=dict(color="blue")))
        fig_tr.add_trace(go.Scatter(x=curve["cycles"], y=curve["pred"],
                                    name="Predicted RUL", line=dict(color="orange")))
        fig_tr.add_hline(y=40, line_dash="dash", line_color="red")
        fig_tr.update_layout(template=_PT)
        st.plotly_chart(fig_tr, use_container_width=True, key="cnn_perf_traj")

    with perf_tab2:
        st.markdown("## 🧪 Test Dataset Performance")
        test_perf = get_test_performance_cnn()
        if test_perf is None:
            st.warning("CNN model not available.")
            return
        tc1, tc2 = st.columns(2)
        _metric_card(tc1, "RMSE — All Engines", f"{test_perf['rmse']:.2f} cycles",
                     "📉", "#2a9d8f", "rgba(42,157,143,0.12)", "Test set")
        _metric_card(tc2, "MAE — All Engines", f"{test_perf['mae']:.2f} cycles",
                     "📏", "#2a9d8f", "rgba(42,157,143,0.12)", "Test set")
        st.markdown("#### Critical Engine Accuracy (HIGH risk only)")
        tc3, tc4 = st.columns(2)
        _metric_card(tc3, "RMSE — HIGH Risk Only", f"{test_perf['rmse_high']:.2f} cycles",
                     "🔴", "#e63946", "rgba(230,57,70,0.12)", "Test set")
        _metric_card(tc4, "MAE — HIGH Risk Only", f"{test_perf['mae_high']:.2f} cycles",
                     "🔴", "#e63946", "rgba(230,57,70,0.12)", "Test set")
        st.markdown("### Classification Performance")
        c_acc2, _ = st.columns([1, 2])
        _metric_card(c_acc2, "Overall Accuracy", f"{test_perf['accuracy']*100:.2f}%",
                     "✅", "#457b9d", "rgba(69,123,157,0.12)", "Test set")
        st.markdown("### 📋 Classification Report")
        st.dataframe(test_perf["classification_df"], use_container_width=True, hide_index=True)

        test_cm = test_perf["confusion_matrix"]
        cm_labels = ["Low", "Medium", "High"]
        fig_tcm = go.Figure(data=go.Heatmap(
            z=test_cm, x=cm_labels, y=cm_labels, colorscale="Blues",
            text=test_cm, texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig_tcm.update_layout(
            template=_PT,
            xaxis_title="Predicted", yaxis_title="Actual",
            title="Confusion Matrix — Test Set",
        )
        st.plotly_chart(fig_tcm, use_container_width=True, key="cnn_test_cm")

        data_t = get_cnn_test_scatter_data()
        if len(data_t["true"]) > 0:
            st.markdown("### 🧪 Test Set — True vs Predicted RUL (all engines)")
            _true_tsc = np.clip(data_t["true"], 0, 125)
            _pred_tsc = np.clip(data_t["pred"], 0, 125)
            fig_tsc = go.Figure()
            fig_tsc.add_trace(go.Scatter(
                x=_true_tsc, y=_pred_tsc, mode="markers", name="Predictions",
                marker=dict(size=5, opacity=0.5, color=_true_tsc, colorscale="RdYlGn",
                            showscale=True, cmin=0, cmax=125,
                            colorbar=dict(title="True RUL", orientation="v",
                                          x=1.02, xanchor="left",
                                          thickness=15, len=0.9)),
            ))
            fig_tsc.add_trace(go.Scatter(x=[0, 125], y=[0, 125],
                                         mode="lines", name="Ideal",
                                         line=dict(color="white", dash="dash")))
            fig_tsc.add_vrect(x0=0, x1=40, fillcolor="red", opacity=0.08, layer="below", line_width=0)
            fig_tsc.add_hrect(y0=0, y1=40, fillcolor="red", opacity=0.08, layer="below", line_width=0)
            fig_tsc.add_vline(x=80, line_dash="dot", line_color="orange", opacity=0.4,
                              annotation_text="RUL=80", annotation_position="top right")
            fig_tsc.update_layout(
                template=_PT,
                xaxis_title="True RUL (capped at 125)", yaxis_title="Predicted RUL (capped at 125)",
                xaxis=dict(range=[0, 130]),
                yaxis=dict(range=[0, 130]),
                legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
                margin=dict(r=80),
            )
            st.plotly_chart(fig_tsc, use_container_width=True, key="cnn_test_scatter")

    with perf_tab3:
        st.markdown("### 🧠 Feature Importance Agreement — LRP vs IG")
        st.caption(
            "Both LRP and Integrated Gradients are applied to the CNN. "
            "High Spearman correlation means the two methods agree on which sensors matter most."
        )
        if not CNN_AVAILABLE:
            st.warning("CNN model not available.")
        else:
            machine_id_cnn_xai = st.selectbox(
                "Select Engine for Explainability",
                fleet_df["Machine_ID"].unique(),
                key="cnn_xai_engine",
            )
            cnn_xai_data = _cached_lrp_ig_cnn(
                machine_id_cnn_xai, st.session_state.current_cycle
            )
            if not cnn_xai_data or len(cnn_xai_data.get("lrp", [])) == 0:
                st.warning("Explainability data not available for this engine at the current cycle.")
            else:
                fig_cnn_xai = go.Figure()
                fig_cnn_xai.add_trace(go.Scatter(
                    x=cnn_xai_data["ig"],
                    y=cnn_xai_data["lrp"],
                    mode="markers",
                    name="Sensors",
                    marker=dict(size=8, opacity=0.7, color="#3a7bd5"),
                    text=list(range(len(cnn_xai_data["ig"]))),
                    hovertemplate="Sensor %{text}<br>IG: %{x:.4f}<br>LRP: %{y:.4f}<extra></extra>",
                ))
                fig_cnn_xai.update_layout(
                    template=_PT,
                    xaxis_title="Integrated Gradients Importance",
                    yaxis_title="LRP Importance",
                    title="LRP vs IG — Per-sensor Feature Importance",
                )
                st.plotly_chart(fig_cnn_xai, use_container_width=True, key="cnn_xai_scatter")
                corr_val = cnn_xai_data["corr"]
                color   = "#22c55e" if corr_val >= 0.7 else "#f4a261" if corr_val >= 0.4 else "#e63946"
                st.metric(
                    "Spearman Correlation (LRP vs IG)",
                    f"{corr_val:.3f}",
                    help="1.0 = perfect agreement · 0.0 = no agreement",
                )
                if corr_val >= 0.7:
                    st.success("✅ Strong agreement — both methods highlight the same sensors.")
                elif corr_val >= 0.4:
                    st.warning("⚠️ Moderate agreement — methods partially disagree.")
                else:
                    st.error("❌ Low agreement — interpret explanations with caution.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB STRUCTURE — badge counts from early fleet computation
# ══════════════════════════════════════════════════════════════════════════════
# Tab labels are kept static so Streamlit treats them as the same widget
# across every rerun — dynamic labels (e.g. _high_str) caused the active
# tab to reset to Home whenever the HIGH-risk count changed.
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "🔧 Maintenance Engineer",
    "📊 Operations Manager",
    "📈 Model Performance",
])

# ══════════════════════════════════════════════════════════════════════════════
# HOME TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("Environment Overview")

    cycle_number = _cycle_slider(
        "lstm_history_home" if selected_model == "LSTM Model" else "cnn_history_home"
    )

    if selected_model == "LSTM Model":
        fleet_df = build_fleet_ranking(cycle_number)
    else:
        if not CNN_AVAILABLE:
            st.warning("CNN model not available. Please train and save the CNN first.")
            st.stop()
        fleet_df = build_fleet_ranking_cnn(cycle_number)

    if len(fleet_df) == 0:
        st.warning("No engines available at this cycle.")
        st.stop()

    _pfx_home = "lstm" if selected_model == "LSTM Model" else "cnn"
    _cfn_home = get_engine_rul_plot if selected_model == "LSTM Model" else get_cnn_engine_rul_plot
    _render_home_tab(fleet_df, prefix=_pfx_home, curve_fn=_cfn_home)

# ══════════════════════════════════════════════════════════════════════════════
# MAINTENANCE ENGINEER TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if selected_model == "LSTM Model":
        # ── LSTM Maintenance Engineer ─────────────────────────────────────────
        st.title("🔧 Maintenance Engineer Dashboard")
        st.title("Fleet Overview")

        cycle_number = _cycle_slider("lstm_history_engineer")
        fleet_df = build_fleet_ranking(cycle_number)

        if len(fleet_df) == 0:
            st.warning("No engines available at this cycle.")
            st.stop()

        # ── Engineer export (Excel, multi-sheet) ─────────────────────────────
        with st.expander("📥 Export Engineer Report (Excel)"):
            st.caption(
                "Generates a 5-sheet Excel workbook: fleet overview, mean sensor "
                "values, last-cycle sensor values, sensor anomaly flags, and the "
                "healthy baseline reference."
            )
            if st.button("Generate Export", key="lstm_engineer_export_btn"):
                with st.spinner("Building engineer report…"):
                    _xl_buf = _build_engineer_excel(cycle_number, use_cnn=False)
                st.download_button(
                    "⬇️ Download Excel",
                    _xl_buf,
                    file_name=f"engineer_report_cycle_{cycle_number}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="lstm_engineer_download",
                )

        most_urgent = fleet_df.sort_values("Predicted_RUL").iloc[0]["Machine_ID"]
        st.markdown(f"## Most Critical Engine: {most_urgent}")

        top_k = st.session_state.get("lstm_top_features_slider", 3)
        result, report_df = explain_machine(most_urgent, cycle_number, fleet_df,
                                            show_plot=False, top_k=top_k)

        # IG heatmap + cumulative bar
        st.markdown("### Sensor Importance Heatmap")
        _render_ig_heatmap_bar(
            np.array(result["ig"]),
            result["feature_names"],
            result["cycles"],
            result["sensor_values"],
            result["healthy_values"],
            result["healthy_std"],
            hm_key="lstm_heatmap_urgent",
            bar_key="lstm_bar_urgent",
            colorscale="RdYlGn",
            zmid=0,
        )

        # Temporal Attention
        st.markdown("### 📈 Temporal Attention")
        fig_att = go.Figure()
        fig_att.add_trace(go.Scatter(
            x=result["cycles"], y=result["attention"],
            mode="lines", line=dict(color="cyan", width=2),
        ))
        fig_att.update_layout(
            template=_PT, xaxis_title="Cycle", yaxis_title="Attention"
        )
        st.plotly_chart(fig_att, use_container_width=True, key="lstm_att_urgent")

        top_k = st.slider("Number of Top Sensors to Show", 3, 10, 3,
                           key="lstm_top_features_slider")

        # Report
        st.markdown("### 📄 Explanation Report")
        _render_risk_rul_cards(result["risk"], result["rul"])
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        # Select Engine
        st.markdown("---")
        st.markdown("## 🔍 Analyze Another Engine")
        available_engines = fleet_df[fleet_df["Machine_ID"] != most_urgent]["Machine_ID"].values
        selected_engine = st.selectbox("Select Engine", available_engines,
                                       key="lstm_engine_select")

        top_k2 = st.session_state.get("lstm_top_features_slider_selected", 3)
        result2, report2_df = explain_machine(selected_engine, cycle_number, fleet_df,
                                              show_plot=False, top_k=top_k2)

        st.markdown("### Sensor Importance Heatmap for Selected Engine")
        _render_ig_heatmap_bar(
            np.array(result2["ig"]),
            result2["feature_names"],
            result2["cycles"],
            result2["sensor_values"],
            result2["healthy_values"],
            result2["healthy_std"],
            hm_key="lstm_heatmap_selected",
            bar_key="lstm_bar_selected",
            colorscale="RdYlGn",
            zmid=0,
        )

        # Temporal Attention (selected)
        st.markdown("### 📈 Temporal Attention")
        fig_att2 = go.Figure()
        fig_att2.add_trace(go.Scatter(
            x=result2["cycles"], y=result2["attention"],
            mode="lines", line=dict(color="cyan", width=2),
        ))
        fig_att2.update_layout(
            template=_PT, xaxis_title="Cycle", yaxis_title="Attention"
        )
        st.plotly_chart(fig_att2, use_container_width=True, key="lstm_att_selected")

        top_k2 = st.slider("Number of Top Sensors to Show", 3, 10, 3,
                            key="lstm_top_features_slider_selected")

        # Report
        st.markdown("### 📄 Explanation Report")
        _render_risk_rul_cards(result2["risk"], result2["rul"])
        st.dataframe(report2_df, use_container_width=True, hide_index=True)

        # Sensor deep dive
        _render_sensor_deepdive(
            fleet_df, selected_engine,
            result_cache={most_urgent: result, selected_engine: result2},
            cycle_number=cycle_number,
            explain_fn=lambda mid, cn, fdf, top_k=3: explain_machine(mid, cn, fdf,
                                                                      show_plot=False,
                                                                      top_k=top_k),
            prefix="lstm",
        )

        # Maintenance controls
        _render_maintenance_controls(fleet_df, prefix="lstm")

    else:
        # ── CNN Maintenance Engineer ──────────────────────────────────────────
        st.title("🔧 Maintenance Engineer Dashboard (CNN)")
        st.title("Fleet Overview")

        if not CNN_AVAILABLE:
            st.warning("CNN model not available. Please train and save the CNN first.")
            st.stop()

        cycle_number = _cycle_slider("cnn_history_engineer")
        fleet_df_cnn = build_fleet_ranking_cnn(cycle_number)

        if len(fleet_df_cnn) == 0:
            st.warning("No engines available at this cycle.")
            st.stop()

        # ── Engineer export (Excel, multi-sheet) ─────────────────────────────
        with st.expander("📥 Export Engineer Report (Excel)"):
            st.caption(
                "Generates a 5-sheet Excel workbook: fleet overview, mean sensor "
                "values (50-cycle window), last-cycle sensor values, sensor anomaly "
                "flags, and the healthy baseline reference."
            )
            if st.button("Generate Export", key="cnn_engineer_export_btn"):
                with st.spinner("Building engineer report…"):
                    _xl_buf_cnn = _build_engineer_excel(cycle_number, use_cnn=True)
                st.download_button(
                    "⬇️ Download Excel",
                    _xl_buf_cnn,
                    file_name=f"engineer_report_cnn_cycle_{cycle_number}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="cnn_engineer_download",
                )

        most_urgent_cnn = fleet_df_cnn.sort_values("Predicted_RUL").iloc[0]["Machine_ID"]
        st.markdown(f"## Most Critical Engine: {most_urgent_cnn}")

        top_k_cnn = st.session_state.get("cnn_top_features_slider", 3)
        result_cnn, report_df_cnn = explain_machine_cnn(
            most_urgent_cnn, cycle_number, fleet_df_cnn, top_k=top_k_cnn
        )

        if result_cnn is None:
            st.warning("Not enough data for the most urgent engine.")
        else:
            # ── Activation Maximization ───────────────────────────────────────
            st.markdown(
                f"### {_tooltip('🧠 Activation Maximization', 'Gradient ascent finds the synthetic input pattern that maximizes the CNN predicted RUL — showing what a perfectly healthy engine looks like according to the model')} — CNN's Ideal Healthy Engine",
                unsafe_allow_html=True,
            )
            st.caption(
                "Red = CNN wants this sensor HIGH to predict long RUL · "
                "Green = CNN wants it LOW · "
                "Reveals what a 'perfectly healthy' engine looks like according to the model."
            )
            act_max   = get_cnn_activation_max()         # (50, 24) raw signed
            act_max_T = act_max.T                          # (24, 50)
            fnames_cnn = result_cnn["feature_names"]
            cycs_am    = np.arange(1, act_max_T.shape[1] + 1)

            col_am_hm, col_am_bar = st.columns([3, 2])
            with col_am_hm:
                fig_am = go.Figure(go.Heatmap(
                    z=act_max_T,
                    x=cycs_am,
                    y=fnames_cnn[:act_max_T.shape[0]],
                    colorscale="RdYlGn",   # red=negative (CNN wants low), green=positive (CNN wants high)
                    zmid=0,
                    colorbar=dict(title="Activation"),
                    hovertemplate="<b>%{y}</b><br>Cycle: %{x}<br>Activation: %{z:.4f}<extra></extra>",
                ))
                fig_am.update_layout(
                    template=_PT, height=650,
                    title="Activation Maximization — Ideal Healthy Input<br>"
                          "<sup>Green = CNN wants HIGH · Red = CNN wants LOW</sup>",
                )
                st.plotly_chart(fig_am, use_container_width=True, key="cnn_act_max_hm")

            with col_am_bar:
                # Per-sensor optimal value at last cycle (signed) — matches notebook
                am_nf    = min(act_max_T.shape[0], len(fnames_cnn))
                am_last  = act_max_T[:am_nf, -1]            # last cycle, signed (scaled space)
                am_si    = np.argsort(am_last)               # ascending: most negative first
                am_cols  = ["#e63946" if v < 0 else "#22c55e" for v in am_last[am_si]]
                fig_am_bar = go.Figure(go.Bar(
                    x=am_last[am_si],
                    y=[fnames_cnn[i] for i in am_si],
                    orientation="h",
                    marker_color=am_cols,
                    hovertemplate=(
                        "%{y}<br>Optimal value (last cycle): %{x:.4f}"
                        "<br><i>Green = CNN prefers high · Red = CNN prefers low</i>"
                        "<extra></extra>"
                    ),
                ))
                fig_am_bar.update_layout(
                    template=_PT, height=650,
                    title="Per-sensor optimal value (last cycle)<br>"
                          "<sup>(what the CNN prefers for high RUL)</sup>",
                    xaxis_title="Optimal Sensor Value (scaled)",
                    yaxis_title="",
                    margin=dict(l=10, r=10, t=70, b=40),
                )
                fig_am_bar.add_vline(x=0, line_dash="dash",
                                     line_color="rgba(255,255,255,0.4)", line_width=1)
                st.plotly_chart(fig_am_bar, use_container_width=True, key="cnn_act_max_bar")

            # ── Healthy Mode Recommendations ──────────────────────────────────
            st.markdown(f"### 🎯 Healthy Mode Recommendations — Engine {most_urgent_cnn}")
            st.caption(
                "Based on Activation Maximization: what sensor adjustments would "
                "move this engine closer to the CNN's 'ideal healthy' profile."
            )
            try:
                _nf_rec   = min(am_nf, len(result_cnn["sensor_values"][0]))
                # sensor_values is already inverse-transformed to raw units in the backend
                _curr_raw = np.array(result_cnn["sensor_values"])[-1, :_nf_rec]
                # Activation max is in SCALED space — inverse-transform to raw
                _tgt_sc   = act_max_T[:_nf_rec, -1]        # last cycle, scaled
                _tgt_raw  = _tgt_sc * scaler.scale_[:_nf_rec] + scaler.mean_[:_nf_rec]
                _delta_raw = _tgt_raw - _curr_raw
                _rec_df = pd.DataFrame({
                    "Sensor":           fnames_cnn[:_nf_rec],
                    "Current (raw)":    np.round(_curr_raw, 3),
                    "CNN Target (raw)": np.round(_tgt_raw,  3),
                    "Change Needed":    np.round(_delta_raw, 3),
                    "Action":           [
                        "⬆ Raise" if d > 0.05 * abs(c) else
                        "⬇ Lower" if d < -0.05 * abs(c) else "✓ OK"
                        for d, c in zip(_delta_raw, _curr_raw)
                    ],
                })
                # Only show sensors with >5% relative change needed
                _rec_df = _rec_df[_rec_df["Action"] != "✓ OK"].sort_values(
                    "Change Needed", key=abs, ascending=False
                )
                if len(_rec_df) > 0:
                    st.dataframe(_rec_df, use_container_width=True, hide_index=True)
                else:
                    st.success("✅ Engine sensors are already close to the CNN healthy profile!")
            except Exception as _rec_err:
                st.info(f"Recommendations unavailable: {_rec_err}")

            # ── LRP Heatmap ───────────────────────────────────────────────────
            st.markdown(
                f"### {_tooltip('🔬 LRP', 'Layer-wise Relevance Propagation: propagates the prediction score backward through CNN layers to assign relevance to each input sensor-cycle')} — Layer-wise Relevance Propagation",
                unsafe_allow_html=True,
            )
            lrp_map = np.array(result_cnn["lrp"])        # (50, 24) signed
            lrp_T   = lrp_map.T                            # (24, 50)
            nf_l    = min(lrp_T.shape[0], len(fnames_cnn))
            nc_l    = min(lrp_T.shape[1], len(result_cnn["cycles"]))

            col_lrp_hm, col_lrp_bar = st.columns([3, 2])
            with col_lrp_hm:
                fig_lrp = go.Figure(go.Heatmap(
                    z=lrp_T[:nf_l, :nc_l],
                    x=result_cnn["cycles"][:nc_l],
                    y=fnames_cnn[:nf_l],
                    colorscale="RdYlGn",
                    zmid=0,
                    colorbar=dict(title="LRP Relevance"),
                    hovertemplate="<b>%{y}</b><br>Cycle: %{x}<br>LRP Relevance: %{z:.4f}<extra></extra>",
                ))
                fig_lrp.update_layout(
                    template=_PT, height=650,
                    title="LRP Relevance Heatmap (red=degrades RUL · green=extends RUL)",
                )
                st.plotly_chart(fig_lrp, use_container_width=True, key="cnn_lrp_hm_urgent")

            with col_lrp_bar:
                lrp_sens = np.abs(lrp_T[:nf_l, :nc_l]).mean(axis=1)
                lrp_sens = lrp_sens / (lrp_sens.sum() + 1e-8)
                lrp_si   = np.argsort(lrp_sens)
                fig_lrp_bar = go.Figure(go.Bar(
                    x=lrp_sens[lrp_si],
                    y=[fnames_cnn[i] for i in lrp_si],
                    orientation="h",
                    marker_color="salmon",
                    hovertemplate="%{y}<br>LRP share: %{x:.2%}<extra></extra>",
                ))
                fig_lrp_bar.update_layout(
                    template=_PT, height=650,
                    title="Sensor LRP Relevance Share",
                    xaxis_title="Share of Total Relevance",
                    xaxis_tickformat=".0%", yaxis_title="",
                    margin=dict(l=10, r=10, t=60, b=40),
                )
                st.plotly_chart(fig_lrp_bar, use_container_width=True, key="cnn_lrp_bar_urgent")

            # ── Report ────────────────────────────────────────────────────────
            st.markdown("### 📄 Explanation Report")
            top_k_cnn = st.slider("Number of Top Sensors to Show", 3, 10, 3,
                                  key="cnn_top_features_slider")
            _render_risk_rul_cards(result_cnn["risk"], result_cnn["rul"])
            st.dataframe(report_df_cnn, use_container_width=True, hide_index=True)

        # ── Select Engine ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🔍 Analyze Another Engine")
        available_engines_cnn = fleet_df_cnn[
            fleet_df_cnn["Machine_ID"] != most_urgent_cnn
        ]["Machine_ID"].values
        selected_engine_cnn = st.selectbox("Select Engine", available_engines_cnn,
                                           key="cnn_engine_select")

        top_k2_cnn = st.session_state.get("cnn_top_features_slider_selected", 3)
        result2_cnn, report2_df_cnn = explain_machine_cnn(
            selected_engine_cnn, cycle_number, fleet_df_cnn, top_k=top_k2_cnn
        )

        if result2_cnn is not None:
            fnames2 = result2_cnn["feature_names"]

            # ── LRP Heatmap — selected engine ─────────────────────────────────
            st.markdown(
                f"### {_tooltip('🔬 LRP', 'Layer-wise Relevance Propagation: propagates the prediction score backward through CNN layers to assign relevance to each input sensor-cycle')} — Layer-wise Relevance Propagation",
                unsafe_allow_html=True,
            )
            lrp2_map = np.array(result2_cnn["lrp"])        # (50, 24) signed
            lrp2_T   = lrp2_map.T                           # (24, 50)
            nf_l2    = min(lrp2_T.shape[0], len(fnames2))
            nc_l2    = min(lrp2_T.shape[1], len(result2_cnn["cycles"]))

            col_lrp2_hm, col_lrp2_bar = st.columns([3, 2])
            with col_lrp2_hm:
                fig_lrp2 = go.Figure(go.Heatmap(
                    z=lrp2_T[:nf_l2, :nc_l2],
                    x=result2_cnn["cycles"][:nc_l2],
                    y=fnames2[:nf_l2],
                    colorscale="RdYlGn",
                    zmid=0,
                    colorbar=dict(title="LRP Relevance"),
                    hovertemplate="<b>%{y}</b><br>Cycle: %{x}<br>LRP Relevance: %{z:.4f}<extra></extra>",
                ))
                fig_lrp2.update_layout(
                    template=_PT, height=650,
                    title="LRP Relevance Heatmap (red=degrades RUL · green=extends RUL)",
                )
                st.plotly_chart(fig_lrp2, use_container_width=True, key="cnn_lrp_hm_selected")

            with col_lrp2_bar:
                lrp2_sens = np.abs(lrp2_T[:nf_l2, :nc_l2]).mean(axis=1)
                lrp2_sens = lrp2_sens / (lrp2_sens.sum() + 1e-8)
                lrp2_si   = np.argsort(lrp2_sens)
                fig_lrp2_bar = go.Figure(go.Bar(
                    x=lrp2_sens[lrp2_si],
                    y=[fnames2[i] for i in lrp2_si],
                    orientation="h",
                    marker_color="salmon",
                    hovertemplate="%{y}<br>LRP share: %{x:.2%}<extra></extra>",
                ))
                fig_lrp2_bar.update_layout(
                    template=_PT, height=650,
                    title="Sensor LRP Relevance Share",
                    xaxis_title="Share of Total Relevance",
                    xaxis_tickformat=".0%", yaxis_title="",
                    margin=dict(l=10, r=10, t=60, b=40),
                )
                st.plotly_chart(fig_lrp2_bar, use_container_width=True, key="cnn_lrp_bar_selected")

            # ── Healthy Mode Recommendations for selected engine (full width) ──
            st.markdown(f"### 🎯 Healthy Mode Recommendations — Engine {selected_engine_cnn}")
            st.caption(
                "Engine specific: what sensor changes would bring this engine "
                "closer to the CNN's ideal healthy profile (last cycle)."
            )
            try:
                _nf2   = min(am_nf, len(result2_cnn["sensor_values"][0]))
                # sensor_values already raw (inverse-transformed in backend)
                _curr2 = np.array(result2_cnn["sensor_values"])[-1, :_nf2]
                # Activation max → inverse transform to raw (same global profile)
                _tgt2_sc  = act_max_T[:_nf2, -1]
                _tgt2_raw = _tgt2_sc * scaler.scale_[:_nf2] + scaler.mean_[:_nf2]
                _delta2   = _tgt2_raw - _curr2
                _rec2_df  = pd.DataFrame({
                    "Sensor":           fnames2[:_nf2],
                    "Current (raw)":    np.round(_curr2,    3),
                    "CNN Target (raw)": np.round(_tgt2_raw, 3),
                    "Change Needed":    np.round(_delta2,   3),
                    "Action":           [
                        "⬆ Raise" if d >  0.05 * abs(c) else
                        "⬇ Lower" if d < -0.05 * abs(c) else "✓ OK"
                        for d, c in zip(_delta2, _curr2)
                    ],
                })
                _rec2_df = _rec2_df[_rec2_df["Action"] != "✓ OK"].sort_values(
                    "Change Needed", key=abs, ascending=False
                )
                if len(_rec2_df) > 0:
                    st.dataframe(_rec2_df, use_container_width=True, hide_index=True)
                else:
                    st.success("✅ Sensors already close to the CNN healthy profile!")
            except Exception as _e2:
                st.info(f"Recommendations unavailable: {_e2}")

            top_k2_cnn = st.slider("Number of Top Sensors to Show", 3, 10, 3,
                                   key="cnn_top_features_slider_selected")
            st.markdown("### 📄 Explanation Report")
            _render_risk_rul_cards(result2_cnn["risk"], result2_cnn["rul"])
            st.dataframe(report2_df_cnn, use_container_width=True, hide_index=True)

        # Sensor Deep Dive
        cache_cnn = {}
        if result_cnn is not None:
            cache_cnn[most_urgent_cnn] = result_cnn
        if result2_cnn is not None:
            cache_cnn[selected_engine_cnn] = result2_cnn

        _render_sensor_deepdive(
            fleet_df_cnn, selected_engine_cnn,
            result_cache=cache_cnn,
            cycle_number=cycle_number,
            explain_fn=lambda mid, cn, fdf, top_k=3: explain_machine_cnn(mid, cn, fdf,
                                                                           top_k=top_k),
            prefix="cnn",
        )

        _render_maintenance_controls(fleet_df_cnn, prefix="cnn")

# ══════════════════════════════════════════════════════════════════════════════
# OPERATIONS MANAGER TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("📊 Operations Manager Dashboard")

    if selected_model == "LSTM Model":
        cycle_number = _cycle_slider("lstm_history_operations")
        fleet_df_ops = build_fleet_ranking(cycle_number)
        prefix_ops = "lstm"
    else:
        if not CNN_AVAILABLE:
            st.warning("CNN model not available.")
            st.stop()
        cycle_number = _cycle_slider("cnn_history_operations")
        fleet_df_ops = build_fleet_ranking_cnn(cycle_number)
        prefix_ops = "cnn"

    st.title("Fleet Overview")

    if len(fleet_df_ops) == 0:
        st.warning("No engines available at this cycle.")
        st.stop()

    _render_ops_tab(fleet_df_ops, prefix=prefix_ops)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("📉 Model Performance Dashboard")

    if selected_model == "LSTM Model":
        _perf_fleet = build_fleet_ranking(st.session_state.current_cycle)
        if len(_perf_fleet) == 0:
            st.warning("No engines available yet.")
            st.stop()
        _render_perf_tab_lstm(_perf_fleet)
    else:
        if not CNN_AVAILABLE:
            st.warning("CNN model not available. Please train and save the CNN first.")
            st.stop()
        _perf_fleet_cnn = build_fleet_ranking_cnn(st.session_state.current_cycle)
        if len(_perf_fleet_cnn) == 0:
            st.warning("No engines available yet.")
            st.stop()
        _render_perf_tab_cnn(_perf_fleet_cnn)

# ── HIGH-risk popup alert ──────────────────────────────────────────────────────
# Rendered unconditionally at the end so the Streamlit widget tree is always
# the same shape — a conditional component earlier in the script would shift
# the index of st.tabs() and reset the active tab to Home on every rerun.
# The JS simply returns early when there is nothing to show.
components.html(
    "<!DOCTYPE html><html><head></head><body><script>"
    "(function() {"
    "  if (!" + ("true" if _show_alert else "false") + ") return;"
    "  var pdoc = window.parent.document;"
    "  var old  = pdoc.getElementById('pm-high-alert');"
    "  if (old) old.remove();"
    "  if (!pdoc.getElementById('pm-alert-style')) {"
    "    var styleEl = pdoc.createElement('style');"
    "    styleEl.id  = 'pm-alert-style';"
    "    styleEl.textContent = '@keyframes pmSlideIn{from{opacity:0;transform:translateX(70px)}to{opacity:1;transform:translateX(0)}}';"
    "    pdoc.head.appendChild(styleEl);"
    "  }"
    "  var popup = pdoc.createElement('div');"
    "  popup.id  = 'pm-high-alert';"
    "  popup.style.cssText = ["
    "    'position:fixed','top:72px','right:24px','width:370px',"
    "    'max-width:calc(100vw - 48px)',"
    "    'background:linear-gradient(145deg,#7f1d1d,#b91c1c,#dc2626)',"
    "    'color:#fff','border-radius:16px','padding:24px 26px',"
    "    'z-index:2147483647',"
    "    'box-shadow:0 12px 48px rgba(127,29,29,0.75),0 0 0 1px rgba(255,80,80,0.3)',"
    "    'border-left:6px solid #ff4040',"
    "    'font-family:Arial,sans-serif','font-size:14px',"
    "    'animation:pmSlideIn 0.35s cubic-bezier(.22,.68,0,1.2) both'"
    "  ].join(';');"
    "  var hdr = pdoc.createElement('div');"
    "  hdr.style.cssText = 'font-size:10px;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;opacity:0.65;margin-bottom:16px;';"
    "  hdr.textContent = '⚠️  Fleet Alert';"
    "  popup.appendChild(hdr);"
    "  var entries = pdoc.createElement('div');"
    "  entries.innerHTML = '" + _alert_entries_js + "';"
    "  popup.appendChild(entries);"
    "  var btnRow = pdoc.createElement('div');"
    "  btnRow.style.cssText = 'display:flex;gap:10px;margin-top:18px;';"
    "  var homeBtn = pdoc.createElement('button');"
    "  homeBtn.textContent = '🏠 Go to Home';"
    "  homeBtn.style.cssText = 'flex:1;background:#fff;color:#9b1c1c;border:none;border-radius:9px;padding:11px 14px;font-size:13px;font-weight:700;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,0.2);';"
    "  var dismissBtn = pdoc.createElement('button');"
    "  dismissBtn.textContent = '✕ Dismiss';"
    "  dismissBtn.style.cssText = 'background:rgba(255,255,255,0.12);color:#fff;border:1px solid rgba(255,255,255,0.3);border-radius:9px;padding:11px 14px;font-size:13px;cursor:pointer;';"
    "  btnRow.appendChild(homeBtn);"
    "  btnRow.appendChild(dismissBtn);"
    "  popup.appendChild(btnRow);"
    "  pdoc.body.appendChild(popup);"
    "  var timer = setTimeout(function() { popup.remove(); }, 12000);"
    "  homeBtn.addEventListener('click', function() {"
    "    var tabs = pdoc.querySelectorAll('button[data-baseweb=\"tab\"]');"
    "    if (tabs && tabs[0]) tabs[0].click();"
    "    popup.remove(); clearTimeout(timer);"
    "  });"
    "  dismissBtn.addEventListener('click', function() {"
    "    popup.remove(); clearTimeout(timer);"
    "  });"
    "})();"
    "</script></body></html>",
    height=0,
)
