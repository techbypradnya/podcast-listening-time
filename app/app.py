"""Dropoff Studio - interactive front-end for the podcast listening-time deep learning model.

Run:  streamlit run app/app.py
"""
from __future__ import annotations

import json
import os
import sys
import zlib
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MODEL_DIR = ROOT / "models"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIMES = ["Morning", "Afternoon", "Evening", "Night"]
SENTS = ["Negative", "Neutral", "Positive"]
REQUIRED = ["Podcast_Name", "Episode_Length_minutes", "Genre", "Host_Popularity_percentage", "Publication_Day",
            "Publication_Time", "Guest_Popularity_percentage", "Number_of_Ads", "Episode_Sentiment"]
C = dict(bg="#12102B", line="#2E2B63", text="#EEEDFF", muted="#9C99C9", on="#7CF2C8", mark="#FFB84D",
         loss="#FF7C96", lilac="#8F7CFF")

st.set_page_config(page_title="Dropoff Studio | Podcast listening-time predictor", page_icon="🎙️", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(f"<style>{(Path(__file__).parent / 'styles.css').read_text()}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading the neural network…")
def load_predictor():
    from podcast_ml.predict import PodcastPredictor
    return PodcastPredictor(MODEL_DIR)


@st.cache_data
def load_json(name: str):
    p = MODEL_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def home_genres() -> dict:
    try:
        from podcast_ml.train import find_dataset
        d = pd.read_csv(find_dataset(ROOT), usecols=["Podcast_Name", "Genre"])
        return d.groupby("Podcast_Name")["Genre"].agg(lambda s: s.value_counts().index[0]).to_dict()
    except Exception:
        return {}


if not (MODEL_DIR / "podcast_model.keras").exists():
    st.error("No trained model found in `models/`. Run the notebook "
             "`notebooks/podcast_listening_time_deep_learning.ipynb` or `python -m podcast_ml.train` first.")
    st.stop()

predictor = load_predictor()
HOME = home_genres()


# --------------------------------------------------------------------------- state & presets
DEFAULTS = dict(podcast=predictor.podcasts[0], genre=predictor.genres[0], length=60.0, length_known=True,
                host=60, has_guest=True, guest=55, ads=1, sentiment="Neutral", day="Wednesday", time="Evening",
                ep_no=50)
PRESETS = {
    "Weekend true-crime binge": dict(podcast="Mystery Matters", genre="True Crime", length=82.0, host=84, has_guest=True,
                                     guest=72, ads=1, sentiment="Positive", day="Sunday", time="Evening", ep_no=42),
    "Ad-heavy Monday news": dict(podcast="Daily Digest", genre="News", length=45.0, host=48, has_guest=False, guest=0,
                                 ads=3, sentiment="Negative", day="Monday", time="Afternoon", ep_no=210),
    "Short tech brief": dict(podcast="Tech Trends", genre="Technology", length=18.0, host=70, has_guest=False, guest=0,
                             ads=0, sentiment="Neutral", day="Tuesday", time="Morning", ep_no=12),
    "Long-form interview": dict(podcast="Business Insights", genre="Business", length=110.0, host=65, has_guest=True,
                                guest=90, ads=2, sentiment="Positive", day="Thursday", time="Evening", ep_no=77),
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset():
    p = PRESETS.get(st.session_state.preset)
    if not p:
        return
    for k, v in p.items():
        if k == "podcast" and v not in predictor.podcasts:
            continue
        if k == "genre" and v not in predictor.genres:
            continue
        st.session_state[k] = v


def sync_genre():
    g = HOME.get(st.session_state.podcast)
    if g in predictor.genres:
        st.session_state.genre = g


def current() -> dict:
    s = st.session_state
    return dict(Podcast_Name=s.podcast, Episode_Title=f"Episode {int(s.ep_no)}",
                Episode_Length_minutes=float(s.length) if s.length_known else np.nan, Genre=s.genre,
                Host_Popularity_percentage=float(s.host), Publication_Day=s.day, Publication_Time=s.time,
                Guest_Popularity_percentage=float(s.guest) if s.has_guest else np.nan,
                Number_of_Ads=float(s.ads), Episode_Sentiment=s.sentiment)


def run(rows: list[dict], intervals: bool = False):
    df = pd.DataFrame(rows)
    return predictor.predict_with_interval(df) if intervals else predictor.predict(df)


def style(fig, h=380):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=C["text"], family="Figtree, Segoe UI, sans-serif"),
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=C["line"], zeroline=False, linecolor=C["line"])
    fig.update_yaxes(gridcolor=C["line"], zeroline=False, linecolor=C["line"])
    return fig


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Episode details")
    st.selectbox("Start from a preset", ["Custom"] + list(PRESETS), key="preset", on_change=apply_preset)
    st.selectbox("Podcast", predictor.podcasts, key="podcast", on_change=sync_genre)
    st.selectbox("Genre", predictor.genres, key="genre")
    st.checkbox("I know the episode length", key="length_known")
    st.slider("Episode length (minutes)", 5.0, 120.0, step=0.5, key="length", disabled=not st.session_state.length_known)
    st.slider("Host popularity (%)", 0, 100, key="host")
    st.checkbox("Has a guest", key="has_guest")
    st.slider("Guest popularity (%)", 0, 100, key="guest", disabled=not st.session_state.has_guest)
    st.slider("Ad breaks", 0, 6, key="ads")
    st.segmented_control("Tone of the episode", SENTS, key="sentiment", selection_mode="single")
    st.selectbox("Publication day", DAYS, key="day")
    st.pills("Publication time", TIMES, key="time", selection_mode="single")
    st.number_input("Episode number", 1, 500, key="ep_no")
    st.caption("Model trained on episodes with up to 3 ad breaks and lengths of 5 to 120 minutes. "
               "Values beyond that are extrapolation.")

# widgets with selection_mode can return None when deselected; fall back to defaults
for k, fb in (("sentiment", "Neutral"), ("time", "Evening")):
    if st.session_state.get(k) is None:
        st.session_state[k] = fb

# --------------------------------------------------------------------------- header
st.markdown(
    "<div class='brand'><span class='logo'>" +
    "".join(f"<i style='height:{h}px'></i>" for h in (8, 18, 12, 26, 16, 22, 9)) +
    "</span><span class='name'>Dropoff Studio</span></div>"
    "<p class='tag'>Where does your audience press stop? A neural network estimates listening time before you hit publish.</p>",
    unsafe_allow_html=True)

# --------------------------------------------------------------------------- prediction
row = current()
res = run([row], intervals=True).iloc[0]
pred, lo, hi, L = float(res.prediction), float(res.low), float(res.high), float(res.episode_length)
known = bool(res.length_known)
ret = pred / L if L else 0.0
pct, lo_p, hi_p = (min(100, 100 * v / L) for v in (pred, lo, hi))

seed = zlib.crc32(f"{row['Podcast_Name']}|{row['Episode_Title']}".encode())
rng = np.random.default_rng(seed)
n_bars = 84
x = np.linspace(0, 1, n_bars)
envelope = 0.55 + 0.35 * np.sin(np.pi * np.clip(x * 1.15, 0, 1)) ** 0.8
heights = np.clip(envelope * (0.45 + 0.75 * rng.random(n_bars)), 0.16, 1.0) * 100
k_on = int(round(n_bars * pct / 100))
bars = "".join(f"<i class='b{' on' if i < k_on else ''}' style='height:{h:.0f}%;animation-delay:{i * 8}ms'></i>"
               for i, h in enumerate(heights))
ads_n = int(row["Number_of_Ads"])
ticks = "".join(f"<u style='left:{100 * (i + 1) / (ads_n + 1):.1f}%'></u>" for i in range(ads_n))
agree = res.mc_std
agree_txt = "Strong" if agree < 0.8 else "Moderate" if agree < 1.3 else "Weak"

if ret >= 0.78:
    verdict = "Listeners stay almost to the end."
elif ret >= 0.6:
    verdict = "Most listeners make it past the halfway mark."
else:
    verdict = "A big share of the audience leaves early."

st.markdown(
    f"<div class='hero'><div class='headline'>The typical listener stops around minute <b>{pred:.0f}</b>.</div>"
    f"<div class='sub'>{verdict} That is {ret:.0%} of a {L:.0f}-minute episode"
    f"{'' if known else ' (length unknown, so a typical length is assumed)'}.</div>"
    f"<div class='wave'><div class='track'>{bars}"
    f"<div class='band' style='left:{lo_p:.1f}%;width:{max(hi_p - lo_p, 0.5):.1f}%'></div>"
    f"<div class='mark' style='left:{pct:.1f}%'><span>{pred:.0f} min</span></div></div>"
    f"<div class='ticks'>{ticks}</div>"
    f"<div class='axis'><span>0</span><span>{L / 2:.0f}</span><span>{L:.0f} min</span></div></div>"
    f"<div class='legend'><span><s style='background:{C['on']}'></s>listened</span>"
    f"<span><s style='background:#2B2860'></s>not reached</span>"
    f"<span><s style='background:rgba(255,184,77,.35)'></s>90% range</span>"
    f"<span><s style='background:{C['mark']};clip-path:polygon(50% 0,0 100%,100% 100%)'></s>ad break</span></div>"
    f"<div class='stats'>"
    f"<div class='stat'><div class='v'>{pred:.1f} min</div><div class='l'>expected listening time</div></div>"
    f"<div class='stat'><div class='v'>{ret:.0%}</div><div class='l'>of the episode heard</div></div>"
    f"<div class='stat'><div class='v'>{lo:.0f} to {hi:.0f}</div><div class='l'>minutes, 90% of the time</div></div>"
    f"<div class='stat'><div class='v'>{agree_txt}</div><div class='l'>network agreement (spread {agree:.1f} min)</div></div>"
    f"</div></div>", unsafe_allow_html=True)

if row["Number_of_Ads"] > 3 or (known and (row["Episode_Length_minutes"] > 120)):
    st.markdown("<div class='note'>These inputs sit outside the range the model was trained on, so treat the number as a rough guide.</div>",
                unsafe_allow_html=True)

# --------------------------------------------------------------------------- levers
def build_levers() -> list[tuple[str, float]]:
    out, base = [], pred
    def with_(**kw):
        r = dict(row); r.update(kw); return r
    cand: list[tuple[str, dict]] = []
    if row["Number_of_Ads"] > 0:
        cand.append(("Drop one ad break", with_(Number_of_Ads=row["Number_of_Ads"] - 1)))
    cand.append(("Add one ad break", with_(Number_of_Ads=row["Number_of_Ads"] + 1)))
    if row["Episode_Sentiment"] != "Positive":
        cand.append(("Make the tone more positive", with_(Episode_Sentiment="Positive")))
    if not np.isnan(row["Guest_Popularity_percentage"]):
        cand.append(("Book a bigger guest (+20 points)", with_(Guest_Popularity_percentage=min(100, row["Guest_Popularity_percentage"] + 20))))
    else:
        cand.append(("Add a mid-profile guest", with_(Guest_Popularity_percentage=55.0)))
    cand.append(("Grow host popularity (+10 points)", with_(Host_Popularity_percentage=min(100, row["Host_Popularity_percentage"] + 10))))
    grid = [with_(Publication_Day=d, Publication_Time=t) for d in DAYS for t in TIMES]
    gp = run(grid)
    b = int(np.argmax(gp))
    best_slot = (DAYS[b // 4], TIMES[b % 4])
    preds = run([c[1] for c in cand])
    for (name, _), p in zip(cand, preds):
        out.append((name, float(p - base)))
    if best_slot != (row["Publication_Day"], row["Publication_Time"]) and gp[b] - base > 0.05:
        out.append((f"Publish {best_slot[0]} {best_slot[1].lower()}", float(gp[b] - base)))
    return sorted(out, key=lambda t: -abs(t[1]))[:6]


levers = build_levers()
mx = max(abs(d) for _, d in levers) or 1
rows_html = ""
for name, d in levers:
    w = 50 * abs(d) / mx
    pos = d >= 0
    style_bar = f"left:50%;width:{w:.1f}%;background:{C['on'] if pos else C['loss']}" if pos else \
        f"left:{50 - w:.1f}%;width:{w:.1f}%;background:{C['loss']}"
    rows_html += (f"<div class='lever'><div class='n'>{name}</div><div class='bar'>"
                  f"<i style='left:50%;width:1px;background:rgba(255,255,255,.25)'></i><i style='{style_bar}'></i></div>"
                  f"<div class='d {'pos' if pos else 'neg'}'>{d:+.1f} min</div></div>")
st.markdown("<div class='sect'>What would move this number</div>"
            "<div class='sectsub'>Each row changes one thing about this episode and asks the network to predict again.</div>"
            + rows_html, unsafe_allow_html=True)

# --------------------------------------------------------------------------- tabs
t_lab, t_batch, t_model, t_about = st.tabs(["What-if lab", "Batch predictions", "Model report", "About"])

# ---- What-if lab
with t_lab:
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown("##### Sweep one lever")
        var = st.selectbox("Variable", ["Ad breaks", "Episode length", "Host popularity", "Guest popularity"], label_visibility="collapsed")
        spec = {"Ad breaks": ("Number_of_Ads", np.arange(0, 7, 1.0)),
                "Episode length": ("Episode_Length_minutes", np.linspace(5, 120, 47)),
                "Host popularity": ("Host_Popularity_percentage", np.linspace(0, 100, 41)),
                "Guest popularity": ("Guest_Popularity_percentage", np.linspace(0, 100, 41))}
        col, vals = spec[var]
        rws = []
        for v in vals:
            r = dict(row); r[col] = float(v)
            rws.append(r)
        sw = predictor.predict_with_interval(pd.DataFrame(rws), n_samples=5)
        fig = go.Figure()
        fig.add_traces([go.Scatter(x=vals, y=sw.high, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
                        go.Scatter(x=vals, y=sw.low, mode="lines", fill="tonexty", fillcolor="rgba(143,124,255,.22)", line=dict(width=0),
                                   name="90% range", hoverinfo="skip"),
                        go.Scatter(x=vals, y=sw.prediction, mode="lines+markers", name="prediction",
                                   line=dict(color=C["on"], width=3), marker=dict(size=6))])
        cur = row[col]
        if not np.isnan(cur):
            fig.add_vline(x=cur, line=dict(color=C["mark"], dash="dot"), annotation_text="now", annotation_font_color=C["mark"])
        fig.update_layout(xaxis_title=var, yaxis_title="Predicted listening time (min)", title=f"Listening time vs {var.lower()}",
                          legend=dict(orientation="h", y=-0.28, x=0))
        st.plotly_chart(style(fig), width="stretch")
        if var == "Guest popularity" and np.isnan(row[col]):
            st.caption("Currently no guest; the curve shows what adding one at each popularity level would do.")
    with c2:
        st.markdown("##### Best time to publish")
        g = [dict(row, Publication_Day=d, Publication_Time=t) for d in DAYS for t in TIMES]
        gp = run(g).reshape(7, 4) / L * 100
        fig = go.Figure(go.Heatmap(z=gp, x=TIMES, y=DAYS, colorscale=[[0, "#2B2860"], [.5, "#5E57C9"], [1, C["on"]]],
                                   text=np.round(gp, 1), texttemplate="%{text}%", hovertemplate="%{y} %{x}: %{z:.1f}% heard<extra></extra>",
                                   colorbar=dict(title="% heard", thickness=10)))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="Share of the episode heard, by slot")
        st.plotly_chart(style(fig), width="stretch")
        b = np.unravel_index(np.argmax(gp), gp.shape)
        st.caption(f"Best slot for this episode: **{DAYS[b[0]]} {TIMES[b[1]].lower()}** at {gp[b]:.1f}% heard.")

    st.markdown("##### A/B compare two versions")
    st.caption("Version A uses the sidebar settings. Change a few things for version B and see the difference.")
    b1, b2, b3 = st.columns(3)
    ads_b = b1.number_input("Ad breaks (B)", 0, 6, int(row["Number_of_Ads"]))
    len_b = b2.number_input("Length in minutes (B)", 5.0, 120.0, float(min(max(L, 5), 120)), step=5.0)
    sent_b = b3.selectbox("Tone (B)", SENTS, index=SENTS.index(row["Episode_Sentiment"]))
    pa = float(pred)
    pb = float(run([dict(row, Number_of_Ads=float(ads_b), Episode_Length_minutes=float(len_b), Episode_Sentiment=sent_b)])[0])
    ma, mb = st.columns(2)
    ma.metric("Version A", f"{pa:.1f} min")
    ma.caption(f"{100 * pa / L:.0f}% of the episode heard")
    mb.metric("Version B", f"{pb:.1f} min", f"{pb - pa:+.1f} min vs A")
    mb.caption(f"{100 * pb / len_b:.0f}% of the episode heard")

# ---- Batch
with t_batch:
    st.markdown("##### Score many episodes at once")
    st.caption("Upload a CSV with these columns: " + ", ".join(REQUIRED) + " (plus Episode_Title or Episode_Number).")
    up = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
    use_sample = st.toggle("Use the bundled sample (500 episodes)", value=up is None)
    src = None
    if up is not None:
        src = pd.read_csv(up)
    elif use_sample and (ROOT / "data" / "sample_batch.csv").exists():
        src = pd.read_csv(ROOT / "data" / "sample_batch.csv")
    if src is not None:
        miss = [c for c in REQUIRED if c not in src.columns]
        if miss:
            st.error(f"Missing columns: {', '.join(miss)}")
        else:
            with st.spinner("Scoring…"):
                out = predictor.predict_with_interval(src, n_samples=10)
            res_df = src.copy()
            res_df["predicted_minutes"] = out.prediction.round(2)
            res_df["low_90"] = out.low.round(2)
            res_df["high_90"] = out.high.round(2)
            res_df["predicted_retention"] = (out.prediction / out.episode_length).round(3)
            m1, m2, m3 = st.columns(3)
            m1.metric("Episodes scored", f"{len(res_df):,}")
            m2.metric("Average predicted listening", f"{out.prediction.mean():.1f} min")
            m3.metric("Average predicted retention", f"{(out.prediction / out.episode_length).mean():.0%}")
            h = px.histogram(res_df, x="predicted_minutes", nbins=40, color_discrete_sequence=[C["lilac"]],
                             title="Distribution of predicted listening time")
            st.plotly_chart(style(h, 300), width="stretch")
            if "Genre" in res_df:
                gb = res_df.groupby("Genre")["predicted_retention"].mean().sort_values()
                fg = go.Figure(go.Bar(x=gb.values, y=gb.index, orientation="h", marker_color=C["on"]))
                fg.update_layout(title="Average predicted retention by genre", xaxis_tickformat=".0%")
                st.plotly_chart(style(fg, 330), width="stretch")
            st.dataframe(res_df, width="stretch", height=300)
            st.download_button("Download predictions (CSV)", res_df.to_csv(index=False).encode(),
                               "podcast_predictions.csv", "text/csv", type="primary")

# ---- Model report
with t_model:
    metrics, hist, imp = load_json("metrics"), load_json("history"), load_json("feature_importance")
    if not metrics:
        st.info("No metrics.json yet. Run the notebook to generate the model report.")
    else:
        t = metrics["test"]
        k1, k2, k3, k4 = st.columns(4)
        for col, (v, l) in zip((k1, k2, k3, k4), ((f"{t['rmse']:.2f}", "test RMSE (minutes)"), (f"{t['mae']:.2f}", "test MAE (minutes)"),
                                                  (f"{t['r2']:.3f}", "test R²"), (f"{metrics.get('coverage_90', 0):.0%}", "interval coverage (target 90%)"))):
            col.markdown(f"<div class='mcard'><div class='v'>{v}</div><div class='l'>{l}</div></div>", unsafe_allow_html=True)
        if "data/kaggle" not in metrics.get("data_source", ""):
            st.markdown("<div class='note'>This model was trained on the bundled <b>synthetic</b> dataset, so these numbers describe "
                        "the synthetic world. Drop the real Kaggle <code>train.csv</code> into <code>data/kaggle/</code> and re-run "
                        "the notebook to train on real data.</div>", unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            allm = {**metrics.get("baselines", {}), "Deep Wide & Deep": t}
            df_b = pd.DataFrame({k: v["rmse"] for k, v in allm.items()}, index=["rmse"]).T.sort_values("rmse", ascending=False)
            fb = go.Figure(go.Bar(x=df_b.rmse, y=df_b.index, orientation="h",
                                  marker_color=[C["on"] if "Deep" in i else C["line"] for i in df_b.index],
                                  text=[f"{v:.2f}" for v in df_b.rmse], textposition="outside"))
            fb.update_layout(title="Test RMSE by model (lower is better)")
            st.plotly_chart(style(fb, 320), width="stretch")
        with b:
            if imp:
                s = pd.Series(imp).sort_values()
                fi = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=C["lilac"]))
                fi.update_layout(title="What the model relies on (permutation importance)", xaxis_title="RMSE increase (min)")
                st.plotly_chart(style(fi, 320), width="stretch")
        a, b = st.columns(2)
        with a:
            if hist:
                ys = hist.get("y_std", 1.0)
                fh = go.Figure()
                fh.add_scatter(y=np.array(hist["rmse"]) * ys, name="train", line=dict(color=C["lilac"], width=3))
                fh.add_scatter(y=np.array(hist["val_rmse"]) * ys, name="validation", line=dict(color=C["mark"], width=3))
                fh.update_layout(title="Training curve", xaxis_title="epoch", yaxis_title="RMSE (min)")
                st.plotly_chart(style(fh, 330), width="stretch")
        with b:
            tp = MODEL_DIR / "test_predictions.csv"
            if tp.exists():
                d = pd.read_csv(tp)
                fp = px.density_heatmap(d, x="Listening_Time_minutes", y="pred", nbinsx=45, nbinsy=45,
                                        color_continuous_scale=[[0, "#1A1740"], [.4, "#5E57C9"], [1, C["on"]]])
                fp.add_scatter(x=[0, 115], y=[0, 115], mode="lines", line=dict(color=C["mark"], dash="dot"), showlegend=False)
                fp.update_layout(title="Predicted vs actual (held-out test episodes)", xaxis_title="actual (min)", yaxis_title="predicted (min)",
                                 coloraxis_showscale=False)
                st.plotly_chart(style(fp, 330), width="stretch")
        st.caption(f"Architecture: Wide & Deep network with entity embeddings and residual blocks, {metrics.get('params', 0):,} parameters. "
                   f"Trained on {metrics['n_train']:,} episodes, validated on {metrics['n_val']:,}, tested on {metrics['n_test']:,}.")

# ---- About
with t_about:
    st.markdown("""
##### How it works
1. **Features.** Five categorical columns (podcast, genre, day, time slot, tone) become learned embeddings. Fifteen numeric features
   include episode length, host and guest popularity, ads per hour, interactions and flags that record whether a value was missing.
2. **Network.** A *deep* tower of residual dense blocks learns interactions, while a *wide* linear path handles the dominant
   length-to-listening relationship. Both are summed to give the prediction.
3. **Limits.** Outputs are clipped to what is physically possible: never negative, never longer than the episode.
4. **Uncertainty.** The 90% range is calibrated on held-out residuals relative to episode length. The network agreement score
   comes from Monte-Carlo dropout: the model is run 30 times with dropout active and the spread is reported.

##### Using your own data
Place the Kaggle *Playground Series S5E4* `train.csv` in `data/kaggle/`, re-run the notebook, and restart this app.
No code changes are needed; the vocabularies, scaling and interval calibration are all refitted from the data.

##### Good to know
Predictions are statistical estimates for planning, not guarantees. A single episode's real audience also depends on things
the data does not contain, such as the topic of that particular episode or how it is promoted.
""")
