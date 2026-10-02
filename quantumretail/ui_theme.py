"""Shared visual system for the Streamlit apps.

Design references (see docs/DESIGN.md): Linear's near-black canvas with layered graphite surfaces and one
restrained accent, Vercel/Stripe hairline borders and tabular numerals, PostHog's rule of using the accent
sparingly so charts and tables stay legible. Neutrals are shared; each project picks one accent.
"""
from __future__ import annotations

import html

import streamlit as st

TOKENS = {
    "bg": "#08090A", "s1": "#0D0E10", "s2": "#131417", "s3": "#1A1C20",
    "t1": "#EDEEF0", "t2": "#A1A5AD", "t3": "#6B7078",
    "ok": "#3DD68C", "warn": "#F5B93E", "bad": "#F2555A",
}
COLORWAY = ["{acc}", "#A78BFA", "#38BDF8", "#3DD68C", "#F5B93E", "#F472B6"]
TONES = {"neutral": "t2", "ok": "ok", "warn": "warn", "bad": "bad", "accent": "acc"}

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg:%(bg)s;--s1:%(s1)s;--s2:%(s2)s;--s3:%(s3)s;--b1:rgba(255,255,255,.07);--b2:rgba(255,255,255,.13);
--t1:%(t1)s;--t2:%(t2)s;--t3:%(t3)s;--acc:%(acc)s;--ok:%(ok)s;--warn:%(warn)s;--bad:%(bad)s;}
html,body,.stApp,[class*="st-"]{font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;}
.stApp{background:var(--bg);color:var(--t1);
background-image:radial-gradient(900px 420px at 12%% -8%%,color-mix(in srgb,var(--acc) 16%%,transparent),transparent 65%%),
radial-gradient(700px 360px at 95%% 0%%,rgba(255,255,255,.03),transparent 70%%);background-repeat:no-repeat;}
header[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"],[data-testid="stDeployButton"],#MainMenu,footer{display:none!important;}
.block-container{padding-top:2.4rem;padding-bottom:4rem;max-width:1240px;}
h1,h2,h3,h4{letter-spacing:-.022em;font-weight:600;color:var(--t1);}
h2{font-size:1.35rem;} h3{font-size:1.1rem;}
p,li,label,.stMarkdown{color:var(--t2);line-height:1.6;}
strong{color:var(--t1);font-weight:600;}
a{color:var(--acc);text-decoration:none;} a:hover{text-decoration:underline;}
hr{border-color:var(--b1);}
section[data-testid="stSidebar"]{background:var(--s1);border-right:1px solid var(--b1);}
section[data-testid="stSidebar"] h1{font-size:1.15rem;}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:var(--t3);}
[data-testid="stMetric"]{background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.01));
border:1px solid var(--b1);border-radius:14px;padding:16px 18px;transition:border-color .2s,transform .2s;}
[data-testid="stMetric"]:hover{border-color:var(--b2);}
[data-testid="stMetricLabel"] p{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--t3);font-weight:500;}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] *{color:var(--t1)!important;}
[data-testid="stMetricValue"]{font-size:28px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.025em;color:var(--t1);}
[data-testid="stMetricDelta"]{font-size:12px;}
div[data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid var(--b1);}
button[data-baseweb="tab"]{background:transparent!important;color:var(--t3);font-size:13.5px;font-weight:500;padding:10px 16px;}
button[data-baseweb="tab"]:hover{color:var(--t1);}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--t1);}
div[data-baseweb="tab-highlight"]{background:var(--acc)!important;height:2px;}
div[data-baseweb="tab-border"]{display:none;}
.stButton>button,.stDownloadButton>button{background:var(--s2);border:1px solid var(--b2);color:var(--t1);border-radius:9px;
font-weight:500;font-size:13.5px;padding:.45rem .95rem;transition:all .18s ease;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--acc);background:var(--s3);color:var(--t1);transform:translateY(-1px);}
.stButton>button[kind="primary"]{background:var(--acc);border-color:var(--acc);color:#07080A;font-weight:600;
box-shadow:0 10px 28px -10px var(--acc);}
.stButton>button[kind="primary"]:hover{filter:brightness(1.08);color:#07080A;}
div[data-baseweb="input"]>div,div[data-baseweb="select"]>div,div[data-baseweb="textarea"],textarea,.stTextInput input{
background:var(--s2)!important;border:1px solid var(--b1)!important;border-radius:9px!important;color:var(--t1)!important;}
div[data-baseweb="input"]>div:focus-within,div[data-baseweb="select"]>div:focus-within,div[data-baseweb="textarea"]:focus-within{
border-color:var(--acc)!important;box-shadow:0 0 0 3px color-mix(in srgb,var(--acc) 22%%,transparent)!important;}
[data-testid="stChatInput"]{background:var(--s2);border:1px solid var(--b2);border-radius:14px;}
div[data-baseweb="slider"] [role="slider"]{background:var(--acc)!important;box-shadow:0 0 0 4px color-mix(in srgb,var(--acc) 25%%,transparent);}
span[data-testid="stIconMaterial"],.material-symbols-rounded,[class*="material-symbols"]{font-family:'Material Symbols Rounded','Material Symbols Outlined'!important;}
code,pre,kbd{font-family:'JetBrains Mono',ui-monospace,Consolas,monospace!important;font-size:12.5px;}
[data-testid="stCode"] pre,.stCodeBlock pre{background:var(--s1)!important;border:1px solid var(--b1);border-radius:12px;}
:not(pre)>code{background:var(--s3);color:var(--t1);border-radius:6px;padding:.1em .4em;}
[data-testid="stDataFrame"]{border:1px solid var(--b1);border-radius:12px;overflow:hidden;}
[data-testid="stAlert"],[data-testid="stAlertContainer"]{border-radius:12px;border:1px solid var(--b1)!important;background:var(--s2)!important;color:var(--t1);}
[data-testid="stAlert"] p{color:var(--t1);}
[data-testid="stExpander"]{border:1px solid var(--b1)!important;border-radius:12px;background:var(--s1);}
[data-testid="stExpander"] summary{color:var(--t1);font-weight:500;}
[data-testid="stPlotlyChart"],[data-testid="stArrowVegaLiteChart"]{border:1px solid var(--b1);border-radius:14px;padding:8px;background:var(--s1);}
[data-testid="stSpinner"] p{color:var(--t2);}
::-webkit-scrollbar{width:10px;height:10px;} ::-webkit-scrollbar-thumb{background:var(--s3);border-radius:8px;border:2px solid var(--bg);}
/* custom components */
.ds-hero{position:relative;padding:30px 32px 28px;border:1px solid var(--b1);border-radius:20px;margin:0 0 22px;overflow:hidden;
background:linear-gradient(180deg,rgba(255,255,255,.035),rgba(255,255,255,.008));}
.ds-hero:before{content:"";position:absolute;inset:-1px;border-radius:20px;pointer-events:none;
background:radial-gradient(520px 180px at 0%% 0%%,color-mix(in srgb,var(--acc) 22%%,transparent),transparent 70%%);}
.ds-eyebrow{position:relative;font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--acc);font-weight:600;margin-bottom:10px;}
.ds-title{position:relative;font-size:34px;line-height:1.12;font-weight:650;letter-spacing:-.035em;margin:0 0 10px;
background:linear-gradient(180deg,#fff 30%%,#9CA0AA);-webkit-background-clip:text;background-clip:text;color:transparent;}
.ds-sub{position:relative;font-size:15px;color:var(--t2);max-width:760px;line-height:1.55;margin:0 0 16px;}
.ds-chips{position:relative;display:flex;flex-wrap:wrap;gap:8px;}
.ds-chip{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:500;padding:4px 10px;border-radius:999px;
border:1px solid var(--b2);color:var(--c);background:color-mix(in srgb,var(--c) 9%%,transparent);}
.ds-chip:before{content:"";width:6px;height:6px;border-radius:50%%;background:var(--c);}
.ds-section{margin:26px 0 10px;} .ds-section h3{margin:0;font-size:1.05rem;} .ds-section p{margin:4px 0 0;font-size:13px;color:var(--t3);}
.ds-card{margin-bottom:14px;border:1px solid var(--b1);border-radius:14px;padding:16px 18px;background:linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,.008));}
.ds-card .k{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--t3);font-weight:500;}
.ds-card .v{font-size:26px;font-weight:600;letter-spacing:-.025em;color:var(--t1);font-variant-numeric:tabular-nums;margin-top:4px;}
.ds-card .h{font-size:12.5px;color:var(--t3);margin-top:4px;}
.ds-banner{display:flex;gap:12px;align-items:flex-start;border:1px solid color-mix(in srgb,var(--c) 38%%,transparent);
background:color-mix(in srgb,var(--c) 8%%,transparent);border-radius:12px;padding:12px 16px;margin:8px 0 14px;}
.ds-banner .l{font-size:11.5px;letter-spacing:.1em;font-weight:700;color:var(--c);text-transform:uppercase;white-space:nowrap;margin-top:2px;}
.ds-banner .t{font-size:14px;color:var(--t1);line-height:1.5;}
.ds-step{display:flex;gap:12px;align-items:baseline;padding:9px 0;border-bottom:1px dashed var(--b1);font-size:13.5px;color:var(--t2);}
.ds-step:last-child{border-bottom:none;}
.ds-step .n{font-family:'JetBrains Mono',monospace;font-size:12.5px;color:var(--acc);min-width:92px;font-weight:500;}
.ds-step .ms{margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--t3);white-space:nowrap;}
.ds-quote{border-left:2px solid var(--acc);padding:6px 14px;margin:6px 0 12px;color:var(--t2);font-size:13.5px;background:color-mix(in srgb,var(--acc) 5%%,transparent);border-radius:0 8px 8px 0;}
.ds-quote b{color:var(--t1);}
"""


def _tone_var(tone: str) -> str:
    return f"var(--{TONES.get(tone, 't2')})"


def apply(accent: str, title: str, icon: str = "◆") -> None:
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown(f"<style>{_CSS % {**TOKENS, 'acc': accent}}</style>", unsafe_allow_html=True)


def chip(text: str, tone: str = "neutral") -> str:
    return f'<span class="ds-chip" style="--c:{_tone_var(tone)}">{html.escape(text)}</span>'


def hero(eyebrow: str, title: str, subtitle: str, chips: list[tuple[str, str]] | None = None) -> None:
    row = "".join(chip(t, tone) for t, tone in (chips or []))
    st.markdown(
        f'<div class="ds-hero"><div class="ds-eyebrow">{html.escape(eyebrow)}</div>'
        f'<div class="ds-title">{html.escape(title)}</div><div class="ds-sub">{html.escape(subtitle)}</div>'
        f'<div class="ds-chips">{row}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, sub: str | None = None) -> None:
    p = f"<p>{html.escape(sub)}</p>" if sub else ""
    st.markdown(f'<div class="ds-section"><h3>{html.escape(title)}</h3>{p}</div>', unsafe_allow_html=True)


def banner(label: str, text: str, tone: str = "accent") -> None:
    st.markdown(
        f'<div class="ds-banner" style="--c:{_tone_var(tone)}"><div class="l">{html.escape(label)}</div>'
        f'<div class="t">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def card(label: str, value: str, hint: str = "") -> str:
    h = f'<div class="h">{html.escape(hint)}</div>' if hint else ""
    return f'<div class="ds-card"><div class="k">{html.escape(label)}</div><div class="v">{html.escape(value)}</div>{h}</div>'


def cards(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (k, v, h) in zip(cols, items):
        col.markdown(card(k, v, h), unsafe_allow_html=True)


def steps(rows: list[tuple[str, str, str]]) -> None:
    """rows of (name, detail, right-aligned text such as '12 ms')."""
    body = "".join(
        f'<div class="ds-step"><span class="n">{html.escape(n)}</span><span>{html.escape(d)}</span>'
        f'<span class="ms">{html.escape(m)}</span></div>'
        for n, d, m in rows
    )
    st.markdown(f'<div class="ds-card">{body}</div>', unsafe_allow_html=True)


def quote(label: str, text: str) -> None:
    st.markdown(f'<div class="ds-quote"><b>{html.escape(label)}</b> {html.escape(text)}</div>', unsafe_allow_html=True)


def style_fig(fig, accent: str, height: int | None = None):
    """Apply the dark chart theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=TOKENS["t2"], size=12),
        colorway=[c.format(acc=accent) for c in COLORWAY],
        margin=dict(l=12, r=12, t=28, b=12),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11.5)),
        hoverlabel=dict(bgcolor=TOKENS["s3"], bordercolor="rgba(255,255,255,.15)", font=dict(family="Inter", color=TOKENS["t1"])),
        xaxis=dict(gridcolor="rgba(255,255,255,.05)", zerolinecolor="rgba(255,255,255,.08)", linecolor="rgba(255,255,255,.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.05)", zerolinecolor="rgba(255,255,255,.08)", linecolor="rgba(255,255,255,.08)"),
    )
    if height:
        fig.update_layout(height=height)
    return fig
