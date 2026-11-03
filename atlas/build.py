#!/usr/bin/env python3
"""
build.py — generate a self-contained atlas.html for the Research Atlas.

Pipeline: read lab members from the website → fetch OpenAlex → fill missing
          abstracts from PubMed → embed → UMAP 3D → HDBSCAN → place
          abstract-less papers by nearest neighbours → label clusters
          (lab vocabulary first, then MeSH / keywords / TF-IDF) → write HTML

The output is a single atlas.html: plain HTML/CSS/JS with Three.js bundled inline.
No server needed — just double-click to open.
"""

import json, math, os, re, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
import umap
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# ── CONFIG ────────────────────────────────────────────────────────────────────
DIMS   = 3
MAILTO = "jonathan.adams@kuleuven.be"

# Filled from the website (content/people) when the script runs — see below.
AUTHOR_IDS = []

# ── Lab members: taken from the website (content/people/*.md) ──
# Everyone with an ORCID iD and/or OpenAlex author ID on the website is
# included automatically; ORCIDs are looked up on OpenAlex to find the author
# ID. The website's spelling of each name is used in the author dropdown.
# The script reads the website repo it sits in (WEBSITE_DIR) first, then (if
# WEBSITE_GITHUB is set) downloads it from GitHub. A private repo needs a token in
# the GITHUB_TOKEN environment variable (read-only access is enough).
# If neither works, the script stops with a message (nothing to build).
# Default: this script lives in the website repo, in a folder "atlas/", so
# the website is the folder one level up. Change only if you keep it elsewhere.
WEBSITE_DIR    = Path(__file__).resolve().parent.parent
WEBSITE_GITHUB = "JonathanAdams02/lab-site"     # owner/repo, or None
WEBSITE_BRANCH = "main"
# Extra OpenAlex IDs to always include on top of the website (e.g. a second,
# split OpenAlex profile of the same person). The website's "OpenAlex author
# ID" field also accepts several IDs separated by commas.
EXTRA_AUTHOR_IDS = []

# OpenAlex needs a free API key for cursor paging (openalex.org/settings/api).
# Set it in the OPENALEX_API_KEY environment variable. Without a key the script
# falls back to page-number paging (fine up to 10,000 works).
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "")

INSTITUTION_ID = None

# Authors stored per paper: the first MAX_AUTHORS, plus the last (senior)
# author and any PI in between, so the author filter still finds them.
MAX_AUTHORS = 8

# ── Which works to leave out ──
# OpenAlex work types to drop. Letters, editorials, reviews, book chapters
# etc. are kept. "peer-review" = the "Author response for …" records.
EXCLUDE_TYPES = {"peer-review", "erratum", "paratext", "retraction"}

# Drop conference proceedings / meeting abstracts (see is_conference()).
EXCLUDE_CONFERENCE = True

# Only keep works where at least one lab member (AUTHOR_IDS) is listed with a
# KU Leuven / UZ Leuven affiliation. This removes papers OpenAlex wrongly put
# in a lab member's profile (other people with the same name), but also
# papers from before someone joined Leuven, and papers where OpenAlex has no
# affiliation at all. Set to False to switch the check off.
REQUIRE_LEUVEN = True

# Papers where OpenAlex recorded NO affiliation for the lab member: keep them
# (True) or drop them (False). Many genuine lab papers lack affiliation data.
# They are only kept when the name printed on the paper matches the person's
# name on the website (see CHECK_NAMES), which keeps out other people with a
# similar name that OpenAlex merged into a lab member's profile.
KEEP_NO_AFFILIATION = True

# Drop a paper when the lab member's name on it clearly isn't theirs, e.g.
# "Nicolas Bouckaert" in Filip Bouckaert's profile, or "Luc Van den hove" in
# Laura Van Hove's. Compares surname + first name (or initial) with the website.
CHECK_NAMES = True

# Matched (case-insensitive) against OpenAlex institution names and the raw
# affiliation text. Plain "Leuven" is deliberately not enough (imec, etc.).
LEUVEN_PATTERN = re.compile(
    r"\bk\.?\s?u\.?\s?leuven"                       # KU Leuven, K.U.Leuven, K.U. Leuven
    r"|katholieke universiteit leuven"
    r"|catholic university (?:of )?leuven"
    r"|university of leuven|leuven university"
    r"|\buz\s?leuven"                               # UZ Leuven
    r"|universit(?:y|air)e?\s+(?:hospitals?|ziekenhui(?:s|zen))\s+(?:gasthuisberg\s+)?leuven"
    r"|university hospitals? (?:of )?leuven"
    r"|gasthuisberg"
    r"|\bupc\s?(?:ku\s?)?leuven",                   # UPC KU Leuven
    re.IGNORECASE,
)

# Many journals (notably Elsevier) don't share abstracts with OpenAlex. For
# papers without one, look the abstract up on PubMed (by PMID, or by DOI).
# Results are cached, so only new papers are looked up on later runs.
# Delete pubmed_abstracts.json to look everything up again.
USE_PUBMED   = True
PUBMED_CACHE = Path(__file__).resolve().parent / "pubmed_abstracts.json"

# ── Papers that still have no abstract (after PubMed) ──
# Only these OpenAlex types are kept without an abstract; anything else
# (book chapters, books, datasets, …) is only kept when it has an abstract.
NO_ABSTRACT_KEEP_TYPES = {"article", "review", "letter"}

# Abstract-less papers are not used to form clusters. Afterwards each one is
# placed in the cluster of the most similar papers, compared on title +
# keywords + topics + journal.
ASSIGN_K       = 15     # how many nearest papers vote on the cluster
MIN_ASSIGN_SIM = 0.35   # below this similarity to its nearest paper → left unclustered (grey)

# How much an abstract-less paper counts when naming clusters, relative to a
# paper with an abstract (1.0 = same weight, 0 = ignored).
NO_ABSTRACT_WEIGHT = 0.3

OUT_PATH = Path(__file__).resolve().parent / "atlas.html"

EMBED_MODEL      = "all-MiniLM-L6-v2"
MIN_CLUSTER_SIZE = 30   # now actually wired into HDBSCAN below — lower = more, smaller clusters
RANDOM_STATE     = 42

# Set to True for verbose cluster-labeling debug output (tier used, top
# candidates + scores considered at each stage, why earlier tiers were
# skipped). Off by default since it's noisy for large runs.
DEBUG_LABELS = True

# How much a term's frequency in OTHER clusters penalizes it when scoring
# cluster labels: score = freq_in_cluster / (freq_everywhere ** DISCRIMINATIVE_POWER + 1)
#   1.0 = full c-TF-IDF discrimination (terms common lab-wide get buried,
#         even if they're a big part of what a given cluster is about)
#   0.0 = pure raw frequency within the cluster, cross-cluster spread ignored
#         entirely (labels become "most talked about in this cluster",
#         not "most distinctive to this cluster")
# Lower this if recurring lab-wide themes (e.g. "social cognition") keep
# losing out to rarer, noisier terms just because they're common everywhere.
DISCRIMINATIVE_POWER = 0.5

# ── Fixed research themes ──
# With USE_THEMES = True the atlas is organised by these themes instead of
# automatic clusters. Each paper gets the colour of the theme it is closest to
# in content (compared with the theme's name, description and keywords, and
# smoothed over its nearest neighbours so colours form coherent regions).
# The layout is the normal organic UMAP map (themes play no part in it).
# THEME_PULL then moves every paper that fraction of the way towards the centre
# of its theme: 0 = the map exactly as UMAP made it, coloured by theme;
# 0.2–0.35 = themes visibly grouped while keeping their organic shapes;
# 1 = each theme collapsed onto one point. Because every paper moves by the
# same proportion, papers between two themes stay between them.
# Edit names/descriptions/keywords freely; order = colour order.
USE_THEMES          = True
THEME_PULL          = 0.25
THEME_SMOOTH_K      = 10    # neighbours that vote on a paper's theme (1 = the paper alone)
THEMES = [
  {"name": "Social cognition",
   "description": "How the brain supports understanding other people: recognising emotions from faces, bodies and voices, empathy, theory of mind and social behaviour, and the role of the amygdala, insula and temporal cortex, studied with fMRI and in patients with brain lesions or neurological and psychiatric conditions.",
   "keywords": ["social cognition", "emotion recognition", "face perception", "body perception", "facial expression", "empathy", "theory of mind", "amygdala", "interoception", "insula"]},
  {"name": "Frontotemporal dementia",
   "description": "Frontotemporal dementia and related disorders: behavioural variant FTD, primary progressive aphasia and FTD with ALS, including genetic forms (C9orf72, GRN, MAPT) studied in cohorts such as GENFI; clinical symptoms, cognition, brain imaging, fluid biomarkers and disease progression.",
   "keywords": ["frontotemporal dementia", "behavioural variant FTD", "primary progressive aphasia", "amyotrophic lateral sclerosis", "C9orf72", "GRN", "MAPT", "GENFI", "frontotemporal lobar degeneration"]},
  {"name": "Late-onset psychiatric disorders",
   "description": "Psychiatric illness in later life and the ageing brain: late-life depression, late-onset psychosis and neuropsychiatric symptoms of neurodegeneration; brain ageing, white matter and diffusion MRI, amyloid, tau and synaptic density PET-MR imaging, Alzheimer's disease, mild cognitive impairment and care for older adults with dementia.",
   "keywords": ["late-life depression", "old age", "ageing", "white matter", "diffusion MRI", "amyloid", "tau", "synaptic density", "PET-MR", "Alzheimer's disease", "mild cognitive impairment", "dementia care", "older adults"]},
  {"name": "Neuromodulation",
   "description": "Brain stimulation treatments in psychiatry: electroconvulsive therapy (ECT), transcranial magnetic stimulation (TMS), transcranial alternating current stimulation (tACS) and deep brain stimulation (DBS); clinical outcome, relapse, cognitive side effects and brain changes after treatment, mainly in severe and treatment-resistant depression.",
   "keywords": ["electroconvulsive therapy", "ECT", "transcranial magnetic stimulation", "TMS", "tACS", "deep brain stimulation", "DBS", "neuromodulation", "treatment-resistant depression", "relapse"]},
  {"name": "Computational psychiatry",
   "description": "Quantitative and computational approaches to mental health: experience sampling and ecological momentary assessment of affect and symptoms in daily life, dynamic and statistical models of psychopathology, machine learning and artificial intelligence, digital tools and multimodal sensing, applied to mood, personality, eating and psychotic disorders.",
   "keywords": ["computational psychiatry", "experience sampling", "ecological momentary assessment", "affect dynamics", "machine learning", "artificial intelligence", "statistical model", "digital", "multimodal sensing", "psychopathology"]},
]

# ── Naming clusters (used when USE_THEMES = False) ──
# "combined" (default): every term a paper has (MeSH, OpenAlex keywords and
# topics, two/three-word phrases from its title, and phrases from the lab's own
# vocabulary found in its title/abstract) competes in one ranking. A term
# scores high when many of the cluster's papers have it AND it is much rarer
# in the other clusters. Lab terms (FRIS project keywords, homepage themes,
# people's research focus — read from the website) get a small bonus.
# "mesh": the original MeSH → keywords → TF-IDF naming.
NAMING_METHOD = "combined"
NAME_MIN_PAPERS       = 3      # a term must occur in at least this many papers of the cluster
NAME_MIN_SECOND_SHARE = 0.10   # the second term must cover at least 10% of the cluster
NAME_LAB_BONUS        = 1.3    # boost for terms from the lab's own vocabulary
# Your own extra candidate terms (treated like lab vocabulary):
LAB_VOCAB_EXTRA = []
# Words that never make a label on their own (too general for this lab):
NAME_GENERIC_WORDS = {
    "research", "study", "studies", "patient", "patients", "human", "humans", "brain",
    "disease", "diseases", "disorder", "disorders", "analysis", "data", "result", "results",
    "effect", "effects", "role", "new", "case", "review", "clinical", "treatment", "therapy",
    "health", "mental", "medicine", "psychology", "psychiatry", "neuroscience", "science",
    "method", "methods", "approach", "evidence", "model", "models", "advanced", "imaging",
    "psychological", "social", "neural", "behavioral", "medical", "technique", "techniques",
    "application", "applications", "cognitive", "cognition", "emotional", "affective",
    "function", "functional", "structural", "people", "adult", "adults", "aged", "related",
    "outcome", "outcomes", "neurodegenerative", "neuropsychiatry", "multimodal",
}

# Hand-picked names: map an automatic label to the name you want shown. Run
# once, copy a label from the "cluster N: …" lines in the output, add it here.
# (Cluster numbers change between runs; the automatic labels mostly don't.)
LABEL_OVERRIDES = {
    # "Late Life Depression · White Matter": "Late-life depression: brain structure",
}

# NOTE: this is the single source of truth for cluster colors — it gets
# injected into the HTML template at build time (see write_atlas()).
CLUSTER_COLORS = [
    '#c2613f', '#c0902e', '#5a9e57', '#2f9ea0',
    '#3f74c0', '#7b6bcf', '#b057a8', '#b8506b',
    '#d97b4f', '#8a9e3f', '#3f9e7a', '#4f8fc0',
    '#9a5fc0', '#c05f8f', '#6b7b3f', '#3f6b9e',
]

# Filled during normalize() — only the display names of the PIs
PI_AUTHOR_NAMES = set()

# ── FETCH THREE.JS LIBS ───────────────────────────────────────────────────────
def fetch_js_libraries():
    """Download Three.js and OrbitControls to embed directly in the HTML."""
    print("Fetching Three.js libraries for bundling…")
    three_url  = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
    orbit_url  = "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"
    with urllib.request.urlopen(three_url) as r:
        three_js = r.read().decode("utf-8")
    with urllib.request.urlopen(orbit_url) as r:
        orbit_js = r.read().decode("utf-8")
    print("  Three.js libraries fetched.")
    return three_js, orbit_js

# ── HTML TEMPLATE ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Research Atlas 3D</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:#fff;font-family:'IBM Plex Sans',system-ui,sans-serif;color:#1c1c1c;overflow:hidden}

/* ── layout ── */
#app{display:flex;height:100vh;width:100%}

/* ── sidebar ── */
#sidebar{width:312px;flex:none;height:100%;border-right:1px solid #ececea;display:flex;flex-direction:column;background:#fcfcfb}
#sb-head{padding:24px 22px 18px;border-bottom:1px solid #ececea}
#sb-lab{font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:#9a9a93;margin-bottom:8px}
#sb-title{font-size:20px;font-weight:600;letter-spacing:-.01em}
#sb-title span{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#b9b9b2;font-weight:400;vertical-align:middle;margin-left:4px}
#sb-desc{font-size:12.5px;color:#76766f;margin-top:5px;line-height:1.45}
#sb-body{flex:1;overflow-y:auto;padding:20px 22px 22px;display:flex;flex-direction:column;gap:24px}
#sb-foot{padding:14px 22px;border-top:1px solid #ececea;display:flex;justify-content:space-between;align-items:center;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#9a9a93}
#sb-foot #count-vis{color:#3a3a36}#sb-foot #count-sep{color:#b6b6ae}

/* ── sidebar controls ── */
.ctrl-label{font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:#9a9a93;display:block;margin-bottom:8px}
#search{width:100%;padding:9px 11px;border:1px solid #e0e0db;border-radius:8px;font-family:inherit;font-size:13px;color:#1c1c1c;background:#fff;outline:none}
#search:focus{border-color:#9a9a93}
.year-row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}
.year-row .ctrl-label{margin-bottom:0}
#year-display{font-family:'IBM Plex Mono',monospace;font-size:12px;color:#3a3a36}
input[type=range]{accent-color:#2d2d2d;width:100%;height:3px;display:block}
input[type=range]+input[type=range]{margin-top:6px}
.topic-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
#reset-btn{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:#9a9a93;background:transparent;border:none;cursor:pointer;padding:2px 0}
#reset-btn:hover{color:#1c1c1c}
#cluster-list{display:flex;flex-direction:column;gap:2px}
.cluster-btn{display:flex;align-items:center;gap:10px;width:100%;padding:6px 8px;border:none;background:transparent;cursor:pointer;text-align:left;border-radius:7px;font-family:inherit;transition:opacity .15s}
.cluster-btn:hover{background:#f1f1ee}
.cluster-btn .dot{width:10px;height:10px;border-radius:50%;flex:none;border:1px solid rgba(0,0,0,.08)}
.cluster-btn .name{flex:1;font-size:13px;color:#262622}
.cluster-btn .cnt{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#a8a8a1}
select{width:100%;padding:9px 11px;border:1px solid #e0e0db;border-radius:8px;font-family:inherit;font-size:13px;color:#1c1c1c;background:#fff;outline:none;cursor:pointer}

/* ── main canvas area ── */
#main{position:relative;flex:1;height:100%;background:#fff;overflow:hidden;cursor:grab}
#main.dragging{cursor:grabbing}
#canvas-wrap{position:absolute;inset:0}
#hint{position:absolute;top:18px;left:20px;font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:.06em;color:#bcbcb4;pointer-events:none;user-select:none}

/* ── zoom buttons ── */
#zoom-btns{position:absolute;right:18px;bottom:18px;display:flex;flex-direction:column;gap:6px}
.zoom-btn{width:34px;height:34px;border:1px solid #e6e6e1;background:#fff;border-radius:8px;font-size:18px;color:#3a3a36;cursor:pointer;line-height:1;display:flex;align-items:center;justify-content:center}
.zoom-btn:hover{background:#f4f4f1}

/* ── tooltip ── */
#tooltip{position:absolute;pointer-events:none;background:#1c1c1c;color:#fff;padding:8px 11px;border-radius:8px;max-width:260px;box-shadow:0 6px 20px rgba(0,0,0,.18);z-index:5;display:none}
#tooltip.visible{display:block}
#tt-cluster{font-family:'IBM Plex Mono',monospace;font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;color:#b9b9b2;display:flex;align-items:center;gap:7px;margin-bottom:4px}
#tt-dot{width:7px;height:7px;border-radius:50%;flex:none}
#tt-title{font-size:12.5px;line-height:1.35;font-weight:500}

/* ── detail panel ── */
#detail{position:absolute;top:0;right:0;height:100%;width:374px;background:#fff;border-left:1px solid #ececea;box-shadow:-10px 0 30px rgba(0,0,0,.05);display:flex;flex-direction:column;z-index:6;transform:translateX(100%);transition:transform .2s ease}
#detail.open{transform:translateX(0)}
#det-head{padding:22px 26px 0;display:flex;justify-content:space-between;align-items:center}
#det-cluster{display:flex;align-items:center;gap:9px}
#det-dot{width:11px;height:11px;border-radius:50%}
#det-clname{font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:#76766f}
#close-btn{width:30px;height:30px;border:1px solid #eaeae5;background:#fff;border-radius:7px;font-size:16px;color:#76766f;cursor:pointer;line-height:1}
#close-btn:hover{background:#f4f4f1}
#det-body{padding:16px 26px 26px;overflow-y:auto}
#det-title{font-size:19px;font-weight:600;line-height:1.3;letter-spacing:-.01em;margin-bottom:12px;text-wrap:pretty}
#det-meta{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:#76766f;line-height:1.6;margin-bottom:20px}
#det-abs-label{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:#b0b0a8;margin-bottom:8px}
#det-abstract{font-size:14px;line-height:1.65;color:#34342f;margin-bottom:24px;text-wrap:pretty}
#det-link{display:inline-flex;align-items:center;gap:8px;padding:10px 16px;background:#1c1c1c;color:#fff;text-decoration:none;border-radius:8px;font-size:13px;font-weight:500}
#det-link:hover{background:#333}

::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-thumb{background:#deded9;border-radius:5px}
::-webkit-scrollbar-track{background:transparent}
</style>
<script>__THREE_JS__</script>
<script>__ORBIT_JS__</script>
</head>
<body>
<div id="app">

  <!-- SIDEBAR -->
  <aside id="sidebar">
    <div id="sb-head">
      <div id="sb-lab">Neuropsychiatry Lab</div>
      <div id="sb-title">Research Atlas <span>3D</span></div>
      <div id="sb-desc">A semantic map of publications positioned by abstract content.</div>
    </div>
    <div id="sb-body">
      <div>
        <label class="ctrl-label" for="search">Search</label>
        <input id="search" type="text" placeholder="title, abstract, author…">
      </div>
      <div>
        <div class="year-row">
          <label class="ctrl-label">Year</label>
          <span id="year-display"></span>
        </div>
        <input id="yr-min" type="range">
        <input id="yr-max" type="range">
      </div>
      <div>
        <div class="topic-head">
          <label class="ctrl-label" style="margin-bottom:0">Topics</label>
          <button id="reset-btn">reset all</button>
        </div>
        <div id="cluster-list"></div>
      </div>
      <div>
              <label class="ctrl-label" for="author-select">Author</label>
              <select id="author-select"><option value="">All authors</option></select>
            </div>
            <div>
              <label class="ctrl-label">Auto-rotate</label>
              <div style="display:flex;gap:6px">
                <button id="rot-on"  style="flex:1;padding:8px;border:1px solid #e0e0db;border-radius:8px;font-family:'IBM Plex Mono',monospace;font-size:11px;cursor:pointer;background:#1c1c1c;color:#fff">On</button>
                <button id="rot-off" style="flex:1;padding:8px;border:1px solid #e0e0db;border-radius:8px;font-family:'IBM Plex Mono',monospace;font-size:11px;cursor:pointer;background:#fff;color:#9a9a93">Off</button>
              </div>
            </div>
          </div>
    <div id="sb-foot">
      <span><span id="count-vis">0</span><span id="count-sep"> / </span><span id="count-tot">0</span> shown</span>
      <span>v · 2026.06</span>
    </div>
  </aside>

  <!-- CANVAS -->
  <main id="main">
    <div id="canvas-wrap"></div>
    <div id="hint">drag to orbit · scroll to zoom · click a point for details</div>
    <div id="zoom-btns">
      <button class="zoom-btn" id="btn-in">+</button>
      <button class="zoom-btn" id="btn-out">−</button>
      <button class="zoom-btn" id="btn-fit" style="font-size:9.5px;font-family:'IBM Plex Mono',monospace">FIT</button>
    </div>
    <div id="tooltip">
      <div id="tt-cluster"><span id="tt-dot"></span><span id="tt-label"></span></div>
      <div id="tt-title"></div>
    </div>
  </main>

  <!-- DETAIL PANEL -->
  <div id="detail">
    <div id="det-head">
      <div id="det-cluster"><span id="det-dot"></span><span id="det-clname"></span></div>
      <button id="close-btn">×</button>
    </div>
    <div id="det-body">
      <div id="det-title"></div>
      <div id="det-meta"></div>
      <div id="det-abs-label">Abstract</div>
      <div id="det-abstract"></div>
      <a id="det-link" href="#" target="_blank">View publication →</a>
    </div>
  </div>
</div>

<script>
// ── DATA ──────────────────────────────────────────────────────────────────────
const PAPERS_RAW = __PAPERS_DATA__;
const PI_AUTHORS = __PI_AUTHORS__;

const COLORS = __CLUSTER_COLORS__;
const NOISE_COLOR = '#9a9a93';

function hexRgb(h){
  return [
    parseInt(h.slice(1,3),16)/255,
    parseInt(h.slice(3,5),16)/255,
    parseInt(h.slice(5,7),16)/255
  ];
}

// Build cluster map (noise / -1 excluded from map, handled separately)
const clusterMap = {};
PAPERS_RAW.forEach(p => {
  const id = p.cluster;
  if (id === -1) return;
  if (!clusterMap[id]) clusterMap[id] = {
    id, name: p.clusterLabel || 'Cluster ' + id,
    color: COLORS[id % COLORS.length],
    count: 0, cx: 0, cy: 0, cz: 0, n: 0
  };
  clusterMap[id].count++;
  clusterMap[id].cx += p.x || 0;
  clusterMap[id].cy += p.y || 0;
  clusterMap[id].cz += p.z || 0;
  clusterMap[id].n++;
});
Object.values(clusterMap).forEach(c => {
  if (c.n > 0) { c.cx /= c.n; c.cy /= c.n; c.cz /= c.n; }
});
const clusters = Object.values(clusterMap).sort((a, b) => a.id - b.id);
const papers   = PAPERS_RAW.map((p, i) => ({...p, _id: i, cluster: p.cluster ?? -1}));

// Year range
const allYears = papers.map(p => p.year).filter(Boolean);
const yearMin  = Math.min(...allYears), yearMax = Math.max(...allYears);

// ── STATE ─────────────────────────────────────────────────────────────────────
const state = {
  q: '', author: '',
  yMin: yearMin, yMax: yearMax,
  clusters: Object.fromEntries(clusters.map(c => [c.id, true])),
  hoverId: null,
  selected: null
};

function active(p) {
  // noise points (cluster === -1) are fully interactive — no early exit
  if (p.year < state.yMin || p.year > state.yMax) return false;
  if (state.author && !(p.authors || []).includes(state.author)) return false;
  // for clustered points, respect the cluster toggle
  if (p.cluster !== -1 && !state.clusters[p.cluster]) return false;
  if (state.q) {
    const q = state.q.toLowerCase();
    if (!(p.title.toLowerCase().includes(q) ||
          (p.abstract || '').toLowerCase().includes(q) ||
          (p.authors || []).join(' ').toLowerCase().includes(q))) return false;
  }
  return true;
}

// ── SIDEBAR INIT ──────────────────────────────────────────────────────────────
const yrMinEl   = document.getElementById('yr-min');
const yrMaxEl   = document.getElementById('yr-max');
const yrDisplay = document.getElementById('year-display');

[yrMinEl, yrMaxEl].forEach(el => { el.min = yearMin; el.max = yearMax; });
yrMinEl.value = yearMin;
yrMaxEl.value = yearMax;

function updateYearDisplay() {
  yrDisplay.textContent = state.yMin + ' – ' + state.yMax;
}
updateYearDisplay();

function updateCounts() {
  const vis = papers.filter(p => active(p)).length;
  const tot = papers.length;   // includes noise points
  document.getElementById('count-vis').textContent = vis;
  document.getElementById('count-tot').textContent = tot;
}

// Author select — PI names only, injected from build.py
const authorSel = document.getElementById('author-select');
PI_AUTHORS.forEach(a => {
  const o = document.createElement('option');
  o.value = a; o.textContent = a;
  authorSel.appendChild(o);
});

// Cluster buttons
const clusterList = document.getElementById('cluster-list');
clusters.forEach(c => {
  const btn = document.createElement('button');
  btn.className = 'cluster-btn';
  btn.dataset.id = c.id;
  btn.innerHTML = `<span class="dot" style="background:${c.color}"></span><span class="name">${c.name}</span><span class="cnt">${c.count}</span>`;
  clusterList.appendChild(btn);
});

// ── TOOLTIP ───────────────────────────────────────────────────────────────────
const tooltip = document.getElementById('tooltip');

function showTooltip(p, sx, sy) {
  const c = clusterMap[p.cluster];
  document.getElementById('tt-dot').style.background = c ? c.color : NOISE_COLOR;
  document.getElementById('tt-label').textContent    = (c ? c.name : 'Unclustered') + ' · ' + p.year;
  document.getElementById('tt-title').textContent    = p.title;
  tooltip.style.left      = sx + 'px';
  tooltip.style.top       = (sy - 16) + 'px';
  tooltip.style.transform = 'translate(-50%,-100%)';
  tooltip.classList.add('visible');
}
function hideTooltip() { tooltip.classList.remove('visible'); }

// ── DETAIL PANEL ──────────────────────────────────────────────────────────────
const detailEl = document.getElementById('detail');

function showDetail(p) {
  state.selected = p;
  const c = clusterMap[p.cluster];
  document.getElementById('det-dot').style.background = c ? c.color : NOISE_COLOR;
  document.getElementById('det-clname').textContent   = c ? c.name : 'Unclustered';
  document.getElementById('det-title').textContent    = p.title;
  document.getElementById('det-meta').textContent     = (p.authors || []).join(', ') + ' · ' + p.year;
  document.getElementById('det-abstract').textContent = p.abstract || 'No abstract available.';
  document.getElementById('det-link').href            = p.url || '#';
  detailEl.classList.add('open');
  updatePoints();
}

document.getElementById('close-btn').addEventListener('click', () => {
  state.selected = null;
  detailEl.classList.remove('open');
  updatePoints();
});

// ── THREE.JS BOOTSTRAP ────────────────────────────────────────────────────────
(function bootstrap() {
  const THREE = window.THREE;
  const wrap   = document.getElementById('canvas-wrap');
  const mainEl = document.getElementById('main');
  let W = mainEl.clientWidth, H = mainEl.clientHeight;

  const scene  = new THREE.Scene();
  scene.background = new THREE.Color('#ffffff');

  const camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 1000);
  camera.position.set(2, 1.5, 17);

  const renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
  renderer.setSize(W, H);
  wrap.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping   = true;
  controls.dampingFactor   = 0.08;
  controls.rotateSpeed     = 0.8;
  controls.autoRotate      = true;
  controls.autoRotateSpeed = 0.5;

  // ── Geometry ──────────────────────────────────────────────────────────────
  const n   = papers.length;
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  const sz  = new Float32Array(n);
  const al  = new Float32Array(n);

  papers.forEach((p, i) => {
    pos[i * 3]     = p.x || 0;
    pos[i * 3 + 1] = p.y || 0;
    pos[i * 3 + 2] = p.z || 0;
  });

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.setAttribute('aColor',   new THREE.BufferAttribute(col, 3));
  geom.setAttribute('aSize',    new THREE.BufferAttribute(sz,  1));
  geom.setAttribute('aAlpha',   new THREE.BufferAttribute(al,  1));

  const mat = new THREE.ShaderMaterial({
    uniforms: {uScale: {value: 108 * renderer.getPixelRatio()}},
    vertexShader: `
      attribute float aSize;
      attribute float aAlpha;
      attribute vec3  aColor;
      varying vec3  vColor;
      varying float vAlpha;
      uniform float uScale;
      void main() {
        vColor = aColor; vAlpha = aAlpha;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = aSize * (uScale / -mv.z);
        gl_Position  = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3  vColor;
      varying float vAlpha;
      void main() {
        vec2  c = gl_PointCoord - 0.5;
        float d = length(c);
        float a = smoothstep(0.5, 0.4, d);
        if (a <= 0.0) discard;
        gl_FragColor = vec4(vColor, vAlpha * a);
      }`,
    transparent: true,
    depthTest:   true,
    depthWrite:  false
  });

  const pts = new THREE.Points(geom, mat);
  scene.add(pts);

  // ── Cluster labels ────────────────────────────────────────────────────────
  const labelEls = {};
  clusters.forEach(c => {
    const d = document.createElement('div');
    d.style.cssText = [
      'position:absolute',
      'transform:translate(-50%,-50%)',
      'pointer-events:none',
      'white-space:nowrap',
      'background:rgba(255,255,255,.88)',
      'border:1px solid rgba(0,0,0,.07)',
      'border-radius:11px',
      'padding:3px 9px 3px 7px',
      "font-family:'IBM Plex Mono',monospace",
      'font-size:11.5px',
      'color:#2a2a26',
      'display:none',
      'align-items:center',
      'gap:7px',
      'z-index:3',
      'transition:opacity .15s'
    ].join(';');
    d.innerHTML = `<span style="width:7px;height:7px;border-radius:50%;background:${c.color};flex:none;display:inline-block"></span>${c.name}`;
    wrap.appendChild(d);
    labelEls[c.id] = d;
  });

  // ── Raycasting ────────────────────────────────────────────────────────────
  const ray = new THREE.Raycaster();
  ray.params.Points.threshold = 0.28;

  function ndc(e) {
    const r = renderer.domElement.getBoundingClientRect();
    return {
      x:  ((e.clientX - r.left) / r.width)  * 2 - 1,
      y: -((e.clientY - r.top)  / r.height) * 2 + 1,
      sx: e.clientX - r.left,
      sy: e.clientY - r.top
    };
  }

  function pickPaper(e) {
    const m = ndc(e);
    ray.setFromCamera({x: m.x, y: m.y}, camera);
    const hits = ray.intersectObject(pts);
    for (const h of hits) {
      const p = papers[h.index];
      if (active(p)) return {paper: p, sx: m.sx, sy: m.sy};
    }
    return {paper: null};
  }

  let downXY = null;
  renderer.domElement.addEventListener('pointermove', e => {
    const r = pickPaper(e);
    mainEl.style.cursor = r.paper ? 'pointer' : 'grab';
    if (r.paper) { showTooltip(r.paper, r.sx, r.sy); state.hoverId = r.paper._id; }
    else         { hideTooltip(); state.hoverId = null; }
    updatePoints();
  });
  renderer.domElement.addEventListener('pointerdown', e => {
    downXY = {x: e.clientX, y: e.clientY};
  });
  renderer.domElement.addEventListener('pointerup', e => {
    if (!downXY) return;
    const moved = Math.hypot(e.clientX - downXY.x, e.clientY - downXY.y);
    downXY = null;
    if (moved > 5) return;
    const r = pickPaper(e);
    if (r.paper) showDetail(r.paper);
  });
  renderer.domElement.addEventListener('pointerleave', () => {
    hideTooltip(); state.hoverId = null; updatePoints();
  });

  // ── Zoom buttons ──────────────────────────────────────────────────────────
  function zoomBy(f) {
    const dir = camera.position.clone().sub(controls.target).multiplyScalar(1 / f);
    camera.position.copy(controls.target.clone().add(dir));
    controls.update();
  }
  document.getElementById('btn-in').onclick  = () => zoomBy(1.3);
  document.getElementById('btn-out').onclick = () => zoomBy(1 / 1.3);
  document.getElementById('btn-fit').onclick = () => {
      controls.target.set(0, 0, 0);
      camera.position.set(2, 1.5, 17);
      controls.update();
    };
    function setRotation(on) {
      controls.autoRotate = on;
      document.getElementById('rot-on').style.background  = on ? '#1c1c1c' : '#fff';
      document.getElementById('rot-on').style.color       = on ? '#fff'    : '#9a9a93';
      document.getElementById('rot-off').style.background = on ? '#fff'    : '#1c1c1c';
      document.getElementById('rot-off').style.color      = on ? '#9a9a93' : '#fff';
    }
    document.getElementById('rot-on').onclick  = () => setRotation(true);
    document.getElementById('rot-off').onclick = () => setRotation(false);

  // ── Resize ────────────────────────────────────────────────────────────────
  new ResizeObserver(() => {
    W = mainEl.clientWidth; H = mainEl.clientHeight;
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    renderer.setSize(W, H);
    mat.uniforms.uScale.value = 108 * renderer.getPixelRatio();
  }).observe(mainEl);

  // ── Sidebar wiring ────────────────────────────────────────────────────────
  document.getElementById('search').addEventListener('input', e => {
    state.q = e.target.value; updatePoints();
  });
  yrMinEl.addEventListener('input', e => {
    state.yMin = Math.min(+e.target.value, state.yMax);
    yrMinEl.value = state.yMin; updateYearDisplay(); updatePoints();
  });
  yrMaxEl.addEventListener('input', e => {
    state.yMax = Math.max(+e.target.value, state.yMin);
    yrMaxEl.value = state.yMax; updateYearDisplay(); updatePoints();
  });
  authorSel.addEventListener('change', e => {
    state.author = e.target.value; updatePoints();
  });
  document.getElementById('reset-btn').addEventListener('click', () => {
    state.q = ''; state.author = ''; state.yMin = yearMin; state.yMax = yearMax;
    document.getElementById('search').value = '';
    authorSel.value = '';
    yrMinEl.value = yearMin; yrMaxEl.value = yearMax; updateYearDisplay();
    clusters.forEach(c => { state.clusters[c.id] = true; });
    document.querySelectorAll('.cluster-btn').forEach(b => b.style.opacity = 1);
    updatePoints();
  });
  document.querySelectorAll('.cluster-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = +btn.dataset.id;
      state.clusters[id] = !state.clusters[id];
      btn.style.opacity = state.clusters[id] ? 1 : 0.42;
      updatePoints();
    });
  });

  // ── Update point colours / sizes ──────────────────────────────────────────
  const noiseRgb = hexRgb(NOISE_COLOR);
  // Which clusters currently have at least one paper matching the active
  // filters (author/search/year/topic-toggle) — used to fade out cluster
  // labels for topics the selected author (etc.) has nothing in, mirroring
  // how the points themselves already dim when filtered out.
  let activeClusterIds = new Set();

  function updatePoints() {
    updateCounts();
    const base = 1.0;
    activeClusterIds = new Set();
    for (let i = 0; i < n; i++) {
      const p       = papers[i];
      const act     = active(p);
      const sel     = state.selected && state.selected._id === p._id;
      const hov     = state.hoverId === p._id;
      const isNoise = p.cluster === -1;

      if (act && !isNoise) activeClusterIds.add(p.cluster);

      let r, g, b, a, s;

      if (!act) {
        // inactive / filtered out — nearly invisible
        r = 0.62; g = 0.62; b = 0.59; a = 0.09; s = base * 0.7;
      } else if (isNoise) {
        // active noise point — visible grey, slightly smaller than clustered
        r = noiseRgb[0]; g = noiseRgb[1]; b = noiseRgb[2]; a = 0.55; s = base * 0.85;
      } else {
        // active clustered point
        const c   = clusterMap[p.cluster];
        const rgb = c ? hexRgb(c.color) : noiseRgb;
        r = rgb[0]; g = rgb[1]; b = rgb[2]; a = 0.95; s = base;
      }

      // hover / selected — slightly smaller pop for noise to stay visually distinct
      if ((sel || hov) && act) {
        a = 1.0;
        s = base * (isNoise ? 1.5 : 1.9);
      }

      col[i*3] = r; col[i*3+1] = g; col[i*3+2] = b;
      al[i] = a; sz[i] = s;
    }
    geom.attributes.aColor.needsUpdate = true;
    geom.attributes.aAlpha.needsUpdate = true;
    geom.attributes.aSize.needsUpdate  = true;
  }

  updatePoints();

  // ── Animation loop ────────────────────────────────────────────────────────
  (function loop() {
    requestAnimationFrame(loop);
    controls.update();

    clusters.forEach(c => {
      const el = labelEls[c.id];
      if (!el) return;
      if (!state.clusters[c.id]) { el.style.display = 'none'; return; }
      const v = new THREE.Vector3(c.cx, c.cy, c.cz).project(camera);
      if (v.z > 1) { el.style.display = 'none'; return; }
      el.style.display = 'flex';
      el.style.left = ((v.x * 0.5 + 0.5) * W) + 'px';
      el.style.top  = ((-v.y * 0.5 + 0.5) * H) + 'px';
      el.style.opacity = activeClusterIds.has(c.id) ? '1' : '0.15';
    });

    renderer.render(scene, camera);
  })();
})();
</script>
</body>
</html>
"""

# ── LAB MEMBERS + VOCABULARY FROM THE WEBSITE ─────────────────────────────────
WEBSITE = {"people": [], "vocab": [], "source": None}
ID_TO_NAME = {}      # OpenAlex author ID (upper) -> name as written on the website

def _front_matter(text):
    """YAML front matter of a .md file as a dict (empty dict if none)."""
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.S)
    if not m:
        return {}
    try:
        import yaml
        return yaml.safe_load(m.group(1)) or {}
    except Exception:
        out = {}
        for line in m.group(1).splitlines():
            k, sep, v = line.partition(":")
            if sep and not line.startswith((" ", "-")):
                out[k.strip()] = v.strip().strip("'\"")
        return out

WANTED = ("content/people/", "data/projects.json", "content/pages/home.md")
WEBSITE_CACHE = Path(__file__).resolve().parent / "website_cache.json"

def _wanted(rel):
    return (rel.startswith("content/people/") and rel.endswith(".md")) or rel in WANTED[1:]

def _github_zip():
    """The whole website repo in ONE request (GitHub allows only 60 anonymous
    API requests per hour, so fetching file by file runs out quickly)."""
    import io, zipfile
    url = f"https://api.github.com/repos/{WEBSITE_GITHUB}/zipball/{WEBSITE_BRANCH}"
    req = urllib.request.Request(url, headers={"User-Agent": "research-atlas-build"})
    if os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    files = {}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            rel = name.split("/", 1)[1] if "/" in name else name   # drop "owner-repo-sha/"
            if _wanted(rel):
                files[rel] = z.read(name).decode("utf-8")
    return files

def _read_website_files():
    """{relative path: text} for the files we need: from a local copy of the
    website repo, else from GitHub (one download), else from the copy saved
    by the last successful run (website_cache.json)."""
    files = {}
    if WEBSITE_DIR and (Path(WEBSITE_DIR) / "content" / "people").is_dir():
        root = Path(WEBSITE_DIR)
        for f in sorted((root / "content" / "people").glob("*.md")):
            files[f"content/people/{f.name}"] = f.read_text(encoding="utf-8")
        for rel in WANTED[1:]:
            if (root / rel).exists():
                files[rel] = (root / rel).read_text(encoding="utf-8")
        WEBSITE["source"] = str(root)
    elif WEBSITE_GITHUB:
        try:
            files = _github_zip()
            WEBSITE["source"] = f"github.com/{WEBSITE_GITHUB}"
        except Exception as e:
            print(f"  Could not download the website from GitHub ({e}).")
    if files:
        WEBSITE_CACHE.write_text(json.dumps(files, ensure_ascii=False), encoding="utf-8")
        return files
    if WEBSITE_CACHE.exists():
        print(f"  Using the copy saved by the last successful run ({WEBSITE_CACHE.name}).")
        WEBSITE["source"] = WEBSITE_CACHE.name
        return json.loads(WEBSITE_CACHE.read_text(encoding="utf-8"))
    return {}

def load_website():
    """Fill WEBSITE with lab members and the lab's own vocabulary."""
    try:
        files = _read_website_files()
    except Exception as e:
        print(f"  Could not read the website ({e}).")
        return
    if not files:
        print("  Website not found (check WEBSITE_DIR / WEBSITE_GITHUB).")
        return

    vocab = []   # (term, source)
    for path, text in files.items():
        if path.startswith("content/people/"):
            fm = _front_matter(text)
            if not fm.get("name"):
                continue
            orcid = re.search(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", str(fm.get("orcid") or ""), re.I)
            ids = [x.upper() for x in re.findall(r"A\d{4,}", str(fm.get("openalex_id") or ""), re.I)]
            WEBSITE["people"].append({"name": clean_name(str(fm["name"])),
                                      "orcid": orcid.group(0).upper() if orcid else "",
                                      "ids": ids})
            for part in re.split(r"[,;/]|\band\b", str(fm.get("focus") or "")):
                vocab.append((part.strip(), "people focus"))
    if "data/projects.json" in files:
        for proj in json.loads(files["data/projects.json"]).get("projects", []):
            for k in proj.get("keywords", []):
                vocab.append((k, "FRIS keywords"))
    if "content/pages/home.md" in files:
        for t in _front_matter(files["content/pages/home.md"]).get("themes") or []:
            if isinstance(t, dict) and t.get("title") and "placeholder" not in t["title"].lower():
                vocab.append((t["title"], "homepage themes"))
    for t in LAB_VOCAB_EXTRA:
        vocab.append((t, "LAB_VOCAB_EXTRA"))
    WEBSITE["vocab"] = [(t, src) for t, src in vocab if t and 2 < len(t) < 60]
    n_ids = sum(1 for p in WEBSITE["people"] if p["orcid"] or p["ids"])
    print(f"  Website ({WEBSITE['source']}): {len(WEBSITE['people'])} people, {n_ids} with ORCID/OpenAlex ID, "
          f"{len(WEBSITE['vocab'])} vocabulary terms")

def openalex_get(path, params):
    params = dict(params)
    params["mailto"] = MAILTO
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    url = f"https://api.openalex.org/{path}?" + urllib.parse.urlencode(params, safe="|,:*")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 4:
                time.sleep(2 * (attempt + 1)); continue
            raise RuntimeError(f"OpenAlex {e.code} for {url.split('api_key=')[0]}: {e.read()[:300]!r}") from None

def resolve_author_ids():
    """AUTHOR_IDS from the website (ORCID -> OpenAlex ID) plus EXTRA_AUTHOR_IDS."""
    global AUTHOR_IDS
    people = WEBSITE["people"]
    ids = set()
    for p in people:
        for aid in p["ids"]:
            ids.add(aid); ID_TO_NAME[aid] = p["name"]
    orcids = {p["orcid"]: p["name"] for p in people if p["orcid"]}
    orcid_list = list(orcids)
    for i in range(0, len(orcid_list), 50):
        batch = orcid_list[i:i + 50]
        data = openalex_get("authors", {"filter": "orcid:" + "|".join(batch),
                                        "select": "id,orcid,display_name", "per-page": "200"})
        found = set()
        for a in data.get("results", []):
            aid = a["id"].split("/")[-1].upper()
            orc = (a.get("orcid") or "").split("/")[-1].upper()
            name = orcids.get(orc)
            if name:
                ids.add(aid); ID_TO_NAME[aid] = name; found.add(orc)
        for orc in set(batch) - found:
            print(f"  ORCID {orc} ({orcids[orc]}): no OpenAlex author found — add their OpenAlex ID on the website")
        time.sleep(0.2)
    for aid in EXTRA_AUTHOR_IDS:
        ids.add(aid.upper())
    missing = [p["name"] for p in people if not p["orcid"] and not p["ids"]]
    if missing:
        print(f"  No ORCID/OpenAlex ID on the website for: {', '.join(missing)}")
    AUTHOR_IDS = sorted(ids)
    print(f"  {len(AUTHOR_IDS)} OpenAlex author IDs for {len(set(ID_TO_NAME.values()))} lab members"
          + (f" (+{len(EXTRA_AUTHOR_IDS)} from EXTRA_AUTHOR_IDS)" if EXTRA_AUTHOR_IDS else ""))

# ── FETCH ─────────────────────────────────────────────────────────────────────
def openalex_filter():
    # No has_abstract filter: papers without an abstract are fetched too, and
    # placed in a cluster after clustering (assign_no_abstract).
    parts = []
    if AUTHOR_IDS:
        parts.append("authorships.author.id:" + "|".join(a.upper() for a in AUTHOR_IDS))
    if INSTITUTION_ID:
        parts.append("authorships.institutions.id:" + INSTITUTION_ID)
    return ",".join(parts)

def fetch_openalex():
    """All works by the lab members. Uses cursor paging with an API key,
    page-number paging without one (OpenAlex refuses cursors without a key).
    Large author lists are split into chunks of 50 IDs (OpenAlex allows 100)."""
    rows, seen = [], set()
    ids = [a.upper() for a in AUTHOR_IDS]
    for i in range(0, max(len(ids), 1), 50):
        chunk = ids[i:i + 50]
        parts = ["authorships.author.id:" + "|".join(chunk)] if chunk else []
        if INSTITUTION_ID:
            parts.append("authorships.institutions.id:" + INSTITUTION_ID)
        filt = ",".join(parts)
        cursor, page = "*", 1
        while True:
            params = {"filter": filt, "per-page": "200"}
            if OPENALEX_API_KEY:
                params["cursor"] = cursor
            else:
                params["page"] = str(page)
            data = openalex_get("works", params)
            batch = data.get("results", [])
            for w in batch:
                if w["id"] not in seen:
                    seen.add(w["id"]); rows.append(w)
            print(f"  fetched {len(rows)} works…")
            time.sleep(0.2)
            if OPENALEX_API_KEY:
                cursor = data["meta"].get("next_cursor")
                if not cursor or not batch:
                    break
            else:
                page += 1
                if len(batch) < 200 or page > 50:
                    break
    if not OPENALEX_API_KEY:
        print("  (no OPENALEX_API_KEY set — used page-number paging)")
    return rows

def abstract_from_inverted(inv):
    if not inv: return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))

# Meeting-abstract codes at the start of a title, e.g. "P.2.c.007 ", "S.05.02 ",
# "T128. ", "445. ", "P2-176 ", "P.0381 ", "Poster #M51 ".
# Letter codes must contain a "." or "-" (so gene names like "CR1" don't match);
# bare numbers need a trailing dot ("445.") or 3+ digits ("157").
CONF_CODE = re.compile(
    r"^(?:[Pp]oster\s*#?\s*\w+"
    r"|[A-Z]{1,2}(?=[\w.\-]*[.\-])[\d.\-]*\d[a-z.\d\-]*\.?"
    r"|\d+\."
    r"|\d{3,})\s"
)

def is_conference(w, has_abstract):
    """True for conference proceedings / meeting abstracts / posters."""
    src = ((w.get("primary_location") or {}).get("source") or {})
    if src.get("type") == "conference":
        return True
    if w.get("type_crossref") == "proceedings-article":
        return True
    # The title-code check only applies to works without an abstract: that's
    # where supplement-printed meeting abstracts end up, and it avoids false
    # hits on real papers whose title starts with a number ("12 weeks of …").
    return not has_abstract and bool(CONF_CODE.match(w.get("title") or ""))

_TITLES = {"prof", "professor", "dr", "drs", "ir", "md", "phd", "msc", "bsc", "ma", "mba",
           "jr", "sr", "mr", "mrs", "ms", "em", "emeritus"}
_PARTICLES = {"van", "von", "den", "der", "de", "del", "della", "di", "da", "du", "le", "la",
              "ter", "ten", "op", "t", "vande", "vanden", "vander"}

def _name_parts(name):
    """'Prof. dr. François‐Laurent De Winter (MD)' -> ['francoislaurent', 'de', 'winter'];
    'Bouckaert, Filip' -> ['filip', 'bouckaert']."""
    import unicodedata
    name = re.sub(r"\(.*?\)", " ", str(name or ""))
    if "," in name:                       # "Surname, First" -> "First Surname"
        last, _, first = name.partition(",")
        name = f"{first} {last}"
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    toks = [re.sub(r"[^a-z]", "", w) for w in name.lower().split()]
    return [t for t in toks if t and t not in _TITLES]

def _close(a, b):
    """Equal, or one letter different for longer names ('Boucaert' ~ 'Bouckaert')."""
    if a == b:
        return True
    if min(len(a), len(b)) < 5 or abs(len(a) - len(b)) > 1:
        return False
    prev = list(range(len(b) + 1))           # Levenshtein distance <= 1
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] <= 1

def same_person(paper_name, site_name):
    """True unless the name on the paper clearly belongs to someone else:
    the website surname must appear in the paper's name (allowing one typo),
    and the first given name must match (or its initial, when either side
    only gives initials). Names that can't be compared count as a match."""
    p, s = _name_parts(paper_name), _name_parts(site_name)
    if not p or not s:
        return True
    surname = s[-1]
    hits = [i for i, t in enumerate(p) if _close(t, surname)]
    if not hits:
        return False
    given = [t for j, t in enumerate(p) if j != hits[-1] and t not in _PARTICLES]
    site_given = [t for t in s[:-1] if t not in _PARTICLES]
    if not given or not site_given:          # only a surname on one side
        return True
    a, b = given[0], site_given[0]
    if len(a) <= 2 or len(b) <= 2:           # initials ("F", "FL")
        return a[0] == b[0]
    return a.startswith(b) or b.startswith(a) or _close(a, b)

REJECTED_NAMES = defaultdict(Counter)   # lab member ID -> names on papers judged "someone else"

def leuven_status(w, pi_ids_upper):
    """'leuven' if a lab member on this work has a KU/UZ Leuven affiliation,
    'other' if lab members have only other affiliations, 'none' if no
    affiliation is recorded for any lab member on this work, 'wrong person'
    if every lab-member name on the work belongs to someone else."""
    seen_any = matched_any = mismatched = False
    for a in w.get("authorships", []):
        aid = (a["author"].get("id") or "").split("/")[-1].upper()
        if aid not in pi_ids_upper:
            continue
        if CHECK_NAMES:
            on_paper = a.get("raw_author_name") or a["author"].get("display_name") or ""
            if not same_person(on_paper, ID_TO_NAME.get(aid)):
                mismatched = True
                REJECTED_NAMES[aid][clean_name(on_paper)] += 1
                continue
        matched_any = True
        texts = [i.get("display_name") or "" for i in a.get("institutions") or []]
        texts += a.get("raw_affiliation_strings") or []
        texts += [x.get("raw_affiliation_string") or "" for x in a.get("affiliations") or []]
        texts = [t for t in texts if t]
        if texts:
            seen_any = True
        if any(LEUVEN_PATTERN.search(t) for t in texts):
            return "leuven"
    if mismatched and not matched_any:
        return "wrong person"
    return "other" if seen_any else "none"

def exclusion_reason(w, has_abstract, pi_ids_upper=frozenset()):
    if w.get("type") in EXCLUDE_TYPES:
        return f"type {w.get('type')}"
    if w.get("is_paratext"):
        return "paratext"
    if EXCLUDE_CONFERENCE and is_conference(w, has_abstract):
        return "conference"
    if REQUIRE_LEUVEN:
        status = leuven_status(w, pi_ids_upper)
        if status == "wrong person":
            return "wrong person: name on the paper doesn't match the website"
        if status == "other":
            return "not Leuven: lab member listed at another institution"
        if status == "none" and not KEEP_NO_AFFILIATION:
            return "not Leuven: no affiliation recorded"
    return None

def clean_name(name):
    """Unify look-alike characters in names (various hyphens/dashes, odd spaces)."""
    name = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", name)
    return " ".join(name.replace("\u00a0", " ").split())

def normalize(works):
    """Parse works and collect PI display names as a side-effect.
    Returns (papers_with_abstract, papers_without_abstract)."""
    pi_ids_upper = {a.upper() for a in AUTHOR_IDS}
    out, no_abs = [], []
    dropped = defaultdict(list)

    # One name per lab member: OpenAlex stores slightly different spellings on
    # different papers ("Chih-Hao Lien" / "Chih‐Hao Lien" with another hyphen
    # character), which would otherwise show up as separate people.
    spellings = defaultdict(lambda: defaultdict(int))
    for w in works:
        for a in w.get("authorships", []):
            aid = (a["author"].get("id") or "").split("/")[-1].upper()
            if aid in pi_ids_upper and a["author"].get("display_name"):
                spellings[aid][clean_name(a["author"]["display_name"])] += 1
    canonical = {aid: max(c, key=c.get) for aid, c in spellings.items()}
    # Names as written on the website win over OpenAlex's spellings, and the
    # same person with two OpenAlex profiles appears once in the dropdown.
    canonical.update({aid: name for aid, name in ID_TO_NAME.items() if aid in pi_ids_upper})
    kept_per, dropped_per, wrong_per = defaultdict(int), defaultdict(int), defaultdict(int)

    for w in works:
        ab = abstract_from_inverted(w.get("abstract_inverted_index"))
        reason = exclusion_reason(w, len(ab) >= 60, pi_ids_upper)
        lab_ids = {(a["author"].get("id") or "").split("/")[-1].upper()
                   for a in w.get("authorships", [])} & pi_ids_upper
        for aid in lab_ids:
            if reason is None:
                kept_per[aid] += 1
            elif reason.startswith("not Leuven"):
                dropped_per[aid] += 1
            elif reason.startswith("wrong person"):
                wrong_per[aid] += 1
        if reason:
            line = f"{w['id'].split('/')[-1]}  ({w.get('publication_year')})  {w.get('title') or '(untitled)'}"
            if reason.startswith(("not Leuven", "wrong person")):
                who = [(a.get("raw_author_name") or a["author"].get("display_name") or "?",
                        "; ".join(i.get("display_name") or "" for i in a.get("institutions") or []) or "—")
                       for a in w.get("authorships", [])
                       if (a["author"].get("id") or "").split("/")[-1].upper() in pi_ids_upper]
                line += "   [" + " | ".join(f"{n}: {inst}" for n, inst in who) + "]"
            dropped[reason].append(line)
            continue
        # Keep the first MAX_AUTHORS authors, the last (senior) author, and any
        # PI in between. "…" marks where authors were left out.
        authorships = w.get("authorships", [])
        last = len(authorships) - 1
        authors, prev = [], -1
        for pos, a in enumerate(authorships):
            aid  = (a["author"].get("id") or "").split("/")[-1].upper()
            is_pi = aid in pi_ids_upper
            name = canonical.get(aid) if is_pi else clean_name(a["author"].get("display_name") or "?")
            if is_pi:
                PI_AUTHOR_NAMES.add(name)
            if pos < MAX_AUTHORS or pos == last or is_pi:
                if pos > prev + 1:
                    authors.append("…")
                authors.append(name)
                prev = pos

        if not out and not no_abs:  # print first paper's raw keywords to inspect
            print("keywords sample:", w.get("keywords", [])[:3])
            print("topics sample:",   w.get("topics",   [])[:3])

        rec = {
            "id":       w["id"].split("/")[-1],
            "title":    w.get("title") or "(untitled)",
            "year":     w.get("publication_year"),
            "authors":  authors,
            "abstract": ab if len(ab) >= 60 else "",
            "url":      w.get("doi") or w["id"],
            "pmid":     ((w.get("ids") or {}).get("pmid") or "").rstrip("/").split("/")[-1],
            "type":     w.get("type") or "",
            "venue":    (((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""),
            "keywords": [k.get("keyword") or k.get("display_name") or "" for k in w.get("keywords", []) if k],
            "mesh":     [m["descriptor_name"] for m in w.get("mesh", [])
                         if m.get("descriptor_name") and m.get("is_major_topic")],
            "topics":   [t["display_name"] for t in w.get("topics", [])[:3] if t.get("display_name")],
        }
        (out if rec["abstract"] else no_abs).append(rec)

    print_dropped(dropped)
    if REQUIRE_LEUVEN:
        print("\nLeuven check per lab member (papers kept / dropped as not Leuven / dropped as someone else):")
        for aid in [a.upper() for a in AUTHOR_IDS]:
            name = canonical.get(aid, "(no papers found)")
            print(f"  {aid:13s} {name:32s} kept {kept_per[aid]:4d}   not Leuven {dropped_per[aid]:4d}   other person {wrong_per[aid]:4d}")
            if wrong_per[aid]:
                seen = ", ".join(f"'{n}' ({c}x)" for n, c in REJECTED_NAMES[aid].most_common(3))
                print(f"  {'':13s}   names on papers judged someone else: {seen}")
                if wrong_per[aid] > kept_per[aid]:
                    print(f"  {'':13s}   <- mostly someone else: if these are really {name}, fix the name on the "
                          f"website; otherwise this OpenAlex ID belongs to another person")
    return out, no_abs

def print_dropped(dropped):
    if not dropped:
        return
    print(f"\nLeft out {sum(map(len, dropped.values()))} works:")
    for reason, items in sorted(dropped.items()):
        print(f"  {reason}: {len(items)}")
        if DEBUG_LABELS:
            for t in items:
                print(f"      {t[:200]}")

def keep_no_abstract(no_abs):
    """After PubMed: keep only articles/letters without an abstract."""
    keep, dropped = [], defaultdict(list)
    for p in no_abs:
        if p["type"] in NO_ABSTRACT_KEEP_TYPES:
            keep.append(p)
        else:
            dropped[f"no abstract, type {p['type'] or 'unknown'}"].append(f"{p['id']}  {p['title']}")
    print_dropped(dropped)
    return keep

# ── PUBMED FALLBACK FOR MISSING ABSTRACTS ─────────────────────────────────────
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def parse_pubmed_xml(xml_bytes):
    """PubMed efetch XML -> {pmid: abstract}. Structured abstracts keep their labels."""
    import xml.etree.ElementTree as ET
    out = {}
    for art in ET.fromstring(xml_bytes).iter("PubmedArticle"):
        pmid = (art.findtext("./MedlineCitation/PMID") or "").strip()
        parts = []
        for at in art.iter("AbstractText"):
            text = " ".join("".join(at.itertext()).split())
            if text:
                label = at.get("Label")
                parts.append(f"{label}: {text}" if label else text)
        if pmid and parts:
            out[pmid] = " ".join(parts)
    return out

def fill_from_pubmed(no_abs):
    """Look up abstracts on PubMed for papers OpenAlex has none for.
    Returns (papers_now_with_abstract, papers_still_without)."""
    cache = json.loads(PUBMED_CACHE.read_text(encoding="utf-8")) if PUBMED_CACHE.exists() else {}
    todo  = [p for p in no_abs if p["id"] not in cache]

    # 1) papers with only a DOI: find their PMID (one request each, NCBI allows ~3/s)
    doi_only = [p for p in todo if not p.get("pmid") and p["url"].startswith("https://doi.org/")]
    if doi_only:
        print(f"\nLooking up {len(doi_only)} DOIs on PubMed…")
    for n, p in enumerate(doi_only, 1):
        doi = p["url"][len("https://doi.org/"):]
        qs  = urllib.parse.urlencode({"db": "pubmed", "retmode": "json", "term": f"{doi}[doi]", "email": MAILTO})
        try:
            with urllib.request.urlopen(f"{EUTILS}/esearch.fcgi?{qs}", timeout=30) as r:
                ids = json.load(r)["esearchresult"]["idlist"]
        except Exception as e:
            print(f"  {p['id']}: lookup failed ({e}), will retry next run")
            ids = None
        if ids is not None and len(ids) == 1:
            p["pmid"] = ids[0]
        elif ids is not None:
            cache[p["id"]] = ""          # not on PubMed (or ambiguous): remember
        if n % 50 == 0:
            print(f"  {n}/{len(doi_only)}")
        time.sleep(0.35)

    # 2) fetch abstracts for all known PMIDs, 200 per request
    by_pmid = {p["pmid"]: p for p in todo if p.get("pmid")}
    pmids   = list(by_pmid)
    if pmids:
        print(f"Fetching {len(pmids)} abstracts from PubMed…")
    for i in range(0, len(pmids), 200):
        batch = pmids[i:i + 200]
        data  = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(batch),
                                        "retmode": "xml", "email": MAILTO}).encode()
        try:
            with urllib.request.urlopen(f"{EUTILS}/efetch.fcgi", data=data, timeout=60) as r:
                found = parse_pubmed_xml(r.read())
        except Exception as e:
            print(f"  batch {i // 200 + 1}: fetch failed ({e}), will retry next run")
            continue
        for pm in batch:
            cache[by_pmid[pm]["id"]] = found.get(pm, "")
        time.sleep(0.35)

    # papers with no PMID and no DOI can't be looked up: remember that too
    for p in todo:
        if not p.get("pmid") and not p["url"].startswith("https://doi.org/"):
            cache.setdefault(p["id"], "")

    PUBMED_CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")

    now_with, still_without = [], []
    for p in no_abs:
        ab = cache.get(p["id"], "")
        if len(ab) >= 60:
            p["abstract"] = ab
            now_with.append(p)
        else:
            still_without.append(p)
    return now_with, still_without

# ── EMBED / CLUSTER ───────────────────────────────────────────────────────────
def build(papers):
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]

    print("Embedding…")
    model = SentenceTransformer(EMBED_MODEL)
    vecs  = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    if USE_THEMES:
        return build_themes(papers, texts, model, vecs)

    print("Reducing to 10D for clustering…")
    vecs_cluster = umap.UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE
    ).fit_transform(vecs)

    print(f"Reducing to {DIMS}D for visualisation…")
    coords = umap.UMAP(
        n_components=DIMS,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_STATE
    ).fit_transform(vecs)

    print(f"Clustering with HDBSCAN (min_cluster_size={MIN_CLUSTER_SIZE})…")
    try:
        from hdbscan import HDBSCAN          # the hdbscan package, if installed
    except ImportError:
        from sklearn.cluster import HDBSCAN  # same algorithm, built into scikit-learn
    labels = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,  # lower = more, smaller clusters
        min_samples=2,                      # reduces noise points
        cluster_selection_method="leaf"     # finer-grained clusters
    ).fit_predict(vecs_cluster)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"  {n_clusters} clusters, {n_noise} noise points")
    if n_clusters > len(CLUSTER_COLORS):
        print(f"  NOTE: {n_clusters} clusters but only {len(CLUSTER_COLORS)} colors defined — "
              f"colors will repeat (cluster id % {len(CLUSTER_COLORS)}). Add more to CLUSTER_COLORS if you want every cluster visually distinct.")

    # Normalise coords to roughly [-5, 5]
    coords -= coords.mean(axis=0)
    coords /= (np.abs(coords).max() + 1e-9) / 5.0

    for i, p in enumerate(papers):
        p["cluster"] = int(labels[i])
        p["_text"]   = texts[i]
        p["_weight"] = 1.0
        for a, v in zip(["x", "y", "z"][:DIMS], coords[i]):
            p[a] = round(float(v), 4)

    return papers, model, vecs, labels, coords

def theme_text(t):
    return f"{t['name']}. {t['description']} Keywords: {', '.join(t.get('keywords', []))}."

def theme_scores(model, vecs):
    """Similarity of each paper to each theme, smoothed over its neighbours."""
    tvecs = model.encode([theme_text(t) for t in THEMES], normalize_embeddings=True, show_progress_bar=False)
    raw = vecs @ tvecs.T
    if THEME_SMOOTH_K > 1:
        k  = min(THEME_SMOOTH_K, len(vecs))
        nn = np.argsort(-(vecs @ vecs.T), axis=1)[:, :k]     # includes the paper itself
        return raw[nn].mean(axis=1), raw, tvecs
    return raw, raw, tvecs

def build_themes(papers, texts, model, vecs):
    """Theme mode: assign papers to THEMES, lay them out with UMAP, then pull
    each theme together by THEME_PULL."""
    scores, raw, _ = theme_scores(model, vecs)
    labels = scores.argmax(axis=1)
    top2 = np.sort(scores, axis=1)[:, -2:]
    close = int(((top2[:, 1] - top2[:, 0]) < 0.1 * np.abs(top2[:, 1])).sum())

    print(f"Reducing to {DIMS}D for visualisation…")
    coords = umap.UMAP(
        n_components=DIMS,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_STATE
    ).fit_transform(vecs)
    if THEME_PULL > 0:
        print(f"  pulling papers {THEME_PULL:.0%} of the way towards their theme's centre")
        for i in range(len(THEMES)):
            members = labels == i
            if members.any():
                centre = np.median(coords[members], axis=0)
                coords[members] += THEME_PULL * (centre - coords[members])

    print("\nPapers per theme (papers with an abstract):")
    for i, t in enumerate(THEMES):
        print(f"  {t['name']:36s} {int((labels == i).sum()):4d}")
    print(f"  close calls (top two themes within 10%): {close}")

    coords -= coords.mean(axis=0)
    coords /= (np.abs(coords).max() + 1e-9) / 5.0
    for i, p in enumerate(papers):
        p["cluster"] = int(labels[i])
        p["_text"]   = texts[i]
        p["_weight"] = 1.0
        for a, v in zip(["x", "y", "z"][:DIMS], coords[i]):
            p[a] = round(float(v), 4)
    return papers, model, vecs, labels, coords

def meta_text(p):
    """Text for a paper without abstract: title + keywords + topics + MeSH + journal."""
    extra = list(dict.fromkeys(e for e in p.get("keywords", []) + p.get("topics", []) + p.get("mesh", []) if e))
    parts = [p["title"]]
    if extra:
        parts.append("; ".join(extra))
    if p.get("venue"):
        parts.append(p["venue"])
    return ". ".join(parts)

def assign_no_abstract(papers, no_abs, model, vecs, labels, coords):
    """Place each abstract-less paper in the cluster its nearest neighbours
    (among papers with an abstract) belong to, and position it among them."""
    if not no_abs:
        return []
    texts = [meta_text(p) for p in no_abs]
    nvecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    sims  = nvecs @ vecs.T
    k     = min(ASSIGN_K, len(papers))
    rng   = np.random.default_rng(RANDOM_STATE)
    spread = coords.std(axis=0) * 0.03   # small jitter so points don't stack

    per_cluster, unclustered = defaultdict(int), []
    for p, text, row in zip(no_abs, texts, sims):
        nn   = np.argsort(-row)[:k]
        top  = float(row[nn[0]])
        votes = defaultdict(float)
        for j in nn:
            votes[int(labels[j])] += float(row[j])
        cl = max(votes, key=votes.get)
        if top < MIN_ASSIGN_SIM and not USE_THEMES:   # in theme mode every paper gets a theme
            cl = -1
        # position: similarity-weighted mean of the neighbours in that cluster
        members = [j for j in nn if cl == -1 or int(labels[j]) == cl]
        w   = np.array([max(float(row[j]), 1e-6) for j in members])
        xyz = (coords[members] * w[:, None]).sum(axis=0) / w.sum() + rng.normal(0, 1, DIMS) * spread

        p["cluster"] = cl
        p["_text"]   = text
        p["_weight"] = NO_ABSTRACT_WEIGHT
        for a, v in zip(["x", "y", "z"][:DIMS], xyz):
            p[a] = round(float(v), 4)
        if cl == -1:
            unclustered.append((top, p))
        else:
            per_cluster[cl] += 1

    print(f"\nPlaced {len(no_abs)} papers without abstract: "
          f"{sum(per_cluster.values())} in clusters, {len(unclustered)} left unclustered.")
    if unclustered:
        print(f"  Unclustered (nearest similarity < {MIN_ASSIGN_SIM}):")
        for top, p in sorted(unclustered, key=lambda t: t[0]):
            print(f"    {p['id']}  ({top:.2f})  {p['title'][:90]}")
    return no_abs

def apply_names(papers, model=None, vecs=None):
    if USE_THEMES:
        names = {i: t["name"] for i, t in enumerate(THEMES)}
        names[-1] = "Unclustered"
        if DEBUG_LABELS:   # show what each theme ended up containing, as a check
            print("\nTypical terms per theme (for checking the theme descriptions):")
            combined_names(papers, names, header=None)
        for p in papers:
            p["clusterLabel"] = names.get(p["cluster"], "Unclustered")
            for k in ("_text", "_weight", "pmid", "type"):
                p.pop(k, None)
        return papers
    names = label_clusters([p["_text"] for p in papers],
                           np.array([p["cluster"] for p in papers]),
                           papers,
                           weights=[p["_weight"] for p in papers],
                           debug=DEBUG_LABELS and NAMING_METHOD == "mesh")
    if NAMING_METHOD == "combined":
        names = combined_names(papers, names)
    names = finalize_names(names)
    for p in papers:
        p["clusterLabel"] = names.get(p["cluster"], "Unclustered")
        for k in ("_text", "_weight", "pmid", "type"):
            p.pop(k, None)
    return papers

# ── NAMING: ONE RANKING OVER ALL TERMS ────────────────────────────────────────
_STOP = set("""a an the of in on for and or with without to from by at as is are was were be
been being this that these those its their our we using use used based via into among
between within across after before during versus vs than""".split())
_TOPIC_TAIL = re.compile(r"\s+(research(?: topics)?|studies|research and treatments?|and treatments?|"
                         r"techniques and applications|mechanisms)$", re.I)

def _key(text):
    """Comparable form of a term: lowercase, no punctuation, simple plurals
    ('Emotions' = 'emotion', "Alzheimer's" = 'Alzheimer'), and a trailing
    acronym of the words before it dropped ('electroconvulsive therapy ECT')."""
    t = str(text).lower().replace("’", "'").replace("'s ", " ")
    t = re.sub(r"'s$", "", t)
    words = re.sub(r"[^a-z0-9]+", " ", t).split()
    words = [w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "is", "us")) else w
             for w in words]
    if len(words) > 2 and words[-1] == "".join(w[0] for w in words[:-1]):
        words = words[:-1]
    return " ".join(words)

def _uninvert(term):
    parts = [x.strip() for x in str(term).split(",")]
    return f"{parts[1]} {parts[0]}" if len(parts) == 2 and all(parts) else str(term)

def combined_names(papers, base_names, header="Cluster names:"):
    """Name each cluster with up to two terms from one shared ranking.
    Clusters without a good term keep their MeSH/keyword/TF-IDF name."""
    lab = {_key(t) for t, _src in WEBSITE.get("vocab") or []} | {_key(t) for t in LAB_VOCAB_EXTRA}
    lab = {k for k in lab if len(k) > 3}
    shown = defaultdict(Counter)

    def usable(key):
        words = key.split()
        return (words and words[0] not in _STOP and words[-1] not in _STOP
                and not all(w in NAME_GENERIC_WORDS or w in _STOP for w in words))

    def terms_of(p):
        found = {}
        def add(raw, weight):
            key = _key(raw)
            if key and usable(key):
                shown[key][raw.strip()] += 1
                found[key] = max(found.get(key, 0.0), weight)
        for m in p.get("mesh", []):
            add(_uninvert(m), 1.0)
        for k in p.get("keywords", []):
            add(k, 0.8)
        for t in p.get("topics", []):
            add(_TOPIC_TAIL.sub("", t).strip(), 0.4)
        text = f" {_key(p.get('title', '') + ' ' + p.get('abstract', ''))} "
        for key in lab:
            if f" {key} " in text:
                add(key, 1.0)
        words = _key(p.get("title", "")).split()
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                add(" ".join(words[i:i + n]), 0.5)
        return found

    clusters = sorted({p["cluster"] for p in papers if p["cluster"] != -1})
    members = {c: [p for p in papers if p["cluster"] == c] for c in clusters}
    tags = {id(p): terms_of(p) for c in clusters for p in members[c]}

    def coverage(ps, key):
        return sum(tags[id(p)].get(key, 0.0) for p in ps) / max(len(ps), 1)

    def ranking(c, others):
        ps = members[c]
        counts = Counter(k for p in ps for k in tags[id(p)])
        out = []
        for key, n in counts.items():
            if n < NAME_MIN_PAPERS:
                continue
            here, there = coverage(ps, key), coverage(others, key)
            lift = (here + 0.01) / (there + 0.01)
            if lift < 1.5:
                continue
            score = here * math.log2(min(lift, 8))      # coverage, once clearly distinctive
            if key in lab:
                score *= NAME_LAB_BONUS
            if len(key.split()) > 1:
                score *= 1.15
            out.append((score, key, here, there))
        return sorted(out, reverse=True)

    def nice(key):
        variants = [v for v, _ in shown[key].most_common()]
        caps = [v for v in variants if any(ch.isupper() for ch in v)]
        words = (caps or variants)[0].split()
        small = {"and", "of", "in", "the", "to", "for", "on", "with", "or"}
        return " ".join(w if any(ch.isupper() for ch in w[1:]) or w.isupper()
                        else (w.lower() if i and w.lower() in small else w[:1].upper() + w[1:])
                        for i, w in enumerate(words))

    def distinct_words(key):
        return set(key.split()) - NAME_GENERIC_WORDS - _STOP

    def second_term(c, first, ranked, check_overlap=True):
        a = {id(p) for p in members[c] if first in tags[id(p)]}
        for score, key, here, _there in ranked:
            if key == first or here < NAME_MIN_SECOND_SHARE or distinct_words(key) & distinct_words(first):
                continue
            b = {id(p) for p in members[c] if key in tags[id(p)]}
            if check_overlap and len(a & b) / max(len(b), 1) >= 0.9:   # nearly always together: says the same thing
                continue
            return key
        return None

    rankings, picks = {}, {}
    for c in clusters:
        others = [p for d in clusters if d != c for p in members[d]]
        rankings[c] = ranking(c, others)
        if rankings[c]:
            first = rankings[c][0][1]
            picks[c] = [first] + [k for k in [second_term(c, first, rankings[c])] if k]

    # Clusters that share their first term: pick the second term that best
    # separates them from each other (e.g. ECT · brain imaging vs ECT · outcome).
    by_first = defaultdict(list)
    for c, terms in picks.items():
        by_first[terms[0]].append(c)
    for first, group in by_first.items():
        if len(group) < 2:
            continue
        for c in group:
            siblings = [p for d in group if d != c for p in members[d]]
            vs_rest = {k: s for s, k, _h, _t in rankings[c]}
            ranked = sorted(((s * vs_rest.get(k, 0), k, h, t) for s, k, h, t in ranking(c, siblings)
                             if k != first and vs_rest.get(k, 0) > 0), reverse=True)
            # (every paper here has the shared first term, so no overlap check)
            second = second_term(c, first, [r for r in ranked if r[2] >= 0.15], check_overlap=False)
            picks[c] = [first] + ([second] if second else [])

    names = dict(base_names)
    if header:
        print("\n" + header)
    for c in clusters:
        if c not in picks:
            print(f"  cluster {c} (n={len(members[c])}): {base_names.get(c)}   [no strong term — MeSH/keyword name kept]")
            continue
        names[c] = " · ".join(nice(k) for k in picks[c])
        if header:
            print(f"  cluster {c} (n={len(members[c])}): {names[c]}")
        else:
            print(f"  {base_names.get(c)} (n={len(members[c])}): {names[c]}")
        if DEBUG_LABELS:
            for score, key, here, there in rankings[c][:6]:
                print(f"      {nice(key):45s} score {score:.2f}   in {here:.0%} of this cluster vs "
                      f"{there:.0%} elsewhere{'   [lab term]' if key in lab else ''}")
    return names

def finalize_names(names):
    """Make names unique and apply LABEL_OVERRIDES."""
    out, seen = {}, Counter()
    for c in sorted(names):
        name = LABEL_OVERRIDES.get(names[c], names[c])
        seen[name] += 1
        out[c] = name if seen[name] == 1 else f"{name} ({seen[name]})"
    return out

def label_clusters(texts, labels, papers, weights=None, debug=None):
    """Name clusters. `weights` sets how much each paper counts (papers
    without abstract count NO_ABSTRACT_WEIGHT, the rest 1.0)."""
    if debug is None:
        debug = DEBUG_LABELS
    if weights is None:
        weights = [1.0] * len(papers)

    # Original casing for display ("ECT", "fMRI", "C9orf72"), keyed by lowercase
    display = {}
    def remember(original):
        display.setdefault(original.lower(), original)
        return original.lower()

    def pretty(term):
        words = display.get(term, term).split()
        out = []
        for w in words:
            keep = (w.isupper() and len(w) <= 5) or any(c.isupper() for c in w[1:]) and not w.isupper() \
                   or any(c.isdigit() for c in w)
            out.append(w if keep else w[:1].upper() + w[1:].lower())
        return " ".join(out)

    def dbg(msg):
        if debug:
            print(msg)

    groups = defaultdict(list)
    for i, (t, c) in enumerate(zip(texts, labels)):
        if c != -1:
            groups[c].append(i)

    BLACKLIST = {
        "van", "de", "den", "der", "het", "een", "ter",
        "humans", "human", "adult", "adults", "female", "females",
        "male", "males", "aged", "child", "children", "animal",
        "medicine", "psychology", "neuroscience",
        "health", "healthcare", "science", "clinical", "pathology",
        "psychiatry", "diagnosis", "prognosis", "treatment",
        "patients", "research", "study", "methods", "results","tests", "processing", "assisted"
    }

    # True filler/connector words only. Do NOT put clinically meaningful
    FUNCTION_WORDS = {
        "with", "without", "following", "after", "before", "during",
        "using", "among", "between", "within", "across", "versus",
        "into", "from", "than", "that", "this", "these", "those",
        "which", "when", "where", "while"
    }

    def term_words(term):
        """Alphabetic word tokens in a term, ignoring commas/hyphens/parens/etc.
        MeSH descriptors are often inverted with punctuation, e.g.
        'Aphasia, Primary Progressive' or 'Dementia, Frontotemporal' —
        punctuation shouldn't disqualify an otherwise clean term."""
        return re.findall(r"[a-z]+", term.lower())

    def is_clean(term):
        words = term_words(term)
        if not words:
            return False  # nothing alphabetic at all (numeric code, junk, etc.)
        if any(w in BLACKLIST for w in words):
            return False
        if any(w in FUNCTION_WORDS for w in words):
            return False
        return True

    def rejection_reason(term):
        """Explain why a raw (pre-is_clean) term got filtered, for debugging."""
        words = term_words(term)
        if not words:
            return "no alphabetic words"
        hit = next((w for w in words if w in BLACKLIST), None)
        if hit:
            return f"blacklisted word '{hit}'"
        hit = next((w for w in words if w in FUNCTION_WORDS), None)
        if hit:
            return f"function word '{hit}'"
        return None

    # Words too general to show that two terms mean the same thing
    GENERIC_WORDS = {"disorder", "disorders", "disease", "diseases", "therapy",
                     "syndrome", "syndromes", "dysfunction", "impairment"}

    def distinctive_words(term):
        return {w for w in term_words(display.get(term, term))
                if w not in GENERIC_WORDS and w not in ENGLISH_STOP_WORDS}

    def pick_diverse(ranked_terms, term_paper_sets, k=2, overlap_threshold=0.9):
        """Greedily pick up to k terms, best score first, skipping a term that
        repeats one already picked. A term counts as a repeat when
          - it shares a distinctive word with the picked term
            ("Frontotemporal Dementia" / "Frontotemporal Lobar Degeneration"), or
          - the two tag practically the same papers (Jaccard >= overlap_threshold).
        Terms that merely co-occur a lot but mean different things (e.g. ECT and
        major depressive disorder in an ECT cluster) are kept."""
        chosen, chosen_sets = [], []
        skipped = []
        for term in ranked_terms:
            tset = term_paper_sets.get(term, set())
            twords = distinctive_words(term)
            redundant_with = None
            for prev_term, pset in zip(chosen, chosen_sets):
                shared = twords & distinctive_words(prev_term)
                if shared:
                    redundant_with = (prev_term, f"shares word '{sorted(shared)[0]}'")
                    break
                if tset and pset:
                    jaccard = len(tset & pset) / len(tset | pset)
                    if jaccard >= overlap_threshold:
                        redundant_with = (prev_term, f"tags the same papers (jaccard={jaccard:.2f})")
                        break
            if redundant_with:
                skipped.append((term, redundant_with))
                continue
            chosen.append(term)
            chosen_sets.append(tset)
            if len(chosen) >= k:
                break
        # If everything left was mutually redundant, fill remaining slots by
        # score alone so we still return k labels rather than fewer.
        if len(chosen) < k:
            for term in ranked_terms:
                if term not in chosen:
                    chosen.append(term)
                if len(chosen) >= k:
                    break
        if debug and skipped:
            for term, (prev_term, why) in skipped:
                dbg(f"      skip {term!r:35s} — repeats picked {prev_term!r}: {why}")
        return chosen[:k]

    def dbg_candidates(label, raw_counts, scored, top_n=8):
        """Print the top raw candidates for a tier, their score, and clean/reject status."""
        if not debug or not raw_counts:
            return
        ranked_all = sorted(raw_counts, key=lambda k: -scored.get(k, 0))[:top_n]
        dbg(f"    {label} candidates (top {len(ranked_all)} of {len(raw_counts)}):")
        for term in ranked_all:
            reason = rejection_reason(term)
            status = "OK" if reason is None else f"REJECTED ({reason})"
            dbg(f"      {term!r:40s} raw={raw_counts[term]:.1f}  score={scored.get(term, 0):.4f}  {status}")

    def uninvert(term):
        """MeSH-style inverted names to natural order, so both spellings count
        as one term: "Depressive Disorder, Major" -> "Major Depressive Disorder",
        "Dementia, Frontotemporal" -> "Frontotemporal Dementia". Terms with more
        than one comma are left as they are."""
        parts = [p.strip() for p in term.split(",")]
        if len(parts) == 2 and all(parts):
            return f"{parts[1]} {parts[0]}"
        return term

    def kw_terms(i):
        """(term, weight) pairs for the keyword tier: OpenAlex keywords, plus
        title word-pairs at 0.4 that skip common English words (no length
        minimum, so acronyms like "ECT" or "PET" can appear)."""
        out = []
        for kw in papers[i].get("keywords", []):
            if kw:
                out.append((remember(uninvert(kw)), weights[i]))
        words = re.findall(r"[A-Za-z][\w\-]*", papers[i].get("title", ""))
        for w1, w2 in zip(words, words[1:]):
            if w1.lower() in ENGLISH_STOP_WORDS or w2.lower() in ENGLISH_STOP_WORDS:
                continue
            out.append((remember(f"{w1} {w2}"), 0.4 * weights[i]))
        # each term at most once per paper (keeps the highest weight)
        best = {}
        for term, w in out:
            best[term] = max(w, best.get(term, 0.0))
        return list(best.items())

    # ── pass 1: count MeSH per cluster and globally ──
    mesh_per_cluster        = {}
    mesh_papers_per_cluster = {}   # term -> set of paper indices tagged with it (for dedup)
    total_mesh_counts       = defaultdict(float)

    for c, indices in groups.items():
        counts       = defaultdict(float)
        term_papers  = defaultdict(set)
        for i in indices:
            # each term counts at most once per paper (OpenAlex repeats a MeSH
            # term once per sub-heading, e.g. "/methods", "/adverse effects")
            for t in {remember(uninvert(term)) for term in papers[i].get("mesh", []) if term}:
                counts[t] += weights[i]
                total_mesh_counts[t] += weights[i]
                term_papers[t].add(i)
        mesh_per_cluster[c]        = counts
        mesh_papers_per_cluster[c] = term_papers

    # ── pass 2: score and label ──
    names = {-1: "Unclustered"}
    for c, indices in groups.items():
        dbg(f"\n── cluster {c} (n={len(indices)}) ──")
        counts = mesh_per_cluster[c]
        dbg(f"    {len(counts)} distinct raw MeSH terms found")

        # c-TF-IDF: freq in cluster / freq across all clusters
        scored = {
            kw: freq / (total_mesh_counts[kw] ** DISCRIMINATIVE_POWER + 1)
            for kw, freq in counts.items()
        }
        clean_scored = {k: v for k, v in scored.items() if is_clean(k)}
        dbg_candidates("MeSH", counts, scored)
        ranked_by_score = sorted(clean_scored, key=lambda k: -clean_scored[k])
        ranked = pick_diverse(ranked_by_score, mesh_papers_per_cluster[c], k=2)

        if ranked:
            names[c] = " · ".join(pretty(k) for k in ranked)
            print(f"  cluster {c} (n={len(indices)}): {names[c]}  [mesh]")
            continue

        dbg(f"    → no clean MeSH candidates, falling back to keywords/title-bigrams")

        # ── fallback 1: discriminative keywords + title bigrams ──
        kw_counts   = defaultdict(float)
        kw_papers   = defaultdict(set)   # term -> set of paper indices (for dedup)
        for i in indices:
            for term, w in kw_terms(i):
                kw_counts[term] += w
                kw_papers[term].add(i)

        # global totals (keywords AND bigrams) across all clusters, so terms
        # common to the whole lab are penalised the same way as in the MeSH tier
        all_kw_total = defaultdict(float)
        for idx2 in groups.values():
            for i in idx2:
                for term, w in kw_terms(i):
                    all_kw_total[term] += w

        scored_kw = {
            kw: freq / (all_kw_total[kw] ** DISCRIMINATIVE_POWER + 1)
            for kw, freq in kw_counts.items()
        }
        clean_scored_kw = {k: v for k, v in scored_kw.items() if is_clean(k)}
        dbg(f"    {len(kw_counts)} distinct raw keyword/bigram candidates found")
        dbg_candidates("keyword/bigram", kw_counts, scored_kw)
        ranked_kw_by_score = sorted(clean_scored_kw, key=lambda k: -clean_scored_kw[k])
        ranked_kw = pick_diverse(ranked_kw_by_score, kw_papers, k=2)

        if ranked_kw:
            names[c] = " · ".join(pretty(k) for k in ranked_kw)
            print(f"  cluster {c} (n={len(indices)}): {names[c]}  [keywords]")
            continue

        dbg(f"    → no clean keyword/bigram candidates, falling back to TF-IDF on abstracts")

        # ── fallback 2: TF-IDF on abstracts ──
        cluster_texts = [texts[i] for i in indices]
        tfidf  = TfidfVectorizer(stop_words="english", max_features=2000,
                                 ngram_range=(2, 3), sublinear_tf=True)
        X      = tfidf.fit_transform(cluster_texts)
        wvec   = np.array([weights[i] for i in indices])
        scores = np.asarray(X.T @ wvec).ravel() / max(wvec.sum(), 1e-9)
        terms  = np.array(tfidf.get_feature_names_out())
        order  = scores.argsort()[::-1]

        if debug:
            top = terms[order][:8]
            top_scores = scores[order][:8]
            dbg(f"    TF-IDF candidates (top {len(top)} of {len(terms)}):")
            for term, sc in zip(top, top_scores):
                reason = rejection_reason(term)
                status = "OK" if reason is None else f"REJECTED ({reason})"
                dbg(f"      {term!r:40s} tfidf={sc:.4f}  {status}")

        # Build paper-index sets for the top clean candidates only (cheap:
        # just which local docs have a nonzero entry in that column) so we
        # can dedup near-synonym n-grams the same way as the other tiers.
        clean_ranked_tf = [t for t in terms[order] if is_clean(t)][:15]
        col_of = {t: i for i, t in enumerate(terms)}
        tf_paper_sets = {}
        for t in clean_ranked_tf:
            rows = X[:, col_of[t]].nonzero()[0]
            tf_paper_sets[t] = {indices[r] for r in rows}

        ranked_tf = pick_diverse(clean_ranked_tf, tf_paper_sets, k=2)
        names[c] = " · ".join(pretty(t) for t in ranked_tf) if ranked_tf else f"Cluster {c}"
        tier = "tfidf" if ranked_tf else "fallback (no clean terms at any tier)"
        print(f"  cluster {c} (n={len(indices)}): {names[c]}  [{tier}]")

    return names

# ── WRITE ─────────────────────────────────────────────────────────────────────
def write_atlas(papers, three_js, orbit_js):
    html = HTML
    html = html.replace("__THREE_JS__",       three_js)
    html = html.replace("__ORBIT_JS__",       orbit_js)
    html = html.replace("__PAPERS_DATA__",    json.dumps(papers, ensure_ascii=False))
    html = html.replace("__PI_AUTHORS__",     json.dumps(sorted(PI_AUTHOR_NAMES), ensure_ascii=False))
    colors = CLUSTER_COLORS
    if USE_THEMES:
        colors = [t.get("color") or CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, t in enumerate(THEMES)]
    html = html.replace("__CLUSTER_COLORS__", json.dumps(colors, ensure_ascii=False))
    OUT_PATH.write_text(html, encoding="utf-8")
    size_mb = OUT_PATH.stat().st_size / 1_048_576
    print(f"\n✓ Wrote {OUT_PATH}")
    print(f"  {len(papers)} papers · {DIMS}D · {size_mb:.1f} MB")
    print(f"  {len(PI_AUTHOR_NAMES)} PI authors in dropdown: {sorted(PI_AUTHOR_NAMES)}")
    print("  Open atlas.html directly in any browser — no server needed.")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    three_js, orbit_js = fetch_js_libraries()

    print("\nReading lab members from the website…")
    load_website()
    resolve_author_ids()
    if not AUTHOR_IDS:
        raise SystemExit(
            "\nNo lab members found: the website couldn't be read, or nobody on it has an "
            "ORCID / OpenAlex ID.\nCheck WEBSITE_DIR (a local copy of the website repo) or "
            "WEBSITE_GITHUB (+ GITHUB_TOKEN for a private repo), or add IDs to EXTRA_AUTHOR_IDS.")

    print("\nFetching from OpenAlex…")
    raw_works = fetch_openalex()
    papers, no_abs = normalize(raw_works)
    print(f"{len(papers)} papers with usable abstracts, {len(no_abs)} without.")
    if USE_PUBMED and no_abs:
        found, no_abs = fill_from_pubmed(no_abs)
        papers += found
        print(f"PubMed supplied {len(found)} abstracts: now {len(papers)} with, {len(no_abs)} without.")
    no_abs = keep_no_abstract(no_abs)
    print(f"Keeping {len(no_abs)} articles/letters without abstract.")
    print(f"{len(PI_AUTHOR_NAMES)} PI author names collected: {sorted(PI_AUTHOR_NAMES)}")

    papers, model, vecs, labels, coords = build(papers)
    papers += assign_no_abstract(papers, no_abs, model, vecs, labels, coords)
    papers  = apply_names(papers)
    write_atlas(papers, three_js, orbit_js)