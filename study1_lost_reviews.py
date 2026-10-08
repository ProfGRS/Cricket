"""Lost reviews vs sixes: no-ball rate in the rest of the over.

Input: Cricsheet JSON match files (https://cricsheet.org/downloads/), one or more folders,
e.g. the unzipped tests_json and odis_json downloads. Men's Tests and ODIs are analysed.
A lost review is a review requested by the fielding (bowling) side whose decision was "struck down".

Usage: python3 study1_lost_reviews.py tests_json/ odis_json/
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

FOLDERS = sys.argv[1:] or ["tests_json", "odis_json"]
WINDOW, LOOKBACK = 5, 6

# 1. One row per delivery (wides and no-balls included), in match order
rows = []
for folder in FOLDERS:
    for f in glob.glob(os.path.join(folder, "*.json")):
        try:
            m = json.load(open(f))
        except Exception:
            continue
        info = m["info"]
        if info.get("gender") != "male" or info.get("match_type") not in ("Test", "ODI"):
            continue
        mid = info["match_type"] + "|" + os.path.basename(f)[:-5]
        for i, inn in enumerate(m["innings"]):
            if inn.get("super_over"):
                continue
            seq = 0
            for over in inn.get("overs", []):
                for dl in over["deliveries"]:
                    seq += 1
                    runs = dl["runs"]
                    rv = dl.get("review", {})
                    lost = bool(rv) and rv.get("by") != inn["team"] and rv.get("decision") == "struck down"
                    rows.append((mid, i, seq, over["over"], dl["bowler"],
                                 "noballs" in dl.get("extras", {}),
                                 runs["batter"] == 6 and not runs.get("non_boundary"),
                                 lost, bool(rv.get("umpires_call"))))
d = pd.DataFrame(rows, columns=["mid", "inn", "seq", "over", "bowler", "nb", "six", "lost", "umpires_call"])
print(f"matches = {d.mid.nunique():,}; deliveries = {len(d):,}; no-balls = {d.nb.sum():,}")

# 2. Strata: bowler-innings
d["g"] = pd.factorize(d.mid + "|" + d.inn.astype(str) + "|" + d.bowler)[0]
d = d.sort_values(["g", "seq"]).reset_index(drop=True)
g, ov, N, nb = d.g.values, d.over.values, len(d), d.nb.values
prev_nb = (pd.Series(nb).groupby(g)
           .transform(lambda s: s.shift(1).rolling(LOOKBACK, min_periods=1).max())
           .fillna(0).astype(bool).values)
clean = ~nb & ~prev_nb

def window(event):
    w = np.zeros(N, bool)
    idx = np.where(event)[0]
    for k in range(1, WINDOW + 1):
        j = idx + k
        ok = j < N
        j, s = j[ok], idx[ok]
        ok = (g[j] == g[s]) & (ov[j] == ov[s])
        w[j[ok]] = True
    return w

def mantel_haenszel(win, base):
    t = pd.DataFrame({"g": g[base], "y": nb[base].astype(float), "w": win[base]})
    a = t[t.w].groupby("g").y.agg(["sum", "count"])
    b = t[~t.w].groupby("g").y.agg(["sum", "count"])
    j = a.join(b, lsuffix="1", rsuffix="0", how="inner")
    T = j.count1 + j.count0
    num, den = (j.sum1 * j.count0 / T).sum(), (j.sum0 * j.count1 / T).sum()
    rr = num / den
    var = ((j.count1 * j.count0 * (j.sum1 + j.sum0)) / T**2).sum() / (num * den)
    lo, hi = np.exp(np.log(rr) + np.array([-1.96, 1.96]) * np.sqrt(var))
    return f"{rr:.2f} [{lo:.2f}, {hi:.2f}], {int(j.sum1.sum())} no-balls"

events = {"six": d.six.values & clean,
          "lost review": d.lost.values & clean,
          "lost review, umpire's call": (d.lost & d.umpires_call).values & clean}
for name, ev in events.items():
    print(f"{name:27s} n = {ev.sum():6d} | {mantel_haenszel(window(ev), ~ev)}")
