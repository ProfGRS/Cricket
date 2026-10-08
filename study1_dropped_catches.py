"""Dropped catches vs sixes: no-ball rate in the rest of the over.

Input: the Cricket Scorecard and Commentary Dataset (Kaggle, Tahlan, 2021), read directly
from the downloaded zip ("archive (1).zip"), using the per-match ESPNcricinfo commentary files.
Expected result (international / all matches):
  six            2.28 [1.74, 2.99], 77 no-balls  /  2.07 [1.68, 2.55], 138 no-balls
  dropped catch  1.01 [0.54, 1.90], 10 no-balls  /  1.17 [0.70, 1.95], 17 no-balls
"""
import io, re, sys, zipfile
import numpy as np
import pandas as pd

ZIP = sys.argv[1] if len(sys.argv) > 1 else "archive (1).zip"
WINDOW, LOOKBACK = 5, 6

# 1. Load commentary for every match, keeping the file's row order
cols = ["PlayType_description", "Innings", "Commentary", "Commentary_short", "Bowler_id", "Over_number"]
parts = []
with zipfile.ZipFile(ZIP) as z:
    meta = pd.read_csv(io.BytesIO(z.read("INTERNATIONAL_MATCH.csv")))[["Match_number", "Match_Type"]]
    for name in z.namelist():
        if not name.endswith("_COMMENTARY.csv"):
            continue
        try:
            c = pd.read_csv(io.BytesIO(z.read(name)), usecols=cols, low_memory=False)
        except Exception:
            continue
        c["mid"] = int(name.split("/")[-1].split("_")[0])
        c["row"] = np.arange(len(c))
        parts.append(c)
d = pd.concat(parts, ignore_index=True).merge(meta, left_on="mid", right_on="Match_number", how="left")

# 2. Code no-balls and dropped catches
d["nb"] = d.Commentary_short.astype(str).str.contains("no ball", case=False)
text = d.Commentary.astype(str).str.replace(r"<[^>]+>", "", regex=True)
DROP = (r"(dropped by\b|\bdropped!|\bdropped\.|\bdropped,|"
        r"\bdropped at (?:first|second|third|fourth|gully|slip|point|cover|mid|long|deep|fine|square|short|leg|the)|"
        r"drops? (?:a|another|the|an) (?:easy |simple |tough |difficult |sharp |straightforward |regulation |hard |sitter|catch|chance|dolly)|"
        r"\bput(?:s|ting)? (?:down|it down)\b|\bputs? down (?:a|the|another)|"
        r"spill(?:s|ed)? (?:the chance|a (?:sitter|catch|chance|simple|tough|sharp|regulation)|another|the catch|a dolly)|"
        r"\bgrass(?:es|ed)\b|\bshelled\b|\bmissed chance\b|\bchance missed\b|has dropped (?:it|him|that|the)|"
        r"dropped (?:a|the|another) (?:catch|chance|sitter|dolly|simple|easy|tough|sharp)|"
        r"dropped catch|\bdrops it!|\bdropped him\b|\bdrops him\b|\bdropped again\b)")
NOT_DROP = (r"(dropped short|drops? (?:it|this|one) (?:short|shorter|down|in|into|towards|on|at|to|wide)|"
            r"drops? back|ball drops|drops (?:his|the) (?:bat|hands|wrists)|stump)")
d["drop"] = text.str.contains(DROP, case=False, regex=True) & ~text.str.contains(NOT_DROP, case=False, regex=True)

# 3. Strata: bowler-innings, deliveries in file order
d["g"] = pd.factorize(d.mid.astype(str) + "|" + d.Innings.astype(str) + "|" + d.Bowler_id.astype(str))[0]
d = d.sort_values(["g", "mid", "Innings", "row"]).reset_index(drop=True)
g, ov, N = d.g.values, d.Over_number.values, len(d)
nb = d.nb.values
prev_nb = (pd.Series(nb).groupby(g)
           .transform(lambda s: s.shift(1).rolling(LOOKBACK, min_periods=1).max())
           .fillna(0).astype(bool).values)
clean = ~nb & ~prev_nb
intl = d.Match_Type.isin(["TEST_MATCH", "ODI", "T20I"]).values

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

events = {"six": (d.PlayType_description == "six").values & clean,
          "dropped catch": d["drop"].values & clean}
for name, ev in events.items():
    w = window(ev)
    print(f"{name:14s} n = {ev.sum():5d} | international {mantel_haenszel(w, ~ev & intl)} | all {mantel_haenszel(w, ~ev)}")
