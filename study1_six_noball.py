"""Study 1: no-ball rate after a six (primary estimate).

Input: ball_by_ball_data.csv from "The Ultimate Ball-by-Ball Cricket Dataset" (Kaggle),
read directly from the downloaded zip. Output: Mantel-Haenszel rate ratio with 95% CI.
Expected result: RR = 2.09 [1.89, 2.31], 609 contributing no-balls, 47,124 sixes.
"""
import subprocess, sys
import numpy as np
import pandas as pd

ZIP = sys.argv[1] if len(sys.argv) > 1 else "archive.zip"
WINDOW = 5      # next five deliveries the bowler sends down in the same over (incl. wides/no-balls)
LOOKBACK = 6    # exclude sixes within six deliveries after the bowler's previous no-ball

# 1. Load men's international matches, keeping file order within each match
cols = ["match_id", "innings", "ball", "bowler", "runs_off_bat", "noballs", "gender", "type"]
src = subprocess.Popen(["unzip", "-p", ZIP], stdout=subprocess.PIPE).stdout
parts = [ch[(ch.gender == "male") & (ch.type == "Intl")]
         for ch in pd.read_csv(src, usecols=cols, chunksize=1_000_000, low_memory=False)]
d = pd.concat(parts, ignore_index=True)
d["ord"] = np.arange(len(d))
d["over"] = np.floor(d.ball).astype(int)

# 2. Order deliveries within each bowler-innings (the stratum)
d["g"] = pd.factorize(d.match_id.astype(str) + "|" + d.innings.astype(str) + "|" + d.bowler.astype(str))[0]
d = d.sort_values(["g", "ord"]).reset_index(drop=True)
g, ov, N = d.g.values, d.over.values, len(d)
nb = (d.noballs.fillna(0) > 0).values

# 3. Events: sixes not struck off a no-ball and not within LOOKBACK deliveries of a previous no-ball
prev_nb = (pd.Series(nb).groupby(g)
           .transform(lambda s: s.shift(1).rolling(LOOKBACK, min_periods=1).max())
           .fillna(0).astype(bool).values)
event = (d.runs_off_bat == 6).values & ~nb & ~prev_nb

# 4. Window: next WINDOW deliveries by the same bowler in the same over
win = np.zeros(N, bool)
idx = np.where(event)[0]
for k in range(1, WINDOW + 1):
    j = idx + k
    ok = j < N
    j, s = j[ok], idx[ok]
    ok = (g[j] == g[s]) & (ov[j] == ov[s])
    win[j[ok]] = True

# 5. Mantel-Haenszel rate ratio: window vs the bowler's other deliveries in the innings.
#    Event deliveries are excluded from both groups; strata lacking either group drop out.
base = ~event
t = pd.DataFrame({"g": g[base], "y": nb[base].astype(float), "w": win[base]})
a = t[t.w].groupby("g").y.agg(["sum", "count"])
b = t[~t.w].groupby("g").y.agg(["sum", "count"])
j = a.join(b, lsuffix="1", rsuffix="0", how="inner")
T = j.count1 + j.count0
num = (j.sum1 * j.count0 / T).sum()
den = (j.sum0 * j.count1 / T).sum()
rr = num / den
var = ((j.count1 * j.count0 * (j.sum1 + j.sum0)) / T**2).sum() / (num * den)  # Greenland & Robins (1985)
lo, hi = np.exp(np.log(rr) + np.array([-1.96, 1.96]) * np.sqrt(var))

print(f"sixes = {event.sum():,}")
print(f"RR = {rr:.2f} [{lo:.2f}, {hi:.2f}]; contributing no-balls = {int(j.sum1.sum())}; "
      f"deliveries in window = {int(j.count1.sum()):,}")
