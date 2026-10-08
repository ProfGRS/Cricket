#!/usr/bin/env python3
"""
Study 2 analysis: Does accumulated international bowling experience moderate the
post-six increase in no-balls?

This script reproduces the Study 2 analyses reported in:
    Opponent-Imposed Setbacks Disproportionately Disrupt Skilled Action in
    Elite Cricket Bowlers

Input
-----
The Ultimate Ball-by-Ball Cricket Dataset (Adaikalam, 2025), supplied either as:
  1) the Kaggle ZIP archive containing ball_by_ball_data.csv, or
  2) the extracted ball_by_ball_data.csv file.

Example
-------
python Study2_experience_analysis.py --input archive.zip --output-dir study2_outputs

Dependencies
------------
pandas, numpy, scipy, statsmodels, matplotlib

Important implementation detail
-------------------------------
Study 2 uses the same post-event window as Study 1: the next five deliveries
sent down by the bowler in the same over after an eligible six, counting wides
and no-balls as deliveries for purposes of the five-delivery window. The six
event delivery itself is excluded from both the post-six window and the
comparison set.
"""

from __future__ import annotations

import argparse
import io
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


INTERNATIONAL_FORMATS = {"TEST", "ODI", "T20"}
COHORT_START = pd.Timestamp("2006-01-01")
DEFAULT_SEED = 20261008
DEFAULT_BOOTSTRAPS = 10_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        required=True,
        help="Path to archive.zip or ball_by_ball_data.csv",
    )
    p.add_argument(
        "--output-dir",
        default="study2_outputs",
        help="Directory for tables and figure (default: study2_outputs)",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAPS,
        help=f"Number of match-bootstrap replicates (default: {DEFAULT_BOOTSTRAPS})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Bootstrap random seed (default: {DEFAULT_SEED})",
    )
    return p.parse_args()


def _open_csv(path: Path):
    """Return a readable handle to ball_by_ball_data.csv."""
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        names = zf.namelist()
        candidates = [n for n in names if n.endswith("ball_by_ball_data.csv")]
        if not candidates:
            zf.close()
            raise FileNotFoundError("ZIP does not contain ball_by_ball_data.csv")
        handle = zf.open(candidates[0])
        return zf, handle
    return None, open(path, "rb")


def load_international_deliveries(path: Path) -> pd.DataFrame:
    """
    Read the source in chunks and retain men's Tests, ODIs, and T20Is.

    Row order is deliberately preserved because the source is already in
    delivery order and the event coding below uses preceding/following rows.
    """
    usecols = [
        "match_id",
        "start_date",
        "innings",
        "ball",
        "bowler",
        "runs_off_bat",
        "noballs",
        "format",
        "type",
        "gender",
    ]

    owner, handle = _open_csv(path)
    parts: list[pd.DataFrame] = []
    try:
        for ch in pd.read_csv(handle, usecols=usecols, chunksize=500_000, low_memory=False):
            keep = (
                ch["gender"].eq("male")
                & ch["type"].eq("Intl")
                & ch["format"].isin(INTERNATIONAL_FORMATS)
            )
            ch = ch.loc[keep].copy()
            ch["noballs"] = ch["noballs"].fillna(0).astype("int8")
            parts.append(ch.drop(columns=["gender", "type"]))
    finally:
        handle.close()
        if owner is not None:
            owner.close()

    df = pd.concat(parts, ignore_index=True)
    df["start_date"] = pd.to_datetime(df["start_date"])
    return df


def add_event_and_exposure_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Code eligible sixes and the five-delivery post-six window used in Study 1.

    Eligible six:
      * six runs off the bat,
      * not itself a no-ball,
      * not within the previous six deliveries by the same bowler in the same
        bowler-innings after a no-ball.

    Post-six delivery:
      * one of the next five deliveries sent down by that bowler in the SAME
        OVER after an eligible six. Wides and no-balls count toward these five
        deliveries because each appears as a delivery row in the source.
      * if another eligible six occurs, its delivery is an event delivery and is
        not itself analyzed as post-six; it starts a new five-delivery window.
    """
    d = df.copy()
    d["over"] = np.floor(d["ball"]).astype("int16")

    bowler_innings = ["match_id", "innings", "bowler"]
    nb_group = d.groupby(bowler_innings, observed=True, sort=False)["noballs"]

    prior_no_ball = np.zeros(len(d), dtype=bool)
    for lag in range(1, 7):
        prior_no_ball |= nb_group.shift(lag).fillna(0).to_numpy() > 0

    d["eligible_six"] = (
        d["runs_off_bat"].eq(6).to_numpy()
        & d["noballs"].eq(0).to_numpy()
        & ~prior_no_ball
    )

    # Position among all deliveries sent down in this over, including wides and
    # no-balls. Locate the most recent eligible six and retain only distances
    # 1..5 from it. A later eligible six resets the window.
    over_group = ["match_id", "innings", "over", "bowler"]
    pos = d.groupby(over_group, observed=True, sort=False).cumcount().to_numpy()
    event_pos = np.where(d["eligible_six"].to_numpy(), pos, -10_000_000)
    d["_event_pos"] = event_pos
    last_event_pos = (
        d.groupby(over_group, observed=True, sort=False)["_event_pos"]
        .cummax()
        .to_numpy()
    )
    distance = pos - last_event_pos
    d["post_six"] = (distance >= 1) & (distance <= 5)
    d["no_ball"] = d["noballs"].gt(0).astype(int)
    return d.drop(columns=["_event_pos"])


def add_experience(df: pd.DataFrame) -> pd.DataFrame:
    """Add prior international deliveries and first observed appearance."""
    d = df.copy()

    by_match = (
        d.groupby(["bowler", "match_id", "start_date"], observed=True, sort=False)
        .size()
        .rename("match_deliveries")
        .reset_index()
        .sort_values(["bowler", "start_date", "match_id"])
    )
    by_match["prior_deliveries"] = (
        by_match.groupby("bowler", observed=True)["match_deliveries"].cumsum()
        - by_match["match_deliveries"]
    )
    by_match["first_observed"] = by_match.groupby("bowler", observed=True)[
        "start_date"
    ].transform("min")

    # Converting bowler to string prevents category-merging issues across frames.
    d["bowler"] = d["bowler"].astype(str)
    by_match["bowler"] = by_match["bowler"].astype(str)

    return d.merge(
        by_match[["bowler", "match_id", "prior_deliveries", "first_observed"]],
        on=["bowler", "match_id"],
        how="left",
        validate="many_to_one",
    )


def make_strata(df: pd.DataFrame, informative_only: bool = False) -> pd.DataFrame:
    """Create bowler-innings counts for post-six and comparison deliveries."""
    keys = ["match_id", "innings", "bowler", "prior_deliveries"]

    agg = (
        df.groupby(keys + ["post_six"], observed=True)["no_ball"]
        .agg(["sum", "count"])
        .reset_index()
    )
    pv = agg.pivot(index=keys, columns="post_six", values=["sum", "count"]).fillna(0)

    for top in ["sum", "count"]:
        for exposure in [False, True]:
            if (top, exposure) not in pv.columns:
                pv[(top, exposure)] = 0

    st = pv.index.to_frame(index=False)
    st["post_nb"] = pv[("sum", True)].to_numpy(float)
    st["post_n"] = pv[("count", True)].to_numpy(float)
    st["comp_nb"] = pv[("sum", False)].to_numpy(float)
    st["comp_n"] = pv[("count", False)].to_numpy(float)

    if informative_only:
        st = st[
            (st["post_n"] > 0)
            & (st["comp_n"] > 0)
            & ((st["post_nb"] + st["comp_nb"]) > 0)
        ].copy()
        st["offset"] = np.log(st["post_n"] / st["comp_n"])

    return st


def two_way_cluster_glm(
    df: pd.DataFrame,
    transform: str = "linear",
) -> dict:
    """
    Conditional-binomial-style grouped GLM.

    Conditional on the total number of no-balls in a bowler-innings stratum,
    the outcome is whether those no-balls occurred on post-six rather than
    comparison deliveries. log(post_n / comp_n) is the offset.
    """
    st = make_strata(df, informative_only=True)

    if transform == "linear":
        x = st["prior_deliveries"] / 1000.0
        x_name = "per_1000_prior_deliveries"
    elif transform == "log10":
        x = np.log10(st["prior_deliveries"] + 1)
        x_name = "log10_prior_plus_1"
    else:
        raise ValueError("transform must be 'linear' or 'log10'")

    X = sm.add_constant(pd.DataFrame({x_name: x}), has_constant="add")
    endog = np.column_stack([st["post_nb"], st["comp_nb"]])

    fit = sm.GLM(
        endog,
        X,
        family=sm.families.Binomial(),
        offset=st["offset"],
    ).fit()

    cov, _, _ = cov_cluster_2groups(
        fit,
        st["match_id"].to_numpy(),
        pd.factorize(st["bowler"])[0],
    )
    beta = fit.params.to_numpy()
    se = np.sqrt(np.diag(cov))
    z = beta / se
    p = 2 * norm.sf(np.abs(z))
    crit = norm.ppf(0.975)
    lo = beta - crit * se
    hi = beta + crit * se

    j = 1
    return {
        "term": x_name,
        "multiplier": float(np.exp(beta[j])),
        "ci_low": float(np.exp(lo[j])),
        "ci_high": float(np.exp(hi[j])),
        "p": float(p[j]),
        "informative_strata": int(len(st)),
        "matches": int(st["match_id"].nunique()),
        "post_six_no_balls": int(st["post_nb"].sum()),
        "comparison_no_balls": int(st["comp_nb"].sum()),
    }


def mh_contributions(df: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mantel-Haenszel rate-ratio contributions, stratified by bowler-innings."""
    keys = ["match_id", "innings", "bowler", group_col]
    agg = (
        df.groupby(keys + ["post_six"], observed=True)["no_ball"]
        .agg(["sum", "count"])
        .reset_index()
    )
    pv = agg.pivot(index=keys, columns="post_six", values=["sum", "count"]).fillna(0)

    for top in ["sum", "count"]:
        for exposure in [False, True]:
            if (top, exposure) not in pv.columns:
                pv[(top, exposure)] = 0

    meta = pv.index.to_frame(index=False)
    y1 = pv[("sum", True)].to_numpy(float)
    n1 = pv[("count", True)].to_numpy(float)
    y0 = pv[("sum", False)].to_numpy(float)
    n0 = pv[("count", False)].to_numpy(float)
    n = n1 + n0

    meta["A"] = y1 * n0 / n
    meta["B"] = y0 * n1 / n
    meta["post_nb"] = y1
    meta["post_n"] = n1
    meta["comp_nb"] = y0
    meta["comp_n"] = n0

    point = (
        meta.groupby(group_col, observed=True)
        .agg(
            A=("A", "sum"),
            B=("B", "sum"),
            post_nb=("post_nb", "sum"),
            post_n=("post_n", "sum"),
            comp_nb=("comp_nb", "sum"),
            comp_n=("comp_n", "sum"),
            strata=("A", "size"),
        )
    )
    point["RR"] = point["A"] / point["B"]
    return point, meta


def bootstrap_band_rrs(
    meta: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    R: int,
    seed: int,
) -> pd.DataFrame:
    """Match bootstrap CIs for grouped MH rate ratios."""
    m = (
        meta.groupby(["match_id", group_col], observed=True)
        .agg(A=("A", "sum"), B=("B", "sum"))
        .unstack(fill_value=0)
    )

    A = np.column_stack(
        [m.get(("A", g), pd.Series(0.0, index=m.index)).to_numpy() for g in group_order]
    )
    B = np.column_stack(
        [m.get(("B", g), pd.Series(0.0, index=m.index)).to_numpy() for g in group_order]
    )

    rng = np.random.default_rng(seed)
    nmatch = len(m)
    probs = np.full(nmatch, 1.0 / nmatch)
    boots = np.empty((R, len(group_order)))

    for r in range(R):
        counts = rng.multinomial(nmatch, probs)
        boots[r] = (counts @ A) / (counts @ B)

    rows = []
    for j, g in enumerate(group_order):
        q = np.quantile(boots[:, j], [0.025, 0.5, 0.975])
        rows.append({"group": g, "boot_low": q[0], "boot_median": q[1], "boot_high": q[2]})
    return pd.DataFrame(rows)


def bootstrap_low_high_contrast(
    df: pd.DataFrame,
    R: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match-bootstrap RRs for <1,000 and >=5,000 deliveries and their ratio."""
    d = df[(df["prior_deliveries"] < 1000) | (df["prior_deliveries"] >= 5000)].copy()
    d["experience_group"] = np.where(
        d["prior_deliveries"] < 1000,
        "<1,000",
        ">=5,000",
    )

    point, meta = mh_contributions(d, "experience_group")
    order = ["<1,000", ">=5,000"]

    m = (
        meta.groupby(["match_id", "experience_group"], observed=True)
        .agg(A=("A", "sum"), B=("B", "sum"))
        .unstack(fill_value=0)
    )
    A = np.column_stack(
        [m.get(("A", g), pd.Series(0.0, index=m.index)).to_numpy() for g in order]
    )
    B = np.column_stack(
        [m.get(("B", g), pd.Series(0.0, index=m.index)).to_numpy() for g in order]
    )

    rng = np.random.default_rng(seed)
    nmatch = len(m)
    probs = np.full(nmatch, 1.0 / nmatch)
    boots = np.empty((R, 3))

    for r in range(R):
        counts = rng.multinomial(nmatch, probs)
        rr = (counts @ A) / (counts @ B)
        boots[r, 0] = rr[0]
        boots[r, 1] = rr[1]
        boots[r, 2] = rr[0] / rr[1]

    q = np.quantile(boots, [0.025, 0.5, 0.975], axis=0)
    summary = pd.DataFrame(
        {
            "quantity": ["RR <1,000", "RR >=5,000", "RR ratio (<1,000 / >=5,000)"],
            "point": [
                point.loc["<1,000", "RR"],
                point.loc[">=5,000", "RR"],
                point.loc["<1,000", "RR"] / point.loc[">=5,000", "RR"],
            ],
            "boot_low": q[0],
            "boot_median": q[1],
            "boot_high": q[2],
        }
    )
    return summary, point


def save_figure(band_results: pd.DataFrame, output_path: Path) -> None:
    """Save the Study 2 forest plot."""
    import matplotlib.pyplot as plt

    order = ["<500", "500-1,999", "2,000-4,999", ">=5,000"]
    r = band_results.set_index("group").loc[order].reset_index()
    y = np.arange(len(r))
    x = r["RR"].to_numpy()
    lo = r["boot_low"].to_numpy()
    hi = r["boot_high"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(x, y, xerr=np.vstack([x - lo, hi - x]), fmt="o", capsize=3)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_yticks(y, r["group"])
    ax.invert_yaxis()
    ax.set_xlabel("Post-six no-ball rate ratio")
    ax.set_ylabel("Prior international deliveries")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading men's international deliveries...")
    df = load_international_deliveries(input_path)
    print(f"Rows: {len(df):,}; matches: {df['match_id'].nunique():,}; bowlers: {df['bowler'].nunique():,}")

    df = add_event_and_exposure_flags(df)
    df = add_experience(df)

    # Report cohort size before removing event deliveries, as in the manuscript.
    cohort_full = df[df["first_observed"] >= COHORT_START].copy()
    print(
        "Primary cohort before event-delivery exclusion: "
        f"{cohort_full['bowler'].nunique():,} bowlers; "
        f"{len(cohort_full):,} deliveries; "
        f"{int(cohort_full['eligible_six'].sum()):,} eligible sixes; "
        f"{int((cohort_full['post_six'] & cohort_full['no_ball'].astype(bool)).sum()):,} raw post-six no-balls"
    )

    # As in Study 1, eligible six event deliveries are excluded from both the
    # post-event and comparison sets. Exposure coding is done before this step
    # so each eligible six can still start/reset a five-delivery window.
    analysis_df = df.loc[~df["eligible_six"]].copy()
    cohort = analysis_df[analysis_df["first_observed"] >= COHORT_START].copy()

    # Continuous moderation models
    primary = two_way_cluster_glm(cohort, transform="linear")
    log_sens = two_way_cluster_glm(cohort, transform="log10")
    all_bowlers = two_way_cluster_glm(analysis_df, transform="linear")

    models = pd.DataFrame(
        [
            {"analysis": "Primary: per 1,000 prior deliveries", **primary},
            {"analysis": "Sensitivity: log10(prior + 1)", **log_sens},
            {"analysis": "Sensitivity: all bowlers, per 1,000", **all_bowlers},
        ]
    )
    models.to_csv(outdir / "study2_moderation_models.csv", index=False)

    # Four descriptive experience bands
    bins = [-1, 499, 1999, 4999, 1e12]
    labels = ["<500", "500-1,999", "2,000-4,999", ">=5,000"]
    cohort["experience_band"] = pd.cut(
        cohort["prior_deliveries"], bins=bins, labels=labels
    )
    point, meta = mh_contributions(cohort, "experience_band")
    boot = bootstrap_band_rrs(meta, "experience_band", labels, args.bootstrap, args.seed)
    band_results = point.reset_index().rename(columns={"experience_band": "group"}).merge(
        boot, on="group", how="left"
    )
    band_results.to_csv(outdir / "study2_experience_bands.csv", index=False)

    # Direct low-versus-high contrast
    contrast, _ = bootstrap_low_high_contrast(cohort, args.bootstrap, args.seed)
    contrast.to_csv(outdir / "study2_low_high_contrast.csv", index=False)

    # Figure
    save_figure(band_results, outdir / "study2_experience_figure.png")

    print("\nContinuous moderation models")
    print(
        models[
            ["analysis", "multiplier", "ci_low", "ci_high", "p", "informative_strata"]
        ].to_string(index=False)
    )

    print("\nExperience-band Mantel-Haenszel estimates")
    print(
        band_results[
            ["group", "RR", "boot_low", "boot_high", "post_nb", "post_n"]
        ].to_string(index=False)
    )

    print("\nLow-versus-high experience contrast")
    print(contrast.to_string(index=False))

    print(f"\nOutputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
