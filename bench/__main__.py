"""python -m bench <command>

  fetch [dfl|metrica|all]          download + convert external matches (DFL: one at a time)
  build-hackathon                  convert the gated BAR-ATH data (run locally, after Oct 20)
  relabel                          hackathon-style possession labels on DFL -> external/dfl_hk/
  reception [--source hackathon]   dead until the first reception -> external/dfl_rx/ (dfl_hk_rx/)
  stats [MATCHES] [--profile P]    aggregate stats of matches, as degraded by P
  run --method M [MATCHES] ...     score a method, print a board + diagnostics, save a report
  plot --run TAG --match NAME ...  timeline strip (GT / prediction / ball seen / errors) as PNG

MATCHES: dfl, metrica, hackathon, all, or match names (default: dfl). With
MTAH_DFL_LABELS=hackathon, "dfl" reads the relabelled matches (bench/relabel.py); with
MTAH_DFL_LABELS=reception, the ones from bench/reception.py. Hackathon output is
aggregate-only on the console; per-frame artefacts stay in reports/ on your machine.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from . import EXTERNAL_DIR, REPORTS_DIR, dfl, epts
from .degrade import PROFILES, degrade
from .evaluate import diagnose, score
from .match import D, Match, graded_mask


def _dfl_dir():
    """MTAH_DFL_LABELS: unset -> DFL's labels; hackathon -> dfl_hk; hackathon-<tag> -> dfl_hk_<tag>;
    reception -> dfl_rx."""
    from .relabel import out_dir
    v = os.environ.get("MTAH_DFL_LABELS", "")
    if not v:
        return os.path.join(EXTERNAL_DIR, "dfl")
    if v == "reception":
        from .reception import OUT_DIR
        if not os.path.isdir(OUT_DIR):
            sys.exit("MTAH_DFL_LABELS=reception: {} missing; run `python -m bench reception`".format(OUT_DIR))
        return OUT_DIR
    if v == "hackathon" or v.startswith("hackathon-"):
        d = out_dir(v[len("hackathon-"):] if "-" in v else None)
        if not os.path.isdir(d):
            sys.exit("MTAH_DFL_LABELS={}: {} missing; run `python -m bench relabel`".format(v, d))
        return d
    sys.exit("MTAH_DFL_LABELS must be unset, 'hackathon', 'hackathon-<tag>' or 'reception'")


def _metrica_dir():
    from .relabel import METRICA_OUT_DIR
    if os.environ.get("MTAH_DFL_LABELS", "").startswith("hackathon") and os.path.isdir(METRICA_OUT_DIR):
        return METRICA_OUT_DIR
    return epts.METRICA_DIR


def match_paths(selectors):
    groups = {
        "dfl": sorted(glob.glob(os.path.join(_dfl_dir(), "*.npz"))),
        # Metrica follows MTAH_DFL_LABELS=hackathon too (relabelled copies in external/metrica_hk)
        "metrica": sorted(glob.glob(os.path.join(
            _metrica_dir(), "metrica-game*.npz"))),
        "pff": sorted(glob.glob(os.path.join(EXTERNAL_DIR, "pff", "*.npz"))),
        "hackathon": [epts.HACKATHON_PATH] if os.path.isfile(epts.HACKATHON_PATH) else [],
    }
    groups["all"] = groups["dfl"] + groups["metrica"]  # hackathon only when asked for by name
    by_name = None                 # loading every file to read its name is slow: only if needed
    paths = []
    for sel in selectors or ["dfl"]:
        found = groups.get(sel)
        if found is None:
            if by_name is None:
                by_name = {Match.load(p).name: p for g in ("dfl", "metrica", "hackathon", "pff")
                           for p in groups[g]}
            found = [by_name[sel]] if sel in by_name else None
        if not found:
            sys.exit("no match data for {!r}; try `python -m bench fetch`".format(sel))
        paths += [p for p in found if p not in paths]
    return paths


def is_private(match):
    return match.name.startswith("hackathon")


def load_degraded(path, profile, seed):
    match = Match.load(path)
    if is_private(match):
        return match  # the real thing is already noisy
    if match.name.startswith("pff-") and profile == "hackathon":
        profile = "pff"  # its ball is already as missing as the real one: no extra gaps
    return degrade(match, profile, seed=seed + sum(map(ord, match.name)))


def resolve_method(name):
    from possession import METHODS
    if name in METHODS:
        return METHODS[name]
    module, _, func = name.partition(":")  # e.g. possession.hmm:predict
    import importlib
    return getattr(importlib.import_module(module), func or "predict")


# -- commands ----------------------------------------------------------------------------------

def cmd_fetch(args):
    if args.what in ("dfl", "all"):
        dfl.fetch(args.ids or None)
    if args.what in ("metrica", "all"):
        for game in (1, 2):
            epts.fetch_metrica_csv(game)
        epts.fetch_metrica()
    if args.what == "pff":                 # proprietary: download it yourself (README)
        from . import pff
        pff.fetch()
    if args.what in ("dfl-events", "all"):
        from . import dfl_events
        dfl_events.fetch(args.ids or None)


def cmd_relabel(args):
    from . import relabel
    rule = relabel.Rule()
    if args.control_s is not None or args.near_m is not None:
        rule = relabel.Rule(control_s=args.control_s if args.control_s is not None else rule.control_s,
                            near_m=args.near_m if args.near_m is not None else rule.near_m)
    relabel.write(relabel.dfl_paths(), rule, args.tag)
    if not args.tag:
        relabel.write(relabel.metrica_paths(), rule, out_d=relabel.METRICA_OUT_DIR)


def cmd_reception(args):
    from . import reception
    from .relabel import out_dir
    reception.write(reception.dfl_paths(args.source),
                    reception.OUT_DIR if args.source == "dfl" else out_dir("rx"))


def cmd_stats(args):
    """Aggregates only (safe on the hackathon data). Compare these to calibrate the degrader."""
    for path in match_paths(args.matches):
        m = load_degraded(path, args.profile, args.seed)
        g = graded_mask(m)
        labelled = g & (m.labels != 3) if m.labels is not None else g
        nb = np.isnan(m.ball[:, 0])
        vis = (~np.isnan(m.players[..., 0])).sum(1)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], nb[g].astype(np.int8), [0]))))
        gaps = (edges[1::2] - edges[::2]) / 25
        print("{} ({} profile)".format(m.name, "raw" if is_private(m) else args.profile))
        print("  frames {} graded {} | slots {} (home {} away {})".format(
            m.n_frames, g.sum(), len(m.slot_team), *np.bincount(m.slot_team, minlength=2)))
        print("  ball missing on graded {:.1%} | gaps {} median {:.1f}s p90 {:.1f}s max {:.0f}s"
              .format(nb[g].mean(), len(gaps), *np.percentile(gaps, [50, 90, 100])))
        print("  ball gap quantiles p10/25/50/75/90/99 (s): {} | <1s gaps {:.0%} | missing time in"
              " gaps >=5s {:.0%} >=30s {:.0%}".format(
                  " ".join("{:.1f}".format(q) for q in np.percentile(gaps, [10, 25, 50, 75, 90, 99])),
                  np.mean(gaps < 1), gaps[gaps >= 5].sum() / gaps.sum(),
                  gaps[gaps >= 30].sum() / gaps.sum()))
        print("  tracked players per graded frame mean {:.2f} p1/5/25/50 {} | <=18 on {:.1%} of frames"
              .format(vis[g].mean(), " ".join("{:.0f}".format(q)
                                              for q in np.percentile(vis[g], [1, 5, 25, 50])),
                      np.mean(vis[g] <= 18)))
        print("  ball missing when <=18 players tracked {:.0%}, when >=21 {:.0%}".format(
            nb[g & (vis <= 18)].mean(), nb[g & (vis >= 21)].mean()))
        tracklets = []
        for s in range(m.players.shape[1]):
            seen = ~np.isnan(m.players[g, s, 0])
            e = np.flatnonzero(np.diff(np.concatenate(([0], seen.astype(np.int8), [0]))))
            tracklets += list((e[1::2] - e[::2]) / 25)
        tracklets = np.asarray(tracklets)
        print("  tracklets {} ({:.1f}/min) length p25/50/75/90 (s): {}".format(
            len(tracklets), len(tracklets) / (g.sum() / 25 / 60),
            " ".join("{:.0f}".format(q) for q in np.percentile(tracklets, [25, 50, 75, 90]))))
        if m.labels is not None and labelled.any():
            dead = m.labels == D
            r = score(m.labels, m.labels)
            alive_min = (labelled & ~dead).sum() / 25 / 60
            print("  labelled {:.1f} min | ball missing alive {:.1%} dead {:.1%} | D share {:.1%}"
                  .format(labelled.sum() / 25 / 60, nb[labelled & ~dead].mean(),
                          nb[labelled & dead].mean(), dead[labelled].mean()))
            print("  GT spells {} ({:.2f} per alive-min) mean {:.1f}s".format(
                r["n_gt_spells"], r["n_gt_spells"] / alive_min, r["gt_mean_spell_duration_s"]))


def cmd_run(args):
    method = resolve_method(args.method)
    tag = args.tag or "{}_{}_s{}".format(args.method.replace(":", "."), args.profile, args.seed)
    out_dir = os.path.join(REPORTS_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for path in match_paths(args.matches):
        m = load_degraded(path, args.profile, args.seed)
        t0 = time.time()
        pred = method(m.without_labels())
        took = time.time() - t0
        if pred.shape != (m.n_frames,):
            sys.exit("{} returned shape {}, expected ({},)".format(args.method, pred.shape,
                                                                  m.n_frames))
        result = score(m.labels, pred, args.bootstrap)
        diag = diagnose(m, pred)
        np.save(os.path.join(out_dir, m.name + "_pred.npy"), pred.astype(np.int8))
        with open(os.path.join(out_dir, m.name + ".json"), "w") as f:
            json.dump({"result": result, "diagnostics": diag}, f, indent=1, default=float)
        rows.append((m, result, diag, took))

    print("\n{}  (profile {}, seed {}) -> {}".format(args.method, args.profile, args.seed,
                                                     os.path.relpath(out_dir)))
    print("{:<22} {:>8} {:>8} {:>9} {:>9} {:>7}".format(
        "match", "spellF1", "macroF1", "spells", "gt", "sec"))
    for m, r, _, took in rows:
        ci = "" if r["spell_bootstrap_low"] is None else "  [{:.3f}, {:.3f}]".format(
            r["spell_bootstrap_low"], r["spell_bootstrap_high"])
        print("{:<22} {:>8.4f} {:>8.4f} {:>9} {:>9} {:>7.1f}{}".format(
            m.name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
            took, ci))
    public = [(m, r, d) for m, r, d, _ in rows if not is_private(m)]
    if len(public) > 1:
        print("{:<22} {:>8.4f} {:>8.4f}".format(
            "MEAN (external)", np.mean([r["spell_f1_mean"] for _, r, _ in public]),
            np.mean([r["macro_f1"] for _, r, _ in public])))
    print_diagnostics([(r, d) for _, r, d, _ in rows], show_worst=not any(
        is_private(m) for m, *_ in rows) and args.worst)


def print_diagnostics(rows, show_worst):
    """Diagnostics summed over matches (counts) or averaged (rates)."""
    print("\nwhere the points go (summed over matches, spells at IoU 0.5)")
    buckets = {}
    for _, d in rows:
        for b, v in d["spells_by_duration"].items():
            acc = buckets.setdefault(b, [0, 0, 0, 0])
            acc[0] += v["n_gt"]
            acc[1] += round((v["recall"] or 0) * v["n_gt"])
            acc[2] += v["n_pred"]
            acc[3] += round((v["precision"] or 0) * v["n_pred"])
    print("  {:<8} {:>6} {:>7} {:>6} {:>9}".format("duration", "gt", "recall", "pred",
                                                    "precision"))
    for b, (ng, hit, npred, ok) in buckets.items():
        print("  {:<8} {:>6} {:>7} {:>6} {:>9}".format(
            b, ng, "{:.0%}".format(hit / ng) if ng else "-", npred,
            "{:.0%}".format(ok / npred) if npred else "-"))
    causes = {}
    for _, d in rows:
        for k, v in d["miss_causes"].items():
            causes[k] = causes.get(k, 0) + v
    print("  missed GT spells by cause: " + ", ".join("{} {}".format(k, v)
                                                     for k, v in causes.items()))
    lags = [d["transition_lag"] for _, d in rows if "median_frames" in d["transition_lag"]]
    if lags:
        print("  transitions found within 2s {:.0%} | lag median {:+.0f} f, p10 {:+.0f}, p90 {:+.0f}"
              .format(np.mean([l["found_within_2s"] for l in lags]),
                      np.mean([l["median_frames"] for l in lags]),
                      np.mean([l["p10_frames"] for l in lags]),
                      np.mean([l["p90_frames"] for l in lags])))
    print("  per-frame error rate by context:")
    for ctx in rows[0][1]["frame_error_by_context"]:
        vals = [d["frame_error_by_context"][ctx] for _, d in rows]
        rates = [v["error_rate"] for v in vals if v["error_rate"] is not None]
        print("    {:<20} {:>6.1%} of frames  error {:>6.1%}".format(
            ctx, np.mean([v["share_of_frames"] for v in vals]), np.mean(rates) if rates else 0))
    if show_worst:
        r, d = rows[0]
        print("\nworst 10 s stretches in the first match (frame range, clock, GT vs pred runs):")
        for w in d["worst_stretches"][:5]:
            print("  {}-{} {} wrong {}\n    gt   {}\n    pred {}".format(
                w["start"], w["stop"], w["clock"], w["wrong_frames"], w["gt"], w["pred"]))


def cmd_scenarios(args):
    """Pass rate per edge-case scenario and method; optional MP4 preview per scenario."""
    from . import scenarios
    contexts, preds = [], {m: {} for m in args.method}
    for path in match_paths(args.matches):
        clean = Match.load(path)
        if is_private(clean):
            sys.exit("scenarios run on labelled external matches only (clips are frame-level)")
        seen = degrade(clean, args.profile, seed=args.seed + sum(map(ord, clean.name)))
        contexts.append(scenarios.context(clean.name, clean, seen))
        for m in args.method:
            preds[m][clean.name] = resolve_method(m)(seen.without_labels())
    table, clips = scenarios.evaluate(contexts, preds)

    print("\n{:<22} {:>6}  ".format("scenario", "clips") + "  ".join(
        "{:>10}".format(m) for m in args.method))
    for sc in scenarios.SCENARIOS:
        row = table[sc.name]
        n = row[args.method[0]][1]
        print("{:<22} {:>6}  ".format(sc.name, n) + "  ".join(
            "{:>10}".format("{:.0%}".format(p / t) if t else "-") for p, t in row.values()))

    out_dir = os.path.join(REPORTS_DIR, "scenarios")
    os.makedirs(out_dir, exist_ok=True)
    summary = {sc.name: {"title": sc.title, "what": sc.what, "expected": sc.expected,
                         "rates": {m: [int(x) for x in v] for m, v in table[sc.name].items()}}
               for sc in scenarios.SCENARIOS}
    if args.preview:
        from .preview import render
        by_name = {c.name: c for c in contexts}
        for sc in scenarios.SCENARIOS:
            found = clips[sc.name]
            if not found:
                continue
            clip = found[len(found) // 2]  # deterministic, not cherry-picked
            ctx = by_name[clip.match]
            verdicts = {m: bool(sc.check(clip, ctx.gt, preds[m][clip.match]))
                        for m in args.method}
            path = os.path.join(out_dir, sc.name + ".mp4")
            render(clip, sc, ctx, {m: preds[m][clip.match] for m in args.method}, verdicts, path)
            summary[sc.name]["preview"] = {"file": sc.name + ".mp4", "match": clip.match,
                                           "anchor": int(clip.anchor), "verdicts": verdicts}
            print("  preview", os.path.relpath(path))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


def cmd_plot(args):
    from .plot import timeline
    out_dir = os.path.join(REPORTS_DIR, args.run)
    pred = np.load(os.path.join(out_dir, args.match + "_pred.npy"))
    path = match_paths([args.match])[0]
    with open(os.path.join(out_dir, args.match + ".json")) as f:
        profile, seed = args.profile, args.seed
        report = json.load(f)
    m = load_degraded(path, profile, seed)
    starts = [args.start] if args.start is not None else [
        w["start"] for w in report["diagnostics"]["worst_stretches"][:args.worst]]
    for start in starts:
        png = os.path.join(out_dir, "{}_{}.png".format(args.match, start))
        timeline(m, pred, start, int(args.seconds * 25), png)
        print(png)


def main():
    parser = argparse.ArgumentParser(prog="python -m bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch")
    p.add_argument("what", choices=["dfl", "metrica", "dfl-events", "pff", "all"], nargs="?",
                   default="all")
    p.add_argument("--ids", nargs="*", help="DFL match ids, e.g. DFL-MAT-J03WOH")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("relabel")
    p.add_argument("--control-s", type=float, help="rule: minimum control duration (s)")
    p.add_argument("--near-m", type=float, help="rule: control distance (m)")
    p.add_argument("--tag", help="write to external/dfl_hk_<tag> (robustness variants)")
    p.set_defaults(func=cmd_relabel)

    p = sub.add_parser("reception")
    p.add_argument("--source", choices=["dfl", "hackathon"], default="dfl",
                   help="labels to extend: DFL's (-> dfl_rx) or the hackathon-style ones (-> dfl_hk_rx)")
    p.set_defaults(func=cmd_reception)

    p = sub.add_parser("build-hackathon")
    p.set_defaults(func=lambda a: epts.hackathon())

    for name, func in (("stats", cmd_stats), ("run", cmd_run), ("plot", cmd_plot),
                       ("scenarios", cmd_scenarios)):
        p = sub.add_parser(name)
        p.add_argument("--profile", choices=sorted(PROFILES), default="hackathon")
        p.add_argument("--seed", type=int, default=0)
        p.set_defaults(func=func)
        if name in ("stats", "run", "scenarios"):
            p.add_argument("matches", nargs="*")
        if name == "scenarios":
            p.add_argument("--method", nargs="+", default=["smooth50", "hmm"])
            p.add_argument("--preview", action="store_true", help="one MP4 per scenario")
        if name == "run":
            p.add_argument("--method", required=True)
            p.add_argument("--tag")
            p.add_argument("--bootstrap", type=int, default=0)
            p.add_argument("--worst", action="store_true", help="print the worst stretches")
        if name == "plot":
            p.add_argument("--run", required=True, help="report folder name under reports/")
            p.add_argument("--match", required=True)
            p.add_argument("--start", type=int, help="first frame (default: worst stretches)")
            p.add_argument("--worst", type=int, default=3)
            p.add_argument("--seconds", type=float, default=30)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
