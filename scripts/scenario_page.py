"""Build reports/scenarios/index.html (+ poster JPGs) from `python -m bench scenarios --preview`.

    uv run python -m bench scenarios --method smooth50 hmm --preview
    uv run python scripts/scenario_page.py

External (DFL) clips only. The page is static: one card per scenario with its preview video.
"""

import html
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "scenarios")

GROUPS = [
    ("Holding the ball", "The team in possession must keep it, even when the evidence wobbles.",
     ["pass_past_opponents", "ball_gap_hold", "slow_play"]),
    ("Hand-overs", "A real change of possession must be found, on time, and only once.",
     ["interception", "short_spell", "change_in_ball_gap", "blackout_change", "duel_flurry"]),
    ("Stoppages and restarts", "Dead ball ends a possession; the restart starts the next one.",
     ["restart_same", "restart_other", "long_stoppage", "dead_ball_visible", "kickoff"]),
]

STYLE = """
:root {
  /* Layout: a match report. Summary table first, then one card per situation, grouped by phase. */
  --bg: #f3f5f1; --surface: #ffffff; --ink: #18211b; --muted: #5d6b61; --line: #d5ddd4;
  --pitch: #2f6b45; --home: #1f5fbf; --away: #d1495b; --dead: #9aa0a6;
  --pass: #2e7d32; --fail: #b3261e;
  --display: "Barlow Condensed", "Arial Narrow", sans-serif;
  --body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121814; --surface: #1a221d; --ink: #e4ebe5; --muted: #9aab9f; --line: #2c3a31;
  --pitch: #6fbf8a; --home: #6f9ff0; --away: #ef7a8b; --dead: #8a9199;
  --pass: #7fd08a; --fail: #f28b82; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #121814; --surface: #1a221d; --ink: #e4ebe5; --muted: #9aab9f; --line: #2c3a31;
  --pitch: #6fbf8a; --home: #6f9ff0; --away: #ef7a8b; --dead: #8a9199;
  --pass: #7fd08a; --fail: #f28b82; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body);
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 32px 64px; }
main { max-width: 1120px; margin: 0 auto; display: grid; gap: 40px; }
h1, h2, h3 { font-family: var(--display); font-weight: 600; text-wrap: balance; margin: 0; }
h1 { font-size: 2.6rem; line-height: 1.05; letter-spacing: 0.01em; }
h2 { font-size: 1.7rem; }
h3 { font-size: 1.3rem; }
p { margin: 0; max-width: 68ch; }
.lede { display: grid; gap: 12px; }
.eyebrow { font: 600 0.75rem var(--mono); letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--pitch); }
.muted { color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 8px 20px; font-size: 0.85rem; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 14px; height: 10px; border-radius: 2px; display: inline-block; }
.sw.h { background: var(--home); } .sw.a { background: var(--away); } .sw.d { background: var(--dead); }
.ring { width: 12px; height: 12px; border: 2px solid var(--dead); border-radius: 50%; display: inline-block; }
.dot { width: 8px; height: 8px; background: var(--ink); border-radius: 50%; display: inline-block; }
.table-wrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--line);
  border-radius: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; min-width: 560px; }
th, td { text-align: left; padding: 9px 14px; border-bottom: 1px solid var(--line); }
th { font: 600 0.72rem var(--mono); letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); }
td.num { font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; }
tr.group td { font-family: var(--display); font-size: 1.05rem; color: var(--pitch);
  background: var(--bg); }
tbody tr:last-child td { border-bottom: 0; }
td a { color: inherit; text-decoration-color: var(--line); text-underline-offset: 3px; }
.rate { display: grid; grid-template-columns: 1fr 3.2em; align-items: center; gap: 8px; }
.bar { height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; }
.bar i { display: block; height: 100%; background: var(--pitch); }
.section { display: grid; gap: 16px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 500px), 1fr));
  gap: 20px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 6px;
  overflow: hidden; display: grid; grid-template-rows: auto 1fr; scroll-margin-top: 16px; }
video { width: 100%; height: auto; display: block; background: #fff; aspect-ratio: 64 / 56;
  max-width: 100%; }
.body { padding: 16px 18px 18px; display: grid; gap: 10px; align-content: start; min-width: 0; }
.kv { display: grid; grid-template-columns: 6.5em 1fr; gap: 4px 12px; font-size: 0.9rem; }
.kv dt { font: 600 0.7rem var(--mono); letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); padding-top: 3px; }
.kv dd { margin: 0; min-width: 0; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; font: 0.8rem var(--mono); }
.chip { border: 1px solid var(--line); border-radius: 999px; padding: 2px 10px;
  font-variant-numeric: tabular-nums; }
.chip.pass { color: var(--pass); border-color: currentColor; }
.chip.fail { color: var(--fail); border-color: currentColor; }
footer { font-size: 0.85rem; color: var(--muted); display: grid; gap: 6px; }
code { font: 0.85em var(--mono); }
a:focus-visible { outline: 2px solid var(--pitch); outline-offset: 2px; }
@media (max-width: 520px) { h1 { font-size: 2rem; } .kv { grid-template-columns: 1fr; } }
"""


def rate_cell(passed, total):
    pct = 100 * passed / total if total else 0
    return ('<td><div class="rate"><div class="bar"><i style="width:{:.0f}%"></i></div>'
            '<span class="num">{:.0f}%</span></div></td>').format(pct, pct)


def main():
    with open(os.path.join(OUT, "summary.json")) as f:
        summary = json.load(f)
    methods = list(next(iter(summary.values()))["rates"])

    for name, s in summary.items():  # poster frame at the clip's key moment (4 s in)
        if "preview" in s:
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "3.5", "-i",
                            os.path.join(OUT, name + ".mp4"), "-frames:v", "1", "-q:v", "4",
                            os.path.join(OUT, name + ".jpg")], check=True)

    rows = []
    for title, _, names in GROUPS:
        rows.append('<tr class="group"><td colspan="{}">{}</td></tr>'.format(
            2 + len(methods), html.escape(title)))
        for n in names:
            s = summary[n]
            total = s["rates"][methods[0]][1]
            rows.append('<tr><td><a href="#{}">{}</a></td><td class="num">{}</td>{}</tr>'.format(
                n, html.escape(s["title"]), total,
                "".join(rate_cell(*s["rates"][m]) for m in methods)))

    sections = []
    for title, intro, names in GROUPS:
        cards = []
        for n in names:
            s = summary[n]
            pv = s.get("preview")
            video = ('<video src="{0}.mp4" poster="{0}.jpg" controls muted loop playsinline '
                     'preload="none" aria-label="Replay of one {1} clip"></video>'.format(
                         n, html.escape(s["title"]))) if pv else ""
            chips = "".join(
                '<span class="chip">{}: {}/{} clips</span>'.format(m, *s["rates"][m])
                for m in methods)
            verdicts = "".join(
                '<span class="chip {}">{} on this clip: {}</span>'.format(
                    "pass" if v else "fail", m, "pass" if v else "fail")
                for m, v in (pv or {}).get("verdicts", {}).items())
            where = ('<dt>Clip</dt><dd><code>{}</code>, frame {}</dd>'.format(
                html.escape(pv["match"]), pv["anchor"])) if pv else ""
            cards.append(
                '<article class="card" id="{}">{}<div class="body"><h3>{}</h3><dl class="kv">'
                '<dt>Situation</dt><dd>{}</dd><dt>Expected</dt><dd>{}</dd>{}</dl>'
                '<div class="chips">{}</div><div class="chips">{}</div></div></article>'.format(
                    n, video, html.escape(s["title"]), html.escape(s["what"]),
                    html.escape(s["expected"]), where, chips, verdicts))
        sections.append('<section class="section"><div class="lede"><h2>{}</h2>'
                        '<p class="muted">{}</p></div><div class="cards">{}</div></section>'
                        .format(html.escape(title), html.escape(intro), "".join(cards)))

    page = """<title>Possession Edge Cases</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{style}</style>
<main>
<header class="lede">
  <span class="eyebrow">AndaAnda · More than a Hack 2027 · test bench</span>
  <h1>Possession edge cases</h1>
  <p>Thirteen situations that break possession labelling, mined automatically from 7 DFL
  matches (Bassek et al. 2025, CC-BY 4.0) with tracking degraded to look like the hackathon
  data. Each has one pass/fail rule. Methods compared: {methods}.</p>
  <p class="muted">Each replay shows one clip, picked as the middle one of its scenario, not a
  hand-picked example. The pitch shows the noisy tracking a method actually sees; the strips below
  show the true labels and each method's labels, with a cursor.</p>
  <div class="legend">
    <span><i class="sw h"></i>home in possession</span>
    <span><i class="sw a"></i>away in possession</span>
    <span><i class="sw d"></i>dead ball</span>
    <span><i class="dot"></i>ball as tracked</span>
    <span><i class="ring"></i>true ball position</span>
  </div>
</header>
<section class="section">
  <h2>Pass rate by situation</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Situation</th><th>Clips</th>{heads}</tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
</section>
{sections}
<footer>
  <p>Generated by <code>python -m bench scenarios --preview</code> and
  <code>scripts/scenario_page.py</code> in the AndaAnda repository.</p>
  <p>The replays use public DFL data only. No hackathon match data appears on this page.</p>
</footer>
</main>""".format(style=STYLE, methods=", ".join(methods),
                  heads="".join("<th>{}</th>".format(html.escape(m)) for m in methods),
                  rows="".join(rows), sections="".join(sections))
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(page)
    print(os.path.join(OUT, "index.html"))


if __name__ == "__main__":
    sys.exit(main())
