# Plan2Map demo site (GitHub Pages)

This `docs/` folder is a self-contained static site explaining the
**Plan2Map** benchmark and the **GeoPlanAgent** pipeline. No build step,
no dependencies — three files (`index.html`, `styles.css`, `app.js`) plus
the figures under `assets/`.

## Deploy to GitHub Pages

1. Commit + push this `docs/` folder to your `main` branch.
2. In the GitHub repo, open **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **Deploy from a branch**.
4. Under **Branch**, pick `main` and **`/docs`**. Save.
5. Wait ~1 min, then the site is live at
   `https://<your-user>.github.io/<your-repo>/`.

That's it — no Actions config, no Jekyll, nothing to configure.

## Sharing pre-publication (paper not out yet)

The page already ships with `<meta name="robots" content="noindex, nofollow, noarchive">`
in `<head>`, so search engines won't index it. Pick the option that
matches how locked-down you want the link to be:

| Option | Setup | Who can see it |
|---|---|---|
| **GitHub Pages on public repo + noindex** | already done if you followed the deploy steps above | anyone with the URL; nothing in Google. Easiest. |
| **Netlify drag-and-drop** | go to <https://app.netlify.com/drop>, drop the `docs/` folder | anyone with the URL; random `*.netlify.app` URL is unguessable in practice. No GitHub needed. |
| **Cloudflare Pages + Access** | connect a GitHub repo to CF Pages, then add an Access rule (free tier covers up to 50 users) | only colleagues whose email is on your allowlist; they get a one-time login code |
| **GitHub Pages on private repo** | requires GitHub Pro / Team ($4/mo) | only GitHub users you've granted repo access |
| **Local + ngrok tunnel** | `cd docs && python3 -m http.server 8000` in one shell, `ngrok http 8000` in another | anyone with the URL — but only while your laptop is on and ngrok is running |

For the "send the link to a few colleagues this week" use case I'd start
with **Netlify drop** — zero accounts, zero config, takes 30 seconds,
gives you `https://something-something-XXXXX.netlify.app/` to paste into
Slack / email.

When the paper is out, delete the `noindex` `<meta>` tag in `index.html`
and the page will be eligible for search indexing again.

## Things to update before publishing

`index.html` has a few placeholders you'll want to fill in:

| Where | What to replace |
|---|---|
| Top-bar **Code →** button | `href="https://github.com/"` → your repo URL |
| Cite section, **Code** link | same as above |
| Cite section, **Paper** link | `href="#"` → arXiv / OpenReview URL when ready |
| Cite section, **Dataset** link | `href="#"` → HuggingFace / Zenodo URL |
| BibTeX block | replace the placeholder `Anonymous` / venue |

A quick `grep` to find them all:

```bash
grep -nE 'href="(#"|https://github.com/")' docs/index.html
```

## Local preview

Any static-file server works. The simplest:

```bash
cd docs
python3 -m http.server 8000
# then open http://localhost:8000/
```

If you prefer Node:

```bash
npx serve docs
```

## What's in here

- `index.html` — the page itself. One-shot, no client-side framework.
- `styles.css` — design tokens + layout. Inspired by the Anthropic blog
  (cream paper, Newsreader serif headlines, Inter body) and the user's
  existing `geomap_writing_plan.html` palette.
- `app.js` — two interactive bits:
  - The **pipeline diagram** — an SVG drawn at runtime from the `NODES` +
    `ARROWS` arrays. Stage buttons (and an autoplay loop) light up the
    nodes and arrows for each stage in the `STAGES` config.
  - The **sliding-window animation** — a real walk through every window
    MINIMA-LoFTR actually evaluated on case `12:00116:ART4` (Loddon,
    Norfolk) at the cached zoom (z17) and scale (0.437×). The data lives
    in `assets/slider_data/windows.json` and is generated offline by
    `_gen_slider_data.py` (no LLM, no API credits — pure local MINIMA
    inference). The "Show correspondences" toggle draws 80 real MINIMA
    keypoint matches at the best window, with inliers in green and
    outliers faded grey.
- `assets/` — the paper figures copied from `figures/`, plus
  `slider_data/` containing the per-window inlier counts + keypoint
  correspondences captured by `_gen_slider_data.py`.

## Updating after a paper-figure refresh

Re-copy the rendered figures and (optionally) re-generate slider data:

```bash
cp figures/pipeline_*.png figures/iou_histogram.png figures/abl_cdfs.png docs/assets/

# Regenerate the slider's real-MINIMA data (offline, ~60 s on M3 Max,
# no API calls — uses MINIMA weights + OS tile cache only).
uv run docs/_gen_slider_data.py
```

Pick a different case by editing the `CASE = "12:00116:ART4"` line in
`_gen_slider_data.py`. The script reads cached metrics from
`results/<benchmark>/<model>/<case>/{metrics,pdf_info,tile_info}.json`
to know the winning anchor + scale, then re-runs the sliding-window
matcher with those settings — so the per-window inlier counts the demo
shows are the ones the pipeline actually saw on that case.

## License

Site content tracks the parent repo's licence. The OS basemap snippets
used in `assets/` derive from Ordnance Survey OpenData (Crown copyright,
OGL v3) and must keep the OS attribution visible — currently surfaced in
the page footer.
