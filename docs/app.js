/* GeoPlanAgent demo site — pipeline diagram + sliding-window animation */

(() => {
  'use strict';

  // ────────────────────────────────────────────────────────────────────────
  //  PIPELINE DIAGRAM
  //
  //  Diagram is hand-laid: each arrow has an explicit orthogonal/curved path
  //  and an explicit label position so nothing overlaps. viewBox 1140 × 520.
  // ────────────────────────────────────────────────────────────────────────

  const NODES = {
    pdf:    { x:  20, y: 140, w:  90, h: 60, label: 'Planning PDF',     sub: 'raw bytes',                color: 'slate', ioPill: true },
    reader: { x: 130, y: 140, w: 110, h: 60, label: 'Reader',           sub: 'VLM → PDFInfo',            color: 'blue'  },
    worker: { x: 330, y: 130, w: 200, h: 100,label: 'Worker loop',      sub: 'tool-calling · multi-area',color: 'amber' },
    output: { x: 950, y: 140, w: 170, h: 60, label: 'WGS84 GeoJSON',    sub: 'Polygon · MultiPolygon',   color: 'slate', ioPill: true },
    locate: { x: 330, y: 360, w: 200, h: 70, label: 'Locate sub-agent', sub: 'OS Open Names',            color: 'green' },
    slide:  { x: 620, y:  60, w: 180, h: 60, label: 'Sliding window',   sub: 'MINIMA-LoFTR',             color: 'amber' },
    sam:    { x: 620, y: 160, w: 180, h: 60, label: 'SAM 3 + LoRA',     sub: 'semantic mask',            color: 'violet'},
    affine: { x: 620, y: 260, w: 180, h: 60, label: 'Affine projection',sub: '4-DOF + tile inverse',     color: 'slate' },
  };

  // Explicit per-arrow geometry. Each entry:
  //   path        = SVG path data
  //   labelX/Y    = label position
  //   labelAlign  = text-anchor (start | middle | end)
  //   dashed      = optional dashed style
  const ARROWS = [
    { id: 'a-pdf-reader',     path: 'M 110 170 L 130 170' },
    { id: 'a-reader-worker',  path: 'M 240 170 L 330 170',
      label: 'PDFInfo', labelX: 285, labelY: 160, labelAlign: 'middle' },

    // Worker ↔ Locate: two vertical arrows side-by-side under the worker.
    // Labels are placed with ~16 px clearance from each arrow so the rect
    // backgrounds don't visually touch the lines.
    { id: 'a-worker-locate',  path: 'M 370 230 L 370 360',
      label: 'propose_centers',           labelX: 354, labelY: 300, labelAlign: 'end' },
    { id: 'a-locate-worker',  path: 'M 490 360 L 490 230',
      label: 'LocatePick',                labelX: 506, labelY: 300, labelAlign: 'start' },

    // Worker → Sliding window (top-right corridor)
    { id: 'a-worker-slide',   path: 'M 530 160 L 580 160 L 580 90 L 620 90',
      label: 'match_at(page=N)', labelX: 635, labelY: 47, labelAlign: 'start' },

    // The match_at internal chain: slide → sam → affine
    { id: 'a-slide-sam',      path: 'M 710 120 L 710 160' },
    { id: 'a-sam-affine',     path: 'M 710 220 L 710 260' },

    // Affine → back into worker (one candidate per match_at; worker calls
    // commit_match to slot it into this area_group's polygon)
    { id: 'a-affine-worker',  path: 'M 620 290 L 580 290 L 580 215 L 530 215',
      label: 'candidate → commit_match', labelX: 575, labelY: 332, labelAlign: 'end' },

    // Worker → output: only fires when the worker submits BoundaryOutcome
    // (after every area_group has been committed). For most docs this is
    // one trip round the loop; for multi-area docs it's one per group.
    { id: 'a-worker-output',  path: 'M 530 150 L 580 150 L 580 30 L 930 30 L 930 170 L 950 170',
      label: 'submit BoundaryOutcome', labelX: 740, labelY: 22, labelAlign: 'middle' },
  ];

  // Stage configuration — for each stage, which nodes are active and which arrows
  const STAGES = [
    { // 0 — Reader
      title: 'Reader',
      detail: 'A single pydantic-ai call on the raw PDF binary returns a typed PDFInfo: postcodes, OS grid refs, the site address, house-number / road pairs, map-page metadata, and the is_district_wide flag.',
      kv: [
        ['input',  'application/pdf'],
        ['output', 'PDFInfo (typed)'],
        ['model',  'Gemini 3 Flash @ temp=0'],
        ['budget', '1 request'],
      ],
      nodes: ['reader'],
      arrows: ['a-pdf-reader', 'a-reader-worker'],
    },
    { // 1 — Locate
      title: 'Locate sub-agent',
      detail: 'Invoked through propose_centers. Sees PDFInfo + the rendered map image. Issues 2–4 OS Open Names queries (free UK gazetteer), clusters the candidates, picks one (lat, lon, σ, confidence). Re-invoked with a feedback string when MINIMA reports a weak match.',
      kv: [
        ['tool',     'place(q, la?)'],
        ['output',   'LocatePick(lat, lon, σ, confidence)'],
        ['σ tight',  '≈200 m (consensus)'],
        ['σ loose',  '800–1500 m (single ambiguous pick)'],
      ],
      nodes: ['locate'],
      arrows: ['a-worker-locate', 'a-locate-worker'],
    },
    { // 2 — Sliding window
      title: 'Sliding-window matcher',
      detail: 'For each (center, zoom, rotation), resize the planning map to match tile pixel scale, then sweep ~100 windows across the OS Open Zoomstack canvas. MINIMA-LoFTR computes cross-modal matches; RANSAC fits a 4-DOF similarity; top-K candidates are reranked by quadrant coverage and road-name agreement.',
      kv: [
        ['matcher',     'MINIMA-LoFTR (LoFTR trained on synthetic cross-modal pairs)'],
        ['basemap',     'OS Open Zoomstack (OGL v3)'],
        ['windows',     '~100 per (center, zoom, rotation)'],
        ['transform',   '4-DOF similarity (rotation + uniform scale + translation)'],
      ],
      nodes: ['slide'],
      arrows: ['a-worker-slide'],
    },
    { // 3 — SAM 3
      title: 'SAM 3 + LoRA boundary segmentation',
      detail: 'SAM 3 fine-tuned with LoRA on the phrase “planning boundary”. 5-fold cross-validation: each evaluation case is segmented by the adapter from the fold that didn’t see it. Produces a binary mask in planning-map pixels.',
      kv: [
        ['backbone',     'facebook/sam3'],
        ['adapter',      'LoRA (r=16), per-fold'],
        ['query',        '"planning boundary"'],
        ['pixel IoU',    '0.912 mean (5-fold OOF, 211 cases)'],
      ],
      nodes: ['sam'],
      arrows: ['a-slide-sam'],
    },
    { // 4 — Affine projection + commit_match
      title: 'Affine projection → commit_match',
      detail: 'Trace SAM 3 mask contours → transform vertices through the RANSAC affine → convert tile pixels to (lat, lon) via the web-mercator inverse. The Worker calls commit_match to slot the polygon into this area_group; multi-area docs loop the whole sequence per group and shapely-union the polygons.',
      kv: [
        ['input',         'SAM 3 mask + RANSAC H (2×3)'],
        ['transform',     'H ∘ tile_inverse_webmercator(zoom)'],
        ['commit_match',  'replaces this area_group\'s slot in the running union'],
        ['multi-area',    'loop per area_group · shapely-union to final geometry'],
      ],
      // Affine produces the candidate; Worker is what actually calls
      // commit_match. Both are genuinely active for this step.
      nodes: ['affine', 'worker'],
      arrows: ['a-sam-affine', 'a-affine-worker'],
    },
    { // 5 — Submit
      title: 'submit BoundaryOutcome → GeoJSON',
      detail: 'When every area_group has been committed, the Worker submits BoundaryOutcome(status="accepted"). An output validator re-reads tool-call state and rejects any mismatched flags. The pipeline always emits a polygon — refusing a case is not supported. Documents flagged is_district_wide skip everything above and submit BoundaryOutcome(status="district_lookup") after a successful lookup_district call.',
      kv: [
        ['status',       'accepted | district_lookup'],
        ['validator',    'auto-corrects rotation_checked + final_n_inliers from state'],
        ['output',       'Feature(Polygon | MultiPolygon, WGS84)'],
        ['district',     'is_district_wide → OS BoundaryLine polygon, bypass MINIMA + SAM'],
      ],
      nodes: ['output'],
      arrows: ['a-worker-output'],
    },
  ];

  function buildDiagramSVG() {
    const W = 1140, H = 480;
    const parts = [];

    parts.push(`<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="GeoPlanAgent pipeline diagram">`);

    // Defs: arrow heads (idle + warm-highlighted)
    parts.push(`
      <defs>
        <marker id="arrow-head" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="9" markerHeight="9" orient="auto">
          <path d="M0 0 L12 6 L0 12 Z" class="arrow-head" />
        </marker>
        <marker id="arrow-head-warm" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="9" markerHeight="9" orient="auto">
          <path d="M0 0 L12 6 L0 12 Z" class="arrow-head is-active" />
        </marker>
      </defs>
    `);

    // Arrows (drawn first so nodes layer on top)
    for (const ar of ARROWS) {
      const cls = `diagram-arrow${ar.dashed ? ' dashed' : ''}`;
      parts.push(`
        <path id="${ar.id}" class="${cls}" d="${ar.path}" marker-end="url(#arrow-head)" />
      `);
      if (ar.label) {
        const labW = ar.label.length * 6.4 + 16;
        const labX0 = ar.labelAlign === 'end'    ? ar.labelX - labW + 6
                    : ar.labelAlign === 'middle' ? ar.labelX - labW / 2
                                                 : ar.labelX - 6;
        parts.push(`
          <g class="arrow-label">
            <rect x="${labX0}" y="${ar.labelY - 11}" width="${labW}" height="15"
                  rx="7" ry="7" fill="rgba(255, 252, 246, 0.94)" />
            <text x="${ar.labelX}" y="${ar.labelY}" text-anchor="${ar.labelAlign || 'middle'}"
                  fill="#5e6b76" font-size="10.5" font-weight="600"
                  font-family="JetBrains Mono, monospace">${ar.label}</text>
          </g>
        `);
      }
    }

    // Nodes
    for (const [id, n] of Object.entries(NODES)) {
      const rx = n.ioPill ? 30 : 12;
      const grpClass = `diagram-node color-${n.color}${n.ioPill ? ' io-pill' : ''}`;
      parts.push(`
        <g id="node-${id}" class="${grpClass}">
          <rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="${rx}" ry="${rx}" />
          <text x="${n.x + n.w / 2}" y="${n.y + n.h / 2 - (n.sub ? 4 : -4)}" text-anchor="middle">${n.label}</text>
          ${n.sub ? `<text class="node-sub" x="${n.x + n.w / 2}" y="${n.y + n.h / 2 + 12}" text-anchor="middle">${n.sub}</text>` : ''}
        </g>
      `);
    }

    parts.push('</svg>');
    return parts.join('');
  }

  function applyStage(stageIdx) {
    const stage = STAGES[stageIdx];
    if (!stage) return;
    const activeNodes = new Set(stage.nodes);
    const activeArrows = new Set(stage.arrows);

    document.querySelectorAll('.diagram-node').forEach(el => {
      const id = el.id.replace(/^node-/, '');
      el.classList.toggle('is-active', activeNodes.has(id));
      el.classList.toggle('dim', !activeNodes.has(id));
    });
    document.querySelectorAll('.diagram-arrow').forEach(el => {
      const isActive = activeArrows.has(el.id);
      el.classList.toggle('is-active', isActive);
      el.setAttribute('marker-end', isActive ? 'url(#arrow-head-warm)' : 'url(#arrow-head)');
    });
    document.querySelectorAll('.stage-btn[data-stage]').forEach(b => {
      b.classList.toggle('is-active', Number(b.dataset.stage) === stageIdx);
    });

    const det = document.getElementById('stageDetail');
    if (det) {
      const kvHtml = stage.kv.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
      det.innerHTML = `
        <h4>${stage.title}</h4>
        <p style="margin: 8px 0 0; color: var(--ink-soft); font-size: 15px; line-height: 1.6;">${stage.detail}</p>
        <dl class="stage-kv">${kvHtml}</dl>
      `;
    }
  }

  function initPipelineDiagram() {
    const host = document.getElementById('pipelineDiagram');
    if (!host) return;
    host.innerHTML = buildDiagramSVG();
    document.querySelectorAll('.stage-btn[data-stage]').forEach(btn => {
      btn.addEventListener('click', () => {
        stopAutoplay();
        applyStage(Number(btn.dataset.stage));
      });
    });
    applyStage(0);

    const autoBtn = document.getElementById('autoplayBtn');
    if (autoBtn) {
      autoBtn.addEventListener('click', () => {
        if (autoplayTimer) stopAutoplay();
        else startAutoplay();
      });
    }
  }

  let autoplayTimer = null;
  function startAutoplay() {
    const autoBtn = document.getElementById('autoplayBtn');
    if (autoBtn) autoBtn.textContent = '⏸ Pause';
    let i = 0;
    applyStage(i);
    autoplayTimer = setInterval(() => {
      i = (i + 1) % STAGES.length;
      applyStage(i);
    }, 2600);
  }
  function stopAutoplay() {
    if (autoplayTimer) { clearInterval(autoplayTimer); autoplayTimer = null; }
    const autoBtn = document.getElementById('autoplayBtn');
    if (autoBtn) autoBtn.textContent = '▶ Autoplay';
  }


  // ────────────────────────────────────────────────────────────────────────
  //  SLIDING-WINDOW ANIMATION
  // ────────────────────────────────────────────────────────────────────────
  //
  // Grounded in REAL MINIMA data. windows.json is generated offline by
  // docs/_gen_slider_data.py — every n_inliers is the actual RANSAC count
  // from re-running sliding_window_position on case 12:00116:ART4 (Loddon,
  // Norfolk) at the cached scale 0.437× and zoom 17.

  const SLIDE = {
    // Geometry (filled from windows.json)
    canvasW: 0, canvasH: 0,
    mapW: 0, mapH: 0,
    windows: [],            // real list from JSON
    bestIdx: 0,
    bestWindow: null,       // {x, y, w, h, n_inliers, mkpts0, mkpts1, mconf, inlier_mask}
    zoom: 17,
    // DOM
    mapOverlay: null,       // SVG over left panel
    tilesOverlay: null,     // SVG over right panel
    corrOverlay: null,      // SVG spanning both panels
    mapImg: null,
    tilesImg: null,
    // Animation state
    cursor: 0,
    playing: false,
    timer: null,
    bestSoFar: 0,
    corrShown: false,
  };

  async function loadSliderData() {
    try {
      const r = await fetch('assets/slider_data/windows.json', { cache: 'no-store' });
      if (!r.ok) throw new Error(`status ${r.status}`);
      return await r.json();
    } catch (e) {
      console.warn('slider data fetch failed:', e);
      return null;
    }
  }

  function tierFor(n) {
    if (n >= 100) return 'strong';
    if (n >= 50)  return 'ok';
    if (n >= 25)  return 'weak';
    return 'toow';
  }
  function tierColor(n) {
    switch (tierFor(n)) {
      case 'strong': return '#236a4a';
      case 'ok':     return '#83560e';
      case 'weak':   return '#b8860b';
      case 'toow':   return '#aa3030';
    }
  }

  function setSvgViewBox(svg, w, h) {
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  }

  function renderSlideOverlay() {
    if (!SLIDE.tilesOverlay || !SLIDE.windows.length) return;
    const p = SLIDE.windows[SLIDE.cursor];
    if (!p) return;
    const svgNS = 'http://www.w3.org/2000/svg';
    const ov = SLIDE.tilesOverlay;
    const isBest = SLIDE.cursor === SLIDE.bestIdx;

    while (ov.firstChild) ov.removeChild(ov.firstChild);

    // Trail of visited windows (subtle outlines)
    for (let i = 0; i < SLIDE.cursor; i++) {
      const prev = SLIDE.windows[i];
      const trail = document.createElementNS(svgNS, 'rect');
      trail.setAttribute('x', prev.x);
      trail.setAttribute('y', prev.y);
      trail.setAttribute('width', prev.w);
      trail.setAttribute('height', prev.h);
      trail.setAttribute('fill', 'none');
      trail.setAttribute('stroke', tierColor(prev.n_inliers));
      trail.setAttribute('stroke-opacity', '0.10');
      trail.setAttribute('stroke-width', '2');
      trail.setAttribute('rx', '4');
      ov.appendChild(trail);
    }

    // Active window
    const stroke = tierColor(p.n_inliers);
    const rect = document.createElementNS(svgNS, 'rect');
    rect.setAttribute('x', p.x);
    rect.setAttribute('y', p.y);
    rect.setAttribute('width', p.w);
    rect.setAttribute('height', p.h);
    rect.setAttribute('fill', 'rgba(0,0,0,0)');
    rect.setAttribute('stroke', stroke);
    rect.setAttribute('stroke-width', isBest ? '16' : '10');
    rect.setAttribute('rx', '6');
    if (isBest) {
      rect.setAttribute('filter', 'drop-shadow(0 0 24px rgba(35, 106, 74, 0.6))');
    }
    ov.appendChild(rect);

    // Inlier badge
    const badgeW = 220, badgeH = 60;
    const badgeBg = document.createElementNS(svgNS, 'rect');
    badgeBg.setAttribute('x', p.x);
    badgeBg.setAttribute('y', Math.max(0, p.y - badgeH - 8));
    badgeBg.setAttribute('width', badgeW);
    badgeBg.setAttribute('height', badgeH);
    badgeBg.setAttribute('rx', '10');
    badgeBg.setAttribute('fill', stroke);
    ov.appendChild(badgeBg);

    const badgeText = document.createElementNS(svgNS, 'text');
    badgeText.setAttribute('x', p.x + 14);
    badgeText.setAttribute('y', Math.max(35, p.y - badgeH + 38));
    badgeText.setAttribute('fill', '#fff');
    badgeText.setAttribute('font-family', 'JetBrains Mono, ui-monospace, monospace');
    badgeText.setAttribute('font-size', '32');
    badgeText.setAttribute('font-weight', '700');
    badgeText.textContent = `inliers ${p.n_inliers}`;
    ov.appendChild(badgeText);
  }

  function updateReadouts() {
    const p = SLIDE.windows[SLIDE.cursor];
    if (!p) return;
    SLIDE.bestSoFar = Math.max(SLIDE.bestSoFar, p.n_inliers);

    const win    = document.getElementById('rdWindow');
    const zoom   = document.getElementById('rdZoom');
    const inl    = document.getElementById('rdInliers');
    const scale  = document.getElementById('rdScale');
    const best   = document.getElementById('rdBest');
    if (win)   win.textContent   = `${SLIDE.cursor + 1} / ${SLIDE.windows.length}`;
    if (zoom)  zoom.textContent  = `z${SLIDE.zoom}`;
    if (inl) {
      inl.textContent = String(p.n_inliers);
      inl.classList.remove('toow', 'weak', 'ok', 'strong');
      inl.classList.add(tierFor(p.n_inliers));
    }
    if (scale) scale.textContent = p.avg_scale != null ? `×${p.avg_scale.toFixed(2)}` : '—';
    if (best)  best.textContent  = String(SLIDE.bestSoFar);
  }

  function slideStep() {
    if (SLIDE.cursor < SLIDE.windows.length - 1) {
      SLIDE.cursor++;
    } else {
      SLIDE.cursor = SLIDE.bestIdx;
    }
    renderSlideOverlay();
    updateReadouts();
    if (SLIDE.cursor === SLIDE.bestIdx && SLIDE.corrShown) {
      renderCorrespondences();
    }
  }

  function slidePlay() {
    if (SLIDE.playing) return;
    SLIDE.playing = true;
    const playBtn = document.getElementById('slidePlay');
    if (playBtn) playBtn.textContent = '⏸ Pause';
    SLIDE.timer = setInterval(() => {
      if (SLIDE.cursor >= SLIDE.windows.length - 1) {
        SLIDE.cursor = SLIDE.bestIdx;
        renderSlideOverlay();
        updateReadouts();
        if (SLIDE.corrShown) renderCorrespondences();
        slidePause();
        return;
      }
      slideStep();
    }, 110);
  }

  function slidePause() {
    SLIDE.playing = false;
    if (SLIDE.timer) { clearInterval(SLIDE.timer); SLIDE.timer = null; }
    const playBtn = document.getElementById('slidePlay');
    if (playBtn) playBtn.textContent = '▶ Play';
  }

  function slideReset() {
    slidePause();
    SLIDE.cursor = 0;
    SLIDE.bestSoFar = 0;
    clearCorrespondences();
    renderSlideOverlay();
    updateReadouts();
  }

  // ── Correspondence visualisation ────────────────────────────────────────

  function clearCorrespondences() {
    [SLIDE.mapOverlay, SLIDE.corrOverlay].forEach(o => {
      if (!o) return;
      // Leave only the tiles-overlay alone (it carries the window box).
      // Clear only the corr/map overlays.
      while (o.firstChild) o.removeChild(o.firstChild);
    });
  }

  /** Map a keypoint (image-pixel) to viewport (CSS-pixel) coords on a panel image. */
  function imgPxToViewport(img, ix, iy) {
    const r = img.getBoundingClientRect();
    const sx = r.width  / img.naturalWidth;
    const sy = r.height / img.naturalHeight;
    return { x: r.left + ix * sx, y: r.top + iy * sy };
  }

  function renderCorrespondences() {
    if (!SLIDE.bestWindow || !SLIDE.corrOverlay) return;
    if (SLIDE.cursor !== SLIDE.bestIdx) return;
    const bw = SLIDE.bestWindow;
    const mkpts0 = bw.mkpts0 || [];
    const mkpts1 = bw.mkpts1 || [];
    const inl    = bw.inlier_mask || [];
    if (mkpts0.length === 0) return;

    const svgNS = 'http://www.w3.org/2000/svg';
    const stage = document.querySelector('.slider-stage');
    const sr = stage.getBoundingClientRect();

    // Match the corr overlay to the stage geometry
    const corr = SLIDE.corrOverlay;
    corr.setAttribute('viewBox', `0 0 ${sr.width} ${sr.height}`);
    corr.setAttribute('width',  sr.width);
    corr.setAttribute('height', sr.height);

    while (corr.firstChild) corr.removeChild(corr.firstChild);
    // Also clear the per-panel dot overlays
    const mapOv  = SLIDE.mapOverlay;
    const tilesOv = SLIDE.tilesOverlay;
    if (mapOv) while (mapOv.firstChild) mapOv.removeChild(mapOv.firstChild);
    // Re-render the active best window box on tiles overlay (since we cleared it)
    renderSlideOverlay();

    // Draw inliers first (bright) on top of outliers
    const order = [];
    for (let i = 0; i < mkpts0.length; i++) if (inl[i] === 0) order.push(i);
    for (let i = 0; i < mkpts0.length; i++) if (inl[i] === 1) order.push(i);

    const mapImg = SLIDE.mapImg;
    const tilesImg = SLIDE.tilesImg;
    if (!mapImg || !tilesImg) return;

    for (const i of order) {
      const isInlier = inl[i] === 1;
      const opacity = isInlier ? 0.85 : 0.18;
      const colour  = isInlier ? '#236a4a' : '#5e6b76';
      const radius  = isInlier ? 4 : 2.5;

      // Left endpoint: keypoint in planning map
      const [mx, my] = mkpts0[i];
      const L = imgPxToViewport(mapImg, mx, my);

      // Right endpoint: keypoint in tile canvas = window-relative + window offset
      const [tx, ty] = mkpts1[i];
      const T = imgPxToViewport(tilesImg, tx + bw.x, ty + bw.y);

      // Stage-relative coords
      const Lx = L.x - sr.left, Ly = L.y - sr.top;
      const Tx = T.x - sr.left, Ty = T.y - sr.top;

      // Connecting line on the top-level overlay
      const line = document.createElementNS(svgNS, 'line');
      line.setAttribute('x1', Lx); line.setAttribute('y1', Ly);
      line.setAttribute('x2', Tx); line.setAttribute('y2', Ty);
      line.setAttribute('stroke', colour);
      line.setAttribute('stroke-width', isInlier ? '1.2' : '0.7');
      line.setAttribute('stroke-opacity', String(opacity));
      corr.appendChild(line);

      // Dots on each end
      [[Lx, Ly], [Tx, Ty]].forEach(([cx, cy]) => {
        const c = document.createElementNS(svgNS, 'circle');
        c.setAttribute('cx', cx); c.setAttribute('cy', cy);
        c.setAttribute('r', radius);
        c.setAttribute('fill', colour);
        c.setAttribute('fill-opacity', String(opacity));
        corr.appendChild(c);
      });
    }
  }

  function toggleCorrespondences() {
    SLIDE.corrShown = !SLIDE.corrShown;
    const btn = document.getElementById('slideCorrToggle');
    if (btn) btn.textContent = SLIDE.corrShown ? 'Hide correspondences' : 'Show correspondences';

    if (SLIDE.corrShown) {
      // Auto-jump to best window so user sees correspondences immediately
      slidePause();
      SLIDE.cursor = SLIDE.bestIdx;
      SLIDE.bestSoFar = Math.max(SLIDE.bestSoFar, SLIDE.bestWindow.n_inliers);
      renderSlideOverlay();
      updateReadouts();
      renderCorrespondences();
    } else {
      const corr = SLIDE.corrOverlay;
      if (corr) while (corr.firstChild) corr.removeChild(corr.firstChild);
    }
  }

  async function initSlider() {
    SLIDE.mapOverlay   = document.getElementById('slideMapOverlay');
    SLIDE.tilesOverlay = document.getElementById('slideTilesOverlay');
    SLIDE.corrOverlay  = document.getElementById('slideCorrOverlay');
    SLIDE.mapImg       = document.getElementById('slideMapImg');
    SLIDE.tilesImg     = document.getElementById('slideTilesImg');
    if (!SLIDE.tilesOverlay) return;

    // Fetch real data
    const data = await loadSliderData();
    if (!data) {
      console.warn('slider running with no data');
      return;
    }

    SLIDE.canvasW   = data.canvas_w;
    SLIDE.canvasH   = data.canvas_h;
    SLIDE.mapW      = data.map_w;
    SLIDE.mapH      = data.map_h;
    SLIDE.windows   = data.windows;
    SLIDE.zoom      = data.zoom;
    SLIDE.bestWindow = data.best_window;
    SLIDE.bestIdx   = data.windows.reduce(
      (best, w, i) => (w.n_inliers > data.windows[best].n_inliers ? i : best), 0,
    );

    // Set viewBoxes so the SVG overlays use image-pixel coords
    setSvgViewBox(SLIDE.tilesOverlay, SLIDE.canvasW, SLIDE.canvasH);
    setSvgViewBox(SLIDE.mapOverlay,   SLIDE.mapW,    SLIDE.mapH);

    // Wire buttons
    document.getElementById('slidePlay')?.addEventListener('click',
      () => SLIDE.playing ? slidePause() : slidePlay());
    document.getElementById('slideStep')?.addEventListener('click',
      () => { slidePause(); slideStep(); });
    document.getElementById('slideReset')?.addEventListener('click',
      () => slideReset());
    document.getElementById('slideCorrToggle')?.addEventListener('click',
      () => toggleCorrespondences());

    // Re-render correspondences on resize (panel widths change)
    window.addEventListener('resize', () => {
      if (SLIDE.corrShown && SLIDE.cursor === SLIDE.bestIdx) {
        renderCorrespondences();
      }
    });

    slideReset();
  }


  // ────────────────────────────────────────────────────────────────────────
  //  SMOOTH SCROLL FOR NAV
  // ────────────────────────────────────────────────────────────────────────

  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(a => {
      a.addEventListener('click', e => {
        const href = a.getAttribute('href');
        if (!href || href === '#') return;
        const target = document.querySelector(href);
        if (!target) return;
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.replaceState(null, '', href);
      });
    });
  }


  // ────────────────────────────────────────────────────────────────────────
  //  BOOT
  // ────────────────────────────────────────────────────────────────────────

  document.addEventListener('DOMContentLoaded', () => {
    initPipelineDiagram();
    initSlider();
    initSmoothScroll();
  });

})();
