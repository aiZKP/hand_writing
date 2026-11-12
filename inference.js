// In-browser inference for the static (GitHub Pages) version of the dashboard.
//
// A faithful JavaScript port of hwr/preprocess.py, hwr/segment.py and hwr/ctc.py.
// The two networks run in the browser with ONNX Runtime Web (WebAssembly).
// HWR.recognize(canvas) returns exactly the same JSON structure as the Python
// server's /api/predict, so app.js renders both identically.
"use strict";

const HWR = (() => {
  let K, charset, crnn, cnn;

  async function load(cfg) {
    K = cfg.constants;
    charset = cfg.geometry.charset;
    ort.env.wasm.numThreads = 1;   // GitHub Pages is not cross-origin isolated -> single thread
    [crnn, cnn] = await Promise.all([
      ort.InferenceSession.create("models/crnn.onnx"),
      ort.InferenceSession.create("models/char_cnn.onnx"),
    ]);
  }

  // ---------------------------------------------------------------- helpers
  const f32 = Math.fround;                       // numpy float32 arithmetic
  function pyRound(x) {                          // Python round(): ties to even
    const f = Math.floor(x), d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }
  function maxOf(a) { let m = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m; }
  function lse(...xs) {                          // log(sum(exp(x))) computed stably
    let m = -Infinity;
    for (const x of xs) if (x > m) m = x;
    if (m === -Infinity) return -Infinity;
    let s = 0;
    for (const x of xs) s += Math.exp(x - m);
    return m + Math.log(s);
  }

  // canvas -> ink map (port of image_to_ink for a transparent canvas)
  function canvasToInk(canvas) {
    const w = canvas.width, h = canvas.height;
    const px = canvas.getContext("2d").getImageData(0, 0, w, h).data;
    const data = new Float32Array(w * h);
    let minA = 1;
    for (let j = 0; j < w * h; j++) minA = Math.min(minA, px[4 * j + 3] / 255);
    for (let j = 0; j < w * h; j++) {
      const a = f32(px[4 * j + 3] / 255);
      const lum = f32((f32(px[4 * j] / 255) + f32(px[4 * j + 1] / 255) + f32(px[4 * j + 2] / 255)) / 3);
      data[j] = minA < 0.99 ? a * (1 - lum) : 1 - lum;
    }
    return { w, h, data };
  }

  // ------------------------------------------- Pillow-compatible LANCZOS resize
  // Bit-exact port of Pillow's Resample.c for 8-bit images: precomputed filter
  // coefficients in 22-bit fixed point, horizontal pass first, then vertical.
  const PREC = 22, ONE = 1 << PREC, HALF = 1 << (PREC - 1);
  function sinc(x) { if (x === 0) return 1; x *= Math.PI; return Math.sin(x) / x; }
  function lanczos(x) { return (-3 <= x && x < 3) ? sinc(x) * sinc(x / 3) : 0; }
  function clip8(v) { if (v >= ONE * 256) return 255; if (v <= 0) return 0; return Math.floor(v / ONE); }

  function coeffs(inSize, outSize) {
    const scale = inSize / outSize;
    const filterscale = Math.max(scale, 1);
    const support = 3 * filterscale, ss = 1 / filterscale;
    const bounds = [], kk = [];
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      let xmin = Math.trunc(center - support + 0.5);
      if (xmin < 0) xmin = 0;
      let xmax = Math.trunc(center + support + 0.5);
      if (xmax > inSize) xmax = inSize;
      xmax -= xmin;
      const k = new Array(xmax);
      let ww = 0;
      for (let x = 0; x < xmax; x++) { k[x] = lanczos((x + xmin - center + 0.5) * ss); ww += k[x]; }
      for (let x = 0; x < xmax; x++) {
        const v = ww !== 0 ? k[x] / ww : k[x];
        k[x] = v < 0 ? Math.trunc(-0.5 + v * ONE) : Math.trunc(0.5 + v * ONE);
      }
      bounds.push([xmin, xmax]);
      kk.push(k);
    }
    return { bounds, kk };
  }

  function resizeU8(src, w, h, ow, oh) {
    let cur = src, cw = w;
    if (ow !== w) {                                       // horizontal pass
      const { bounds, kk } = coeffs(w, ow);
      const out = new Uint8Array(ow * h);
      for (let y = 0; y < h; y++) {
        for (let xx = 0; xx < ow; xx++) {
          const [xmin, xmax] = bounds[xx], k = kk[xx];
          let s = HALF;
          for (let x = 0; x < xmax; x++) s += cur[y * cw + x + xmin] * k[x];
          out[y * ow + xx] = clip8(s);
        }
      }
      cur = out; cw = ow;
    }
    if (oh !== h) {                                       // vertical pass
      const { bounds, kk } = coeffs(h, oh);
      const out = new Uint8Array(cw * oh);
      for (let yy = 0; yy < oh; yy++) {
        const [ymin, ymax] = bounds[yy], k = kk[yy];
        for (let x = 0; x < cw; x++) {
          let s = HALF;
          for (let y = 0; y < ymax; y++) s += cur[(y + ymin) * cw + x] * k[y];
          out[yy * cw + x] = clip8(s);
        }
      }
      cur = out;
    }
    return cur;
  }

  // ------------------------------------------------ band_to_model_input (Model B)
  function bandToModelInput(band) {
    const mx = maxOf(band.data);
    if (!(mx > 0)) return null;
    const thr = f32(K.ink_threshold * mx);
    let first = -1, last = -1;
    for (let x = 0; x < band.w; x++) {
      for (let y = 0; y < band.h; y++) {
        if (band.data[y * band.w + x] > thr) { if (first < 0) first = x; last = x; break; }
      }
    }
    if (first < 0) return null;
    const bh = band.h, margin = pyRound(K.crop_margin * bh);
    const x0 = first - margin, x1 = last + 1 + margin, cw = x1 - x0;
    const u8 = new Uint8Array(cw * bh);                   // crop (zero padding outside)
    for (let y = 0; y < bh; y++) {
      for (let x = 0; x < cw; x++) {
        const sx = x0 + x;
        const v = sx >= 0 && sx < band.w ? band.data[y * band.w + sx] : 0;
        u8[y * cw + x] = Math.trunc(f32(v * 255));        // (crop*255).astype(uint8)
      }
    }
    let nw = Math.max(1, pyRound(cw * (K.img_h / bh)));
    nw = Math.min(nw, K.max_w);
    const r = resizeU8(u8, cw, bh, nw, K.img_h);
    const W = Math.max(nw, K.min_w);
    const data = new Float32Array(W * K.img_h);
    for (let y = 0; y < K.img_h; y++) for (let x = 0; x < nw; x++) data[y * W + x] = r[y * nw + x] / 255;
    return { w: W, h: K.img_h, data };
  }

  // ------------------------------------------------ segmentation (Model A)
  function segment(ink) {
    const { w, h, data } = ink;
    const mx = maxOf(data);
    if (!(mx > 0)) return [];
    const thr = f32(K.ink_threshold * mx);
    const labels = new Int32Array(w * h), stack = new Int32Array(w * h);
    const minArea = K.seg_min_area_frac * h * h;
    let comps = [], n = 0;
    for (let i = 0; i < w * h; i++) {                     // raster order like scipy
      if (labels[i] || !(data[i] > thr)) continue;
      n++;
      let top = 0, area = 0, x0 = w, x1 = -1, y0 = h, y1 = -1;
      stack[top++] = i; labels[i] = n;
      while (top) {
        const p = stack[--top], px = p % w, py = (p - px) / w;
        area++;
        if (px < x0) x0 = px; if (px > x1) x1 = px; if (py < y0) y0 = py; if (py > y1) y1 = py;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {                   // 8-connectivity
            const qx = px + dx, qy = py + dy;
            if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
            const q = qy * w + qx;
            if (!labels[q] && data[q] > thr) { labels[q] = n; stack[top++] = q; }
          }
        }
      }
      if (area >= minArea) comps.push({ x0, x1: x1 + 1, y0, y1: y1 + 1, labs: [n] });
    }
    let merged = true;                                    // merge strongly x-overlapping parts
    while (merged) {
      merged = false;
      comps.sort((a, b) => a.x0 - b.x0);
      outer:
      for (let i = 0; i < comps.length; i++) {
        for (let j = i + 1; j < comps.length; j++) {
          const a = comps[i], b = comps[j];
          const overlap = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0);
          const narrow = Math.min(a.x1 - a.x0, b.x1 - b.x0);
          if (overlap > K.seg_merge_overlap * narrow) {
            comps[i] = { x0: Math.min(a.x0, b.x0), x1: Math.max(a.x1, b.x1), y0: Math.min(a.y0, b.y0),
                         y1: Math.max(a.y1, b.y1), labs: a.labs.concat(b.labs) };
            comps.splice(j, 1);
            merged = true;
            break outer;
          }
        }
      }
    }
    comps.sort((a, b) => a.x0 - b.x0);
    return comps.map((c) => {
      const cw = c.x1 - c.x0, ch = c.y1 - c.y0, labs = new Set(c.labs);
      const crop = new Float32Array(cw * ch);
      for (let y = 0; y < ch; y++) {
        for (let x = 0; x < cw; x++) {
          const q = (y + c.y0) * w + x + c.x0;
          crop[y * cw + x] = labs.has(labels[q]) ? data[q] : 0;
        }
      }
      return { box: [c.x0, c.y0, c.x1, c.y1], crop: { w: cw, h: ch, data: crop } };
    });
  }

  // EMNIST-style 28x28 normalization (port of normalize_char_emnist_style)
  function normalizeChar(crop) {
    const S = K.char_size, inner = S - 4;
    const out = new Float32Array(S * S);
    const mx = maxOf(crop.data);
    const thr = f32(K.ink_threshold * Math.max(mx, 1e-6));
    let ymin = crop.h, ymax = -1, xmin = crop.w, xmax = -1;
    for (let y = 0; y < crop.h; y++) {
      for (let x = 0; x < crop.w; x++) {
        if (crop.data[y * crop.w + x] > thr) {
          if (y < ymin) ymin = y; if (y > ymax) ymax = y; if (x < xmin) xmin = x; if (x > xmax) xmax = x;
        }
      }
    }
    if (ymax < 0) return out;
    const h = ymax - ymin + 1, w = xmax - xmin + 1, side = Math.max(h, w);
    const oy = Math.floor((side - h) / 2), ox = Math.floor((side - w) / 2);
    const sq = new Uint8Array(side * side);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        sq[(y + oy) * side + x + ox] = Math.trunc(f32(crop.data[(y + ymin) * crop.w + x + xmin] * 255));
      }
    }
    const r = resizeU8(sq, side, side, inner, inner);
    for (let y = 0; y < inner; y++) for (let x = 0; x < inner; x++) out[(y + 2) * S + x + 2] = r[y * inner + x] / 255;
    const m = maxOf(out);
    if (m > 0) for (let i = 0; i < out.length; i++) out[i] = out[i] / m;
    return out;
  }

  // ------------------------------------------------ CTC decoding
  function greedy(lp, T, C) {
    const path = [], confs = [];
    let text = "", prev = 0;
    for (let t = 0; t < T; t++) {
      let k = 0;
      for (let c = 1; c < C; c++) if (lp[t * C + c] > lp[t * C + k]) k = c;
      const p = Math.exp(lp[t * C + k]);
      if (k !== 0 && k !== prev) { text += charset[k - 1]; confs.push(p); }
      else if (k !== 0 && k === prev) confs[confs.length - 1] = Math.max(confs[confs.length - 1], p);
      path.push(k);
      prev = k;
    }
    return { text, confs, path };
  }

  function beamSearch(lp, T, C, beam = 10, topK = 5, prune = Math.log(1e-3)) {
    const desc = (a, b) => (b.s > a.s ? 1 : b.s < a.s ? -1 : 0);   // stable, -Infinity safe
    let beams = [{ prefix: [], pb: 0, pnb: -Infinity }];
    for (let t = 0; t < T; t++) {
      const row = lp.subarray(t * C, (t + 1) * C);
      const order = Array.from({ length: C }, (_, c) => c).sort((a, b) => row[b] - row[a]);
      const cands = order.slice(0, 8).filter((c) => row[c] > prune || c === 0);
      const nxt = new Map();
      const get = (prefix) => {
        const key = prefix.join(",");
        let e = nxt.get(key);
        if (!e) { e = { prefix, pb: -Infinity, pnb: -Infinity }; nxt.set(key, e); }
        return e;
      };
      for (const { prefix, pb, pnb } of beams) {
        for (const c of cands) {
          const p = row[c];
          if (c === 0) { const e = get(prefix); e.pb = lse(e.pb, pb + p, pnb + p); continue; }
          const last = prefix.length ? prefix[prefix.length - 1] : null;
          const eNew = get(prefix.concat([c]));
          if (c === last) {
            const eSame = get(prefix);
            eSame.pnb = lse(eSame.pnb, pnb + p);           // "a"+"a" (no blank) stays "a"
            eNew.pnb = lse(eNew.pnb, pb + p);              // "a"+"-"+"a" -> "aa"
          } else {
            eNew.pnb = lse(eNew.pnb, pb + p, pnb + p);
          }
        }
      }
      beams = [...nxt.values()].map((e) => ({ ...e, s: lse(e.pb, e.pnb) })).sort(desc).slice(0, beam);
    }
    return beams.map((e) => ({ text: e.prefix.map((k) => charset[k - 1]).join(""), s: lse(e.pb, e.pnb) }))
      .sort(desc).slice(0, topK);
  }

  // ------------------------------------------------ rendering helpers
  function toDataUrl(arr, w, h) {
    const cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
    const ctx = cv.getContext("2d"), im = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      // dark ink on white; float32 arithmetic like numpy: ((1 - x).clip(0, 1) * 255).astype(uint8)
      const g = Math.trunc(f32(Math.min(1, Math.max(0, f32(1 - arr[i]))) * 255));
      im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = g;
      im.data[4 * i + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    return cv.toDataURL("image/png");
  }
  const r4 = (v) => Math.round(v * 1e4) / 1e4, r3 = (v) => Math.round(v * 1e3) / 1e3;

  // ------------------------------------------------ public: recognize
  async function recognize(canvas, beamWidth = 10) {
    const result = { empty: false, crnn: null, cnn: null, timing_ms: {} };
    const ink = canvasToInk(canvas);
    const x = bandToModelInput(ink);
    if (!x) { result.empty = true; return result; }
    result.model_input = toDataUrl(x.data, x.w, x.h);
    result.model_input_size = [x.w, x.h];

    let t0 = performance.now();                           // ---- Model B: CRNN
    const out = await crnn.run({ image: new ort.Tensor("float32", x.data, [1, 1, x.h, x.w]) });
    const lp = out.log_probs.data, T = out.log_probs.dims[0], C = out.log_probs.dims[2];
    const g = greedy(lp, T, C);
    const beams = beamSearch(lp, T, C, beamWidth, 5);
    const best = beams.length ? beams[0] : { text: g.text, s: 0 };
    const frameConf = [];
    for (let t = 0; t < T; t++) { let m = -Infinity; for (let c = 0; c < C; c++) m = Math.max(m, lp[t * C + c]); frameConf.push(r3(Math.exp(m))); }
    result.crnn = {
      text: best.text, confidence: Math.exp(best.s), greedy_text: g.text,
      char_conf: best.text === g.text ? g.confs.map(r4) : null,
      beams: beams.map((b) => ({ text: b.text, prob: Math.exp(b.s) })),
      frames: g.path.map((k) => (k === 0 ? "·" : charset[k - 1])).join(""),
      frame_conf: frameConf, num_frames: T,
    };
    result.timing_ms.crnn = Math.round((performance.now() - t0) * 10) / 10;

    t0 = performance.now();                               // ---- Model A: segmentation + CharCNN
    const segs = segment(ink);
    if (!segs.length) result.cnn = { text: "", chars: [], confidence: 0 };
    else {
      const S = K.char_size, batch = new Float32Array(segs.length * S * S);
      const chars28 = segs.map((s, i) => { const c = normalizeChar(s.crop); batch.set(c, i * S * S); return c; });
      const logits = (await cnn.run({ image: new ort.Tensor("float32", batch, [segs.length, 1, S, S]) })).logits.data;
      const K2 = charset.length, chars = [];
      let conf = 1;
      segs.forEach((s, i) => {
        const z = logits.subarray(i * K2, (i + 1) * K2), m = maxOf(z);
        let sum = 0;
        const p = Array.from(z, (v) => { const e = Math.exp(v - m); sum += e; return e; }).map((e) => e / sum);
        const top = p.map((v, k) => k).sort((a, b) => p[b] - p[a]).slice(0, 3);
        conf *= p[top[0]];
        chars.push({ box: s.box, top3: top.map((k) => [charset[k], p[k]]), image: toDataUrl(chars28[i], S, S) });
      });
      result.cnn = { text: chars.map((c) => c.top3[0][0]).join(""), chars, confidence: conf };
    }
    result.timing_ms.cnn = Math.round((performance.now() - t0) * 10) / 10;
    return result;
  }

  // exposed for testing against the Python implementation
  return { load, recognize, _internals: { canvasToInk, bandToModelInput, segment, normalizeChar, resizeU8, greedy, beamSearch } };
})();
