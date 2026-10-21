/* Shared client-side DIP pipeline.
   Pure canvas/typed-array implementations of the operators used in the project,
   so the live camera page and the dashboard upload panel run identical code. */
(function (root) {
  "use strict";

  function toGray(rgba, w, h) {
    const g = new Uint8ClampedArray(w * h);
    for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
      g[p] = (rgba[i] * 0.299 + rgba[i + 1] * 0.587 + rgba[i + 2] * 0.114) | 0;
    }
    return g;
  }

  function histogram(g) {
    const h = new Uint32Array(256);
    for (let i = 0; i < g.length; i++) h[g[i]]++;
    return h;
  }

  /* Histogram equalisation via the cumulative distribution function. */
  function equalize(g) {
    const h = histogram(g), n = g.length, lut = new Uint8ClampedArray(256);
    let cum = 0, cdfMin = 0;
    for (let i = 0; i < 256; i++) { if (h[i]) { cdfMin = cum + h[i]; break; } cum += h[i]; }
    cum = 0;
    for (let i = 0; i < 256; i++) {
      cum += h[i];
      lut[i] = n === cdfMin ? i : Math.round(((cum - cdfMin) / (n - cdfMin)) * 255);
    }
    const out = new Uint8ClampedArray(n);
    for (let i = 0; i < n; i++) out[i] = lut[g[i]];
    return out;
  }

  function gamma(g, y) {
    const lut = new Uint8ClampedArray(256);
    for (let i = 0; i < 256; i++) lut[i] = Math.round(255 * Math.pow(i / 255, y));
    const out = new Uint8ClampedArray(g.length);
    for (let i = 0; i < g.length; i++) out[i] = lut[g[i]];
    return out;
  }

  /* Border pixels replicate the nearest edge, so no operator paints a black frame. */
  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  /* 3x3 median — rank-order rejection, the Task 4/5 winner on impulse noise. */
  function median3(g, w, h) {
    const out = new Uint8ClampedArray(g.length), v = new Uint8Array(9);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let k = 0;
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++)
            v[k++] = g[clamp(y + dy, 0, h - 1) * w + clamp(x + dx, 0, w - 1)];
        for (let i = 1; i < 9; i++) { const t = v[i]; let j = i - 1;
          while (j >= 0 && v[j] > t) { v[j + 1] = v[j]; j--; } v[j + 1] = t; }
        out[y * w + x] = v[4];
      }
    }
    return out;
  }

  function mean3(g, w, h) {
    const out = new Uint8ClampedArray(g.length);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++)
            s += g[clamp(y + dy, 0, h - 1) * w + clamp(x + dx, 0, w - 1)];
        out[y * w + x] = (s / 9) | 0;
      }
    return out;
  }

  /* Otsu: pick the threshold maximising between-class variance. */
  function otsuThreshold(g) {
    const h = histogram(g), n = g.length;
    let sum = 0; for (let i = 0; i < 256; i++) sum += i * h[i];
    let sumB = 0, wB = 0, best = 0, bestT = 127;
    for (let t = 0; t < 256; t++) {
      wB += h[t]; if (!wB) continue;
      const wF = n - wB; if (!wF) break;
      sumB += t * h[t];
      const mB = sumB / wB, mF = (sum - sumB) / wF;
      const between = wB * wF * (mB - mF) * (mB - mF);
      if (between > best) { best = between; bestT = t; }
    }
    return bestT;
  }

  /* Foreground = darker than the threshold: the pupil and iris are the dark region. */
  function threshold(g, t) {
    const out = new Uint8ClampedArray(g.length);
    for (let i = 0; i < g.length; i++) out[i] = g[i] <= t ? 255 : 0;
    return out;
  }

  function morph(bin, w, h, op, iters) {
    let cur = bin;
    for (let it = 0; it < iters; it++) {
      const out = new Uint8ClampedArray(cur.length);
      for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
          let hit = op === "dilate" ? 0 : 255;
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++) {
              const v = cur[clamp(y + dy, 0, h - 1) * w + clamp(x + dx, 0, w - 1)];
              if (op === "dilate") { if (v) hit = 255; }
              else if (!v) hit = 0;
            }
          out[y * w + x] = hit;
        }
      cur = out;
    }
    return cur;
  }

  function opening(bin, w, h) { return morph(morph(bin, w, h, "erode", 1), w, h, "dilate", 1); }
  function closing(bin, w, h) { return morph(morph(bin, w, h, "dilate", 1), w, h, "erode", 1); }

  function stats(g) {
    let s = 0; for (let i = 0; i < g.length; i++) s += g[i];
    const m = s / g.length;
    let v = 0; for (let i = 0; i < g.length; i++) v += (g[i] - m) * (g[i] - m);
    return { mean: m, std: Math.sqrt(v / g.length) };
  }

  function foregroundFraction(bin) {
    let c = 0; for (let i = 0; i < bin.length; i++) if (bin[i]) c++;
    return c / bin.length;
  }

  /* Vertical extent of the dark region, normalised — an openness proxy only.
     NOT the trained classifier; the project's SVM runs in Python. */
  function verticalExtent(bin, w, h) {
    let top = -1, bot = -1;
    for (let y = 0; y < h; y++) {
      let row = 0;
      for (let x = 0; x < w; x++) if (bin[y * w + x]) row++;
      if (row > w * 0.04) { if (top < 0) top = y; bot = y; }
    }
    return top < 0 ? 0 : (bot - top + 1) / h;
  }

  function toImageData(g, w, h) {
    const d = new Uint8ClampedArray(w * h * 4);
    for (let i = 0, p = 0; i < g.length; i++, p += 4) {
      d[p] = d[p + 1] = d[p + 2] = g[i]; d[p + 3] = 255;
    }
    return new ImageData(d, w, h);
  }

  root.DIP = { toGray, histogram, equalize, gamma, median3, mean3, otsuThreshold,
    threshold, morph, opening, closing, stats, foregroundFraction, verticalExtent, toImageData };
})(typeof window !== "undefined" ? window : globalThis);
