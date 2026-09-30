#!/usr/bin/env python3
"""Build docs/task8_report.html: reuses the head/CSS/frame/script of
docs/task7_report.html verbatim and generates the Table 3 content from the
results/task8_*.csv files. Run after make_task8_figures.py."""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
SE_L = {"square3": "3x3 square", "cross3": "3x3 cross", "square5": "5x5 square"}
IMGS = ["good_lighting", "poor_lighting", "glasses_reflection"]


def rd(n):
    with open(R / n) as f:
        return list(csv.DictReader(f))


seg, binr, inp = rd("task8_segmentation_source.csv"), rd("task8_binary_thresholds.csv"), rd("task8_input_binary.csv")
ops, vm, thin, dual = rd("task8_operations.csv"), rd("task8_vanish_merge.csv"), rd("task8_thin_gap.csv"), rd("task8_duality.csv")

t7 = (ROOT / "docs/task7_report.html").read_text()
i0 = t7.index("<!-- ============================== Table 3")
i1 = t7.index("</div>\n</td></tr>\n</tbody>")
prefix, suffix = t7[:i0], t7[i1:]
prefix = prefix.replace("Experiment No. 7", "Experiment No. 8")
prefix = re.sub(r"<title>.*?</title>", "<title>Task 8 -- Binary Morphological Processing: Erosion and Dilation</title>", prefix, flags=re.S)


def row(label, body, compact=False):
    c = " compact" if compact else ""
    return (f'<table class="t3-row{c}"><colgroup><col class="lbl"><col></colgroup>\n'
            f'  <tr><td class="label">{label}</td><td>\n{body}\n  </td></tr>\n</table>\n\n')


def fig(name, n, cap, w=None):
    st = f' style="max-width:{w}%"' if w else ""
    return (f'  <figure>\n    <img src="../results/{name}" alt="Figure {n}"{st}>\n'
            f'    <figcaption>Figure {n}: {cap}</figcaption>\n  </figure>\n')


def table(head, rows, cls="data"):
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'  <table class="{cls}">\n    <tr>{h}</tr>\n    {b}\n  </table>\n'


def sg(v):
    return f"{float(v):+.1f}%"


def op_table(op, img):
    i = next(r for r in inp if r["image"] == img)
    rows = [["input", "0", i["fg_pixels"], i["area_fraction"], "0.0%", i["components"], i["boundary_len"]]]
    for r in ops:
        if r["image"] == img and r["op"] == op:
            rows.append([SE_L[r["se"]], r["iter"], r["fg_pixels"], r["area_fraction"], sg(r["pct_change"]), r["components"], r["boundary_len"]])
    return table(["SE", "Iter.", "Foreground px", "Area fraction", "Change vs input", "Components (8-conn.)", "Boundary length (px)"], rows)


# ---------------------------------------------------------------- content
out = ""
out += row("Experiment Number", "8", True)
out += row("Experiment Title", "Binary Morphological Processing &ndash; Erosion and Dilation", True)

out += row("Aim", """
    To convert the segmented eye image of the previous week into a binary image at more than one
    threshold; to apply <b>binary erosion</b> and <b>binary dilation</b> with three structuring
    elements (3x3 square, 3x3 cross, 5x5 square) and study their effect on object size,
    boundaries, small noise, thin structures, gaps, broken objects and neighbouring objects; to
    compare the two operations; and to study how repeated application (1, 2 and 3 iterations)
    shrinks or grows and eventually removes or merges structures. Every effect is <b>measured</b>
    (foreground pixel count, area fraction, connected components, boundary length, the iteration
    at which regions vanish or merge) rather than described visually only, and the
    erosion/dilation <b>duality</b> is verified numerically.
    <br><br>
    <b>Lineage.</b> The input of this experiment is &ldquo;the final segmented image obtained in the
    previous week&rsquo;s activity&rdquo;, i.e. the <b>Task 7 region-growing output</b>:
    acquisition &rarr; histogram equalization (Task 3) &rarr; seeded region growing
    (<code>src/region_growing.py</code>, automatic darkest-point seed, running-mean criterion,
    T&nbsp;=&nbsp;45, 8-connectivity &ndash; the &ldquo;High&rdquo; tolerance of Task 7) &rarr;
    <b>binary morphology (this report)</b>. T&nbsp;=&nbsp;45 is deliberately the loosest Task 7
    setting: it also captures the eyebrow and speckle around the iris, which gives the
    morphological operators real structure (thin bridges, tiny fragments, gaps) to act on; a
    clean single blob would make erosion/dilation trivial to describe.
""")

out += row("Theory", """
  <h4 class="sub">Binary image and structuring element</h4>
  <p>A binary image is a set A &sub; &#8484;&sup2; of foreground pixel coordinates (here the dark
  pupil/iris/eyebrow pixels; background is the complement A<sup>c</sup>). A <b>structuring
  element</b> (SE) B is a small set of offsets with a designated <b>origin</b>. B<sub>z</sub> denotes
  B translated so its origin sits at pixel z, and the <b>reflection</b> is
  B&#770; = {&minus;b : b &isin; B}.</p>
  <h4 class="sub">Erosion (fit)</h4>
  <p style="text-align:center">A &#8854; B = { z : B<sub>z</sub> &sub; A } = &#8898;<sub>b&isin;B</sub> A<sub>&minus;b</sub></p>
  <p>A pixel z survives only if the SE centred at z <b>fits entirely inside</b> the object.
  Erosion shrinks objects, strips a layer of boundary pixels, deletes objects and protrusions
  narrower than the SE, and widens gaps between objects; it can also split an object joined
  by a thin neck.</p>
  <h4 class="sub">Dilation (hit)</h4>
  <p style="text-align:center">A &#8853; B = { z : B&#770;<sub>z</sub> &cap; A &ne; &empty; } = &#8899;<sub>b&isin;B</sub> A<sub>+b</sub></p>
  <p>A pixel z is set if the (reflected) SE centred at z <b>hits</b> (overlaps) at least one
  object pixel. Dilation grows objects, thickens boundaries, closes narrow gaps and holes, joins
  broken pieces and merges nearby objects. Figure 2 shows both rules on a small shape.</p>
  <h4 class="sub">Duality</h4>
  <p style="text-align:center">(A &#8854; B)<sup>c</sup> = A<sup>c</sup> &#8853; B&#770; &nbsp;&nbsp;&hArr;&nbsp;&nbsp; A &#8854; B = (A<sup>c</sup> &#8853; B&#770;)<sup>c</sup></p>
  <p>Eroding the foreground is the same as dilating the background with the reflected SE, and
  taking the complement. All three SEs here are symmetric so B&#770; = B, but the code
  reflects anyway. The identity is verified pixel-for-pixel in Output (Table 10, Figure 17).</p>
  <h4 class="sub">Structuring-element geometry</h4>
  <p>SE size and shape set how much and in which directions an object changes (Figure 1). The
  <b>3x3 square</b> (9 pixels) is 8-connected: it changes an object by one pixel in all eight
  directions, so diagonal edges move faster (by &radic;2) than axis-aligned ones. The
  <b>3x3 cross</b> (5 pixels: centre + 4 neighbours) is 4-connected: it moves edges by one pixel
  orthogonally but only ~0.7 pixel along diagonals, so it is gentler and rounds corners
  into diamonds. The <b>5x5 square</b> (25 pixels) is the Minkowski sum of two 3x3 squares,
  B<sub>5</sub> = B<sub>3</sub> &#8853; B<sub>3</sub>, so one 5x5 pass equals two 3x3 passes
  (confirmed pixel-identically for both operations on all 3 images). Iterating with the same SE
  n times equals one operation with the n-fold Minkowski sum: a growing square (or diamond for
  the cross).</p>
  <h4 class="sub">Related composite operations (from <code>src/morphology.py</code>)</h4>
  <p>Opening = erosion then dilation (removes structures smaller than B); closing = dilation then
  erosion (bridges gaps smaller than B). They are used below as quantitative probes:
  |A| &minus; |open(A)| counts &ldquo;thin/small&rdquo; pixels and |close(A)| &minus; |A| counts bridgeable gap pixels.</p>
""")

out += row("Resources / Apparatus Required", """
    Software: Python 3.11 &bull; OpenCV &bull; NumPy &bull; Matplotlib &bull; scikit-image &bull;
    Pillow &bull; Jupyter Notebook / VS Code. Modules reused: <code>src/morphology.py</code>,
    <code>src/segment.py</code>, <code>src/region_growing.py</code> (Task 7),
    <code>src/enhancement.py</code> (Task 3). New: <code>src/binary_morphology.py</code>.
""", True)

out += row("Algorithm", """
  <h4 class="sub">A. Binary Conversion</h4>
  <ol>
    <li>Read the test image (grayscale; convert with <code>cvtColor</code> only if it has 3 channels).</li>
    <li>Apply histogram equalization (Task 3) and upscale as in Tasks 3&ndash;7.</li>
    <li>Run seeded region growing (Task 7; auto seed, T&nbsp;=&nbsp;45, 8-connected) to obtain the
    segmentation; form the <b>segmented image</b> S = enhanced pixel inside the region, 255 outside.</li>
    <li>Threshold: foreground = pixels of S with intensity &lt; t (dark = object), for
    t &isin; {40, 60, 100}; display original / grayscale / binary and record counts.</li>
    <li>Select t&nbsp;=&nbsp;60 as the working binary image for Steps B&ndash;D.</li>
  </ol>
  <h4 class="sub">B. Erosion</h4>
  <ol>
    <li>Build the SEs: 3x3 square, 3x3 cross (<code>cv2.MORPH_CROSS</code>), 5x5 square.</li>
    <li>For each SE compute A &#8854; B (one iteration).</li>
    <li>Record foreground count, area fraction, % change, connected components and boundary length;
    record thin-structure pixels |A| &minus; |open(A)|.</li>
  </ol>
  <h4 class="sub">C. Dilation</h4>
  <ol>
    <li>For each SE compute A &#8853; B (one iteration).</li>
    <li>Record the same measures plus bridgeable-gap pixels |close(A)| &minus; |A|.</li>
    <li>Compare erosion, dilation and their difference (morphological gradient).</li>
  </ol>
  <h4 class="sub">D. Iteration Study and Duality</h4>
  <ol>
    <li>Apply erosion and dilation for 1, 2 and 3 iterations with every SE; tabulate all measures.</li>
    <li>Keep eroding (up to 150 iterations) until the foreground is empty; record the vanishing
    iteration of the whole image and of each original component.</li>
    <li>Keep dilating; record the iteration at which the component count first drops (first merge)
    and the iteration at which a single component remains.</li>
    <li>Verify A &#8854; B = (A<sup>c</sup> &#8853; B&#770;)<sup>c</sup> and its dual for every image, SE
    and iteration count, and verify OpenCV against an independent NumPy fit/hit implementation.</li>
  </ol>
""")

out += row("Program code", """
  <p class="note">Trimmed to the essentials. Full module: <code>src/binary_morphology.py</code>;
  driver: <code>scripts/make_task8_figures.py</code>. No existing module is modified.</p>
<pre>def to_binary(img, threshold, dark_foreground=True):
    if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = (img &lt; threshold) if dark_foreground else (img &gt;= threshold)
    return m.astype(np.uint8) * 255

def make_se(shape, size):                      # 'square' or 'cross'
    return morph.structuring_element("rect" if shape == "square" else shape, size)

def erode(mask, se, iterations=1):  return cv2.erode(mask, se, iterations=iterations)
def dilate(mask, se, iterations=1): return cv2.dilate(mask, se, iterations=iterations)

def erode_ref(mask, se, iterations=1):         # A (-) B = AND of shifted copies
    a = mask &gt; 0; ph, pw = se.shape[0]//2, se.shape[1]//2
    for _ in range(iterations):
        p = np.pad(a, ((ph, ph), (pw, pw)), constant_values=True)
        out = np.ones_like(a)
        for i, j in zip(*np.nonzero(se)):
            out &amp;= p[i:i+a.shape[0], j:j+a.shape[1]]
        a = out
    return a.astype(np.uint8) * 255            # dilate_ref: OR of shifts, pad False

def n_components(mask):  return cv2.connectedComponents((mask &gt; 0).astype(np.uint8), connectivity=8)[0] - 1

def boundary_length(mask):                     # fg pixels with a 4-neighbour background
    inner = cv2.erode(mask, cross3, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return int(((mask &gt; 0) &amp; (inner == 0)).sum())

def iterations_to_vanish(mask, se, max_iter=150):
    cur = mask
    for k in range(1, max_iter + 1):
        cur = cv2.erode(cur, se)
        if not cur.any(): return k

def merge_iterations(mask, se, max_iter=40):   # first drop in component count / single component
    n0 = n_components(mask); first = one = None; cur = mask
    for k in range(1, max_iter + 1):
        cur = cv2.dilate(cur, se); n = n_components(cur)
        if first is None and n &lt; n0: first = k
        if n &lt;= 1: one = k; break
    return {"n0": n0, "first_merge_iter": first, "single_iter": one}

def duality_check(mask, se, iterations=1):
    comp, se_r = 255 - mask, se[::-1, ::-1].copy()
    return int((erode(mask, se, iterations) != 255 - dilate(comp, se_r, iterations)).sum())</pre>
""")

# ---------------------------------------------------------------- Output
o = ""
o += """
  <h4 class="sub">Test images and source segmentation (Task 7 lineage)</h4>
""" + table(["Label", "File", "Condition", "Upscaled size", "Task 7 seed (r,c)", "Region px (T=45)", "Otsu of S"],
            [["good_lighting", "<code>alert/s0031_00366_1_0_1_0_1_02.png</code>", "good lighting", seg[0]["size"], seg[0]["seed"], seg[0]["region_pixels"], seg[0]["otsu_on_segmented"]],
             ["poor_lighting", "<code>alert/s0014_08297_0_0_1_1_0_02.png</code>", "poor / uneven lighting", seg[1]["size"], seg[1]["seed"], seg[1]["region_pixels"], seg[1]["otsu_on_segmented"]],
             ["glasses_reflection", "<code>alert/s0035_00133_0_1_1_2_1_02.png</code>", "glasses + IR reflection", seg[2]["size"], seg[2]["seed"], seg[2]["region_pixels"], seg[2]["otsu_on_segmented"]]], cls="data wide7")
o += """  <p class="note">The native images are 128x128, 130x130 and 144x144 and are upscaled 2x (nearest-neighbour) exactly as in Tasks 3&ndash;7.
  The source files are single-channel near-infrared images, so the grayscale-conversion step is a <b>no-op</b>:
  the &ldquo;grayscale&rdquo; panel in Figures 3&ndash;5 is numerically identical to the original and is shown only to
  document the step, not because a colour conversion took place.</p>
  <h4 class="sub">Diagrams</h4>
""" + fig("task8_diagram_se.png", 1, "The three structuring elements as pixel grids, origin marked in red.") \
    + fig("task8_diagram_fit_hit.png", 2, "Erosion (fit) and dilation (hit) as a 3x3 cross sliding over a small binary shape.")

o += "  <h4 class=\"sub\">Task 1 &ndash; Binary conversion at three thresholds</h4>\n"
for k, l in enumerate(IMGS):
    o += fig(f"task8_step1_{l}.png", 3 + k, f"Original, grayscale, region-grown segmentation and binary images at t = 40, 60, 100 ({l}).", 68)
o += """  <p>Foreground = pixels of the segmented image S darker than t. Because S already contains only region-grown
  pixels (values up to about 100 after equalization; background 255), raising t admits progressively brighter
  boundary pixels of the region:</p>
""" + table(["Image", "t", "Foreground px", "Area fraction", "Components", "Boundary length", "Small comps (&le;30 px)"],
            [[r["image"], r["threshold"], r["fg_pixels"], r["area_fraction"], r["components"], r["boundary_len"], r["small_comps_le30"]] for r in binr])
o += """  <p class="note">Table 1. Effect of the binarization threshold. Low t keeps only the darkest pupil core (few, separate objects); t = 60
  additionally keeps the iris/eyebrow texture as many small fragments (17 components on poor_lighting, 14 of them &le;30 px);
  t = 100 recovers essentially the whole region-grown mask, which fuses into a single component on all three images. The Otsu
  threshold of S (80 / 88 / 62) lies between t = 60 and t = 100. <b>t = 60 is used for all morphology experiments</b> because it
  contains both large objects and genuine small noise/thin structures, which is what erosion and dilation must be judged on.</p>
"""
o += "  <p><b>Working binary images (t = 60):</b></p>\n" + table(
    ["Image", "Foreground px", "Area fraction", "Components", "Boundary length (px)"],
    [[r["image"], r["fg_pixels"], r["area_fraction"], r["components"], r["boundary_len"]] for r in inp])

o += "  <h4 class=\"sub\">Task 2 &ndash; Erosion (3 structuring elements, 1&ndash;3 iterations)</h4>\n"
for k, l in enumerate(IMGS):
    o += fig(f"task8_erosion_{l}.png", 6 + k, f"Erosion of the t = 60 binary image, rows = 3x3 square / 3x3 cross / 5x5 square, columns = input and 1&ndash;3 iterations ({l}).", 50)
for l in IMGS:
    o += f"  <p><b>Table {2+IMGS.index(l)} &ndash; erosion, {l}</b></p>\n" + op_table("erosion", l)

o += "  <h4 class=\"sub\">Task 3 &ndash; Dilation (3 structuring elements, 1&ndash;3 iterations)</h4>\n"
for k, l in enumerate(IMGS):
    o += fig(f"task8_dilation_{l}.png", 9 + k, f"Dilation of the t = 60 binary image, same layout as Figures 6&ndash;8 ({l}).", 50)
for l in IMGS:
    o += f"  <p><b>Table {5+IMGS.index(l)} &ndash; dilation, {l}</b></p>\n" + op_table("dilation", l)
o += """  <p class="note">Tables 2&ndash;7 (erosion then dilation, per image). Boundary length = foreground pixels with a 4-neighbour background pixel.
  Component counts use 8-connectivity. &ldquo;Change vs input&rdquo; is relative to the input binary&rsquo;s foreground pixel count.</p>
"""

o += "  <h4 class=\"sub\">Task 4 &ndash; Erosion versus dilation</h4>\n"
for k, l in enumerate(IMGS):
    o += fig(f"task8_compare_{l}.png", 12 + k, f"Input, erosion, dilation, morphological gradient and overlay (3x3 square, 1 iteration; {l}).")
o += fig("task8_area_bars.png", 15, "Foreground area change after one iteration, per structuring element and image.")

o += "  <h4 class=\"sub\">Task 5 &ndash; Iterations: vanishing and merging</h4>\n"
o += fig("task8_curves.png", 16, "Top: erosion area change (%) versus iterations to 40. Bottom: dilation component count versus iterations.")
vrows = []
for r in vm:
    vrows.append([r["image"], SE_L[r["se"]], r["components0"], r["erosion_all_vanish_iter"], r["erosion_component_vanish_iters"].replace("/", ", ") if r["image"] != "poor_lighting" else "1 (x13), 2 (x1&ndash;2), 3 (x0&ndash;1), then main (see text)", r["dilation_first_merge_iter"], r["dilation_single_component_iter"]])
# use exact strings for poor_lighting from CSV, collapsed
def collapse(s):
    parts = s.split("/")
    from collections import Counter
    c = Counter(parts)
    order = []
    for p in parts:
        if p not in [x for x, _ in order]:
            order.append((p, c[p]))
    return ", ".join(f"{p}" + (f" (x{n})" if n > 1 else "") for p, n in order)
vrows = [[r["image"], SE_L[r["se"]], r["components0"], r["erosion_all_vanish_iter"], collapse(r["erosion_component_vanish_iters"]),
          r["dilation_first_merge_iter"], r["dilation_single_component_iter"]] for r in vm]
o += table(["Image", "SE", "Initial components", "Iteration at which ALL foreground vanishes (erosion)", "Vanishing iteration of each original component (erosion)",
            "First merge iteration (dilation: component count drops)", "Iteration reaching a single component (dilation)"], vrows, cls="data wide7")
o += """  <p class="note">Table 8. Vanishing and merging iterations, measured by iterating until the foreground is empty (erosion) or one component remains (dilation).</p>
  <h4 class="sub">Thin structures and bridgeable gaps (SE-size probes)</h4>
""" + table(["Image", "SE", "|A| &minus; |open(A)| (thin / small pixels)", "|close(A)| &minus; |A| (bridgeable gap pixels)"],
            [[r["image"], SE_L[r["se"]], r["thin_pixels_A_minus_open"], r["gap_pixels_close_minus_A"]] for r in thin])
o += """  <p class="note">Table 9. Pixels belonging to structures thinner than the SE (removed by opening) and background gaps narrower than the SE (filled by closing).</p>
  <h4 class="sub">Duality verification</h4>
"""
o += fig("task8_duality.png", 17, "Erosion of A versus complement of dilation of the complement, 3x3 cross, good_lighting; the difference image is empty.")
n_cases = len(dual)
tot = {c: sum(int(r[c]) for r in dual) for c in ("erode_vs_dual_diff", "dilate_vs_dual_diff", "erode_cv_vs_ref_diff", "dilate_cv_vs_ref_diff")}
drows = []
for l in IMGS:
    for k in SE_L:
        rs = [r for r in dual if r["image"] == l and r["se"] == k]
        drows.append([l, SE_L[k], "1, 2, 3"] + [sum(int(r[c]) for r in rs) for c in ("erode_vs_dual_diff", "dilate_vs_dual_diff", "erode_cv_vs_ref_diff", "dilate_cv_vs_ref_diff")])
o += table(["Image", "SE", "Iterations", "A&#8854;B vs (A<sup>c</sup>&#8853;B&#770;)<sup>c</sup> differing px", "A&#8853;B vs (A<sup>c</sup>&#8854;B&#770;)<sup>c</sup> differing px", "OpenCV vs NumPy erosion differing px", "OpenCV vs NumPy dilation differing px"], drows, cls="data wide7")
o += f"""  <p class="note">Table 10. Duality check (differing pixels summed over iterations 1, 2, 3) over {n_cases} cases (3 images &times; 3 SEs &times; 3 iteration counts). Total differing pixels:
  erosion&ndash;dilation duality {tot['erode_vs_dual_diff']}, dilation&ndash;erosion duality {tot['dilate_vs_dual_diff']}, OpenCV-vs-set-definition {tot['erode_cv_vs_ref_diff']} / {tot['dilate_cv_vs_ref_diff']}.
  <b>The duality holds exactly (0 differing pixels in every case)</b>, and the independent NumPy fit/hit implementation reproduces OpenCV exactly.
  Border handling matters here: OpenCV treats out-of-image pixels as foreground for erosion and background for dilation, which is precisely the dual convention.</p>
"""
out += row("Output", o)

# ---------------------------------------------------------------- discussion
def g(img, se, op, it, key="pct_change"):
    return next(r[key] for r in ops if r["image"] == img and r["se"] == se and r["op"] == op and r["iter"] == str(it))


cmp = """
  <p>Both operators use the same SE and are duals, but they act on opposite sets: erosion trims the object, dilation trims the
  background. The measured one-iteration changes (Figure 15, Tables 2&ndash;7) make the symmetry concrete:</p>
""" + table(["Property", "Erosion A&#8854;B", "Dilation A&#8853;B"], [
    ["Object size (3x3 square, 1 it.)", f"good {sg(g('good_lighting','square3','erosion',1))}, poor {sg(g('poor_lighting','square3','erosion',1))}, glasses {sg(g('glasses_reflection','square3','erosion',1))}",
     f"good {sg(g('good_lighting','square3','dilation',1))}, poor {sg(g('poor_lighting','square3','dilation',1))}, glasses {sg(g('glasses_reflection','square3','dilation',1))}"],
    ["Object size (5x5 square, 1 it.)", f"good {sg(g('good_lighting','square5','erosion',1))}, poor {sg(g('poor_lighting','square5','erosion',1))}, glasses {sg(g('glasses_reflection','square5','erosion',1))}",
     f"good {sg(g('good_lighting','square5','dilation',1))}, poor {sg(g('poor_lighting','square5','dilation',1))}, glasses {sg(g('glasses_reflection','square5','dilation',1))}"],
    ["Boundaries", "Move inward; boundary pixel count falls on all images and SEs (e.g. poor 1417 &rarr; 981, 3x3 square) as ragged edges and specks disappear", "Move outward; boundary pixel count also falls (poor 1417 &rarr; 1090) because fragments fuse and notches fill"],
    ["Small noise", "Removed: poor_lighting drops from 17 to 8 components (3x3 square), 4 (5x5); the 32-px dot on glasses_reflection dies at iteration 2", "Enlarged and absorbed into neighbours: the same 4-px speck on good_lighting is swallowed by the main object at iteration 1"],
    ["Thin structures", "Eroded away first; the 1-3-px bridges between eyebrow and eye break (good_lighting splits into 2 components at iteration 3)", "Thickened; thin strands become solid"],
    ["Gaps / neighbours", "Gaps between objects widen (components can increase)", "Gaps close, neighbours merge (components decrease)"],
    ["Total effect on shape", "Anti-extensive: A&#8854;B &sub; A (origin in B)", "Extensive: A &sub; A&#8853;B"]])
cmp += f"""
  <p>The asymmetry between good/glasses (about &plusmn;6&ndash;7 % per iteration) and poor_lighting (about &plusmn;24 %) is a size effect: poor_lighting
  has a much smaller, more fragmented foreground (7160 px, 1417 boundary pixels) so the boundary layer removed or added is a far larger
  fraction of it. On the compact good/glasses objects (14&thinsp;784 and 11&thinsp;760 px, boundary 962 and 612) the same one-pixel shell is a
  few percent. A larger SE scales the effect: the 5x5 square changes area about twice as much as the 3x3 square per pass
  (good {sg(g('good_lighting','square5','erosion',1))} vs {sg(g('good_lighting','square3','erosion',1))}), being exactly two 3x3 passes. The cross is the gentlest
  (good {sg(g('good_lighting','cross3','erosion',1))}, {sg(g('good_lighting','cross3','dilation',1))}) because it does not move diagonal edges.</p>
  <p>Erosion followed by dilation (opening) does not return the original: the pixels that stay lost are exactly the structures thinner than
  the SE (Table 9: up to 952 pixels on poor_lighting with the 5x5 square), and closing bridges gaps by up to 872 pixels. This is why the
  pair is used together in <code>src/morphology.py</code>&rsquo;s <code>clean_mask</code>.</p>
"""
out += row("Comparison of Erosion and Dilation", cmp)

it = f"""
  <p><b>Repeated erosion.</b> Each iteration peels a one-pixel shell (3x3 square), so area falls roughly linearly in iterations for a compact
  object (Figure 16, top) and the object finally disappears. Measured area change after 1, 2, 3 iterations (3x3 square):
  good {sg(g('good_lighting','square3','erosion',1))} / {sg(g('good_lighting','square3','erosion',2))} / {sg(g('good_lighting','square3','erosion',3))};
  poor {sg(g('poor_lighting','square3','erosion',1))} / {sg(g('poor_lighting','square3','erosion',2))} / {sg(g('poor_lighting','square3','erosion',3))};
  glasses {sg(g('glasses_reflection','square3','erosion',1))} / {sg(g('glasses_reflection','square3','erosion',2))} / {sg(g('glasses_reflection','square3','erosion',3))}.
  With the 5x5 square poor_lighting loses {sg(g('poor_lighting','square5','erosion',3))} after only 3 iterations.</p>
  <p><b>Which structures disappear.</b> Structures vanish in order of width. On poor_lighting (3x3 square) 15 of the 17 original components (the speckle, areas 1&ndash;52 px) vanish at
  iteration 1 and one more at iteration 2; only the main eye/eyebrow object survives, and it sheds small pieces itself (8 components after iteration 1: a 5412-px body plus
  pieces of 4&ndash;20 px). By iteration 3 the body has split into 3 pieces (56, 1084 and 2316 px), the eyebrow band separating from the iris ring (Figure 7). On glasses_reflection the isolated 32-px glare dot
  vanishes at iteration 2 (3x3 square) or 3 (cross), and on good_lighting the 4-px speck vanishes at iteration 1. The last surviving structure is always the
  widest part of the pupil/iris region, which vanishes entirely at iteration
  {next(r['erosion_all_vanish_iter'] for r in vm if r['image']=='good_lighting' and r['se']=='square3')} (good), {next(r['erosion_all_vanish_iter'] for r in vm if r['image']=='poor_lighting' and r['se']=='square3')} (poor),
  {next(r['erosion_all_vanish_iter'] for r in vm if r['image']=='glasses_reflection' and r['se']=='square3')} (glasses) for the 3x3 square, {next(r['erosion_all_vanish_iter'] for r in vm if r['image']=='good_lighting' and r['se']=='cross3')} / {next(r['erosion_all_vanish_iter'] for r in vm if r['image']=='poor_lighting' and r['se']=='cross3')} / {next(r['erosion_all_vanish_iter'] for r in vm if r['image']=='glasses_reflection' and r['se']=='cross3')} for the cross and
  {next(r['erosion_all_vanish_iter'] for r in vm if r['image']=='good_lighting' and r['se']=='square5')} / {next(r['erosion_all_vanish_iter'] for r in vm if r['image']=='poor_lighting' and r['se']=='square5')} / {next(r['erosion_all_vanish_iter'] for r in vm if r['image']=='glasses_reflection' and r['se']=='square5')} for the 5x5 square (Table 8). Vanishing time is
  about half the object&rsquo;s smallest width divided by the SE radius, so the cross (which moves diagonal edges more slowly) lasts longest and the 5x5 square shortest.
  Erosion can also <i>increase</i> the component count by cutting a thin neck: good_lighting goes 2 &rarr; 1 &rarr; 1 &rarr; 2 (3x3 square) because the speck is removed first
  and the neck to a lower structure breaks at iteration 3.</p>
  <p><b>Repeated dilation.</b> Each iteration adds a one-pixel shell: area grows about {sg(g('good_lighting','square3','dilation',1))}, {sg(g('good_lighting','square3','dilation',2))},
  {sg(g('good_lighting','square3','dilation',3))} on good_lighting, but the growth of poor_lighting is far larger ({sg(g('poor_lighting','square3','dilation',3))} after 3 iterations;
  {sg(g('poor_lighting','square5','dilation',3))} with the 5x5 square) because gaps between fragments are being filled as well as the outline moving outward.
  The measured component counts fall while the boundary length falls (poor 1417 &rarr; 829): the outline becomes smoother.</p>
  <p><b>Which regions merge.</b> The first merge happens at iteration 1 with the 3x3 square, 3x3 cross (poor, good) and 5x5 square on every image; on glasses_reflection
  the cross needs 2 iterations (the 32-px dot lies diagonally off the main object, and the cross does not extend along diagonals). On poor_lighting the
  17 components collapse as 5 (iteration 1) &rarr; 3 &rarr; 2 &rarr; 1 for the 3x3 square (single component at iteration 4), 6 &rarr; 3 &rarr; 2 &rarr; 2 &rarr; 1 for the cross (iteration 5)
  and 3 &rarr; 1 (iteration 2) for the 5x5 square. The speckle around the iris merges with the iris ring and eyebrow first (after 3 iterations of the 3x3 square only a 64-px fragment remains separate from the
  11&thinsp;052-px main body); the last merge absorbs that residual fragment. Beyond that point further dilation only inflates one merged blob (good_lighting: a single
  component from iteration 1, area {sg(g('good_lighting','square3','dilation',3))} at 3 iterations and still growing).</p>
  <p><b>Erosion and dilation are not inverses.</b> Eroding then dilating with the same SE (opening) restores the large object&rsquo;s shape but not deleted
  fragments; dilating then eroding (closing) keeps merged objects merged. Repeating either operation is therefore a controlled loss of information, which is
  the price of noise removal or gap bridging.</p>
"""
out += row("Effect of Iterations", it)

obs = """
  <ol>
    <li>All three source images are single-channel infrared, so the grayscale step changes nothing (verified: the panels are numerically identical).</li>
    <li>The binary image depends strongly on the threshold: for poor_lighting t = 40 / 60 / 100 give 8 / 17 / 1 components and 4528 / 7160 / 11&thinsp;130 foreground pixels. Thresholds
    close to the equalized Otsu value (62&ndash;88) keep the iris texture as fragments; higher values fill it in.</li>
    <li>Erosion (3x3 square, 1 iteration) removed 5.9 &ndash; 23.6 % of foreground and dilation added 5.8 &ndash; 23.9 %; for every image the two are nearly equal and opposite (e.g. good &minus;7.02 % / +6.97 %), differing slightly because of local concavities and image structure.</li>
    <li>Structuring-element order (least to most aggressive) is 3x3 cross &lt; 3x3 square &lt; 5x5 square for both operators on all 3 images, and the 5x5 square equals two 3x3-square iterations pixel-for-pixel.</li>
    <li>Erosion removes small noise: poor_lighting component count goes 17 &rarr; 8 (3x3 square, iteration 1) and the 32-px glare dot on glasses_reflection is gone by iteration 2.</li>
    <li>Dilation merges: poor_lighting reaches a single component at iteration 4 (3x3 square), 5 (cross), 2 (5x5).</li>
    <li>The complete foreground disappears under erosion at iteration 27 / 12 / 29 (3x3 square), 37 / 16 / 34 (cross) and 14 / 6 / 15 (5x5 square) for good / poor / glasses.</li>
    <li>Boundary length decreases under both operators (smoothing), so it is not a monotone indicator of object size: use area for size and components for noise/merging.</li>
    <li>The duality A&#8854;B = (A<sup>c</sup>&#8853;B&#770;)<sup>c</sup> holds with 0 differing pixels in all 27 cases, for both directions, and OpenCV matches an independent set-definition implementation exactly.</li>
    <li>Region-grown segmentation at T = 45 includes the eyebrow and speckle around the iris (Task 7 &ldquo;leakage&rdquo;); erosion followed by dilation is the natural way to separate the iris from the eyebrow, and dilation/closing the natural way to unify the iris speckle &ndash; this is the practical link to the drowsiness pipeline.</li>
  </ol>
"""
out += row("Observations", obs)

concl = """
  <p>Binary erosion and dilation were implemented in <code>src/binary_morphology.py</code>, reusing <code>src/morphology.py</code> for structuring elements and the
  Task 7 region-growing output (<code>src/region_growing.py</code>) as the segmented input, and evaluated on the same 3 dataset images with three thresholds
  (t = 40, 60, 100; t = 60 used for the experiments), three structuring elements and 1&ndash;3 iterations, all with measured pixel counts, component counts and boundary
  lengths. Erosion shrinks objects, removes noise and thin structures and can split objects; dilation grows objects, closes gaps and merges neighbours. Per iteration the
  3x3 square changes area by about 6&ndash;24 %, the cross slightly less and the 5x5 square about twice as much; the effect is largest on the small, fragmented
  poor_lighting foreground. Repeated erosion eliminates the whole foreground after 6&ndash;37 iterations depending on image and SE, small fragments first and the
  iris/pupil body last; repeated dilation merges all fragments into one component within 1&ndash;5 iterations. The duality of the two operators was confirmed numerically
  (0 differing pixels in 27 cases), and an independent fit/hit implementation matched OpenCV exactly. For the drowsiness pipeline the useful operations are the
  composites: opening to remove speckle and detach the eyebrow, and closing to make the iris a single solid region before the eye-openness measurement.</p>
"""
out += row("Conclusion", concl)

html = prefix + "<!-- ============================== Table 3 ============================== -->\n\n" + out + suffix
(ROOT / "docs/task8_report.html").write_text(html)
print("wrote docs/task8_report.html", len(html))
