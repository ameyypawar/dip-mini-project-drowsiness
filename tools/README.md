# Dashboard

One local build with everything: the measured results, an interactive pipeline explorer,
and a live camera feed running the project's image-processing stages in real time.

```bash
./serve.sh          # http://localhost:8777/dashboard.html, opens automatically
./serve.sh 9000     # different port
```

Then click **Start camera**, or drop in any image.

Serving over `localhost` is required. Browsers refuse camera access on `file://` URLs, and
a published web page cannot open a camera at all — which is why the full build lives here.

| File | What it is |
|---|---|
| `dashboard.html` | The complete build. Self-contained; all images and code are inlined. |
| `dip_pipeline.js` | Source of truth for the operators: grayscale, histogram equalisation via the CDF, gamma, mean and median 3×3, Otsu, erosion, dilation, opening, closing. `dashboard.html` inlines a copy. |
| `serve.sh` | Starts the local server and opens the page. |

## Reading the stages

Contrast rises sharply between grayscale and equalised, then the median filter pulls the
noise figure back down without softening the edges. Raising the morphology slider clears
speckle out of the segmented region.

## What this does not do

Every stage shown is the real operator. The openness reading is a geometric proxy — the
vertical extent of the dark region — and is **not** the trained classifier. The support
vector machine runs in Python on 1,774 HOG and LBP features and scores 89.65% on held-out
subjects:

```bash
../.venv/bin/python ../scripts/run_demo.py --source webcam --display
```
