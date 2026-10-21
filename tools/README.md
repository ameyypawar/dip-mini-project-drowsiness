# Interactive tools

Two browser demos that run the project's image-processing stages client-side.

| File | What it is |
|---|---|
| `live_pipeline.html` | Live webcam demo. All six stages on the camera feed in real time. |
| `dashboard.html` | The full project dashboard, with an upload-based pipeline panel. Also published as an Artifact. |
| `dip_pipeline.js` | Shared implementations: grayscale, histogram equalisation, gamma, mean and median filtering, Otsu, erosion, dilation, opening, closing. |

## Running the live camera demo

Browsers only grant camera access over `https` or `localhost`, so serve the folder:

```bash
cd tools
python3 -m http.server 8777
```

Then open <http://localhost:8777/live_pipeline.html> and click **Start camera**.
Opening the file directly with `file://` will not work — the browser blocks the camera.

## What these do and do not show

Every stage is the real operator, reimplemented in JavaScript so it runs on live video.
The openness reading is a geometric proxy (vertical extent of the dark region), **not** the
trained classifier. The support vector machine runs in Python on 1,774 HOG and LBP features
and scores 89.65% on held-out subjects — use `scripts/run_demo.py --source webcam` for that.
