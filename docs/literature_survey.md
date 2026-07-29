# Literature Survey — Driver Drowsiness Detection

Team: Amey Pawar (23108B0057), Sejal Andhale (23108B0049)
Course: Digital Image Processing, Mini Project — Week 1

## 1. Classical / heuristic vision approaches

**PERCLOS (PERcentage of eyelid CLOSure)** is the standard adopted by Carnegie Mellon
Research Institute (Wierwille et al., 1994) and later validated by NHTSA/FHWA driver-fatigue
studies as the single most reliable optical fatigue metric. It measures the percentage of time
per minute (or per fixed window) that the eyelids are ≥80% closed, i.e. it is inherently a
**temporal** measure computed over a video sequence, not a single frame.

**Eye Aspect Ratio (EAR)** was introduced by Soukupová & Čech (2016, *21st Computer Vision
Winter Workshop*) as a lightweight, landmark-based proxy for eyelid closure. EAR is a scalar
ratio of vertical to horizontal distances among 6 points per eye (typically from a 68-point
dlib/iBUG facial landmark model), thresholded (commonly ~0.2–0.25) and combined with a
consecutive-frame counter to declare a blink/closure event and trigger an alarm.

**Mouth Aspect Ratio (MAR)** is the analogous ratio computed over mouth landmarks, used to
detect yawning as a secondary fatigue cue, usually fused with EAR in threshold-based systems.

Both EAR and MAR require a face-landmark detector (dlib HOG+SVM or CNN, or MediaPipe Face
Mesh) to first localize eye/mouth corner points on a full face frame — they are **not**
computed on pre-cropped eye-patch datasets such as MRL.

### 1.1 Reference implementations (EAR-thresholding lineage)

| Project | Approach | Notes |
|---|---|---|
| PyImageSearch, *"Drowsiness detection with OpenCV"* (Rosebrock, 2017) | dlib 68-landmark detector + EAR threshold + consecutive-frame counter + audio alarm | Most-cited tutorial; baseline this project's methodology extends |
| GitHub: SafwenNaimi/Drowsiness-detection-with-OpenCV | dlib + EAR + imutils, alarm on sustained low EAR | Direct derivative of PyImageSearch pipeline |
| GitHub: Practical-CV/Drowsiness-detection-with-OpenCV | Same EAR-threshold + frame-counter pattern | Confirms pattern is a de-facto community standard |
| GitHub: Nisarg1112/Driver-s-Drowsiness-Detection-using-OpenCV-Python | dlib + EAR, single fixed threshold | Illustrates single-threshold limitation (see §3) |

All four repos share the same weakness: a **single, hand-tuned EAR threshold** applied
uniformly across users.

## 2. Deep-learning approaches

CNN-based eye-state / drowsiness classifiers report high benchmark accuracy:

- **VGG16, ResNet50V2, InceptionV3** transfer-learning classifiers on eye-state / drowsiness
  datasets: reported **92–99%** accuracy across various closed/open-eye benchmarks.
- **MDPI *Sensors* 2024, 24(19):6261** — CNN + MAR fusion for an embedded real-time
  drowsiness-detection system.
- **MDPI *Applied Sciences* 2023, 13(13):7849** — CNN-based real-time eye-state
  classification pipeline.
- **Nature *Scientific Reports* 2025** — transformer / Vision-Transformer and Swin-Transformer
  architectures applied to drowsiness/eye-state classification, reporting gains over CNN
  baselines on cross-subject generalization.
- **arXiv 2511.13618** — MediaPipe Face Mesh + EAR computed in real time, showing the
  landmark-based EAR approach remains competitive when paired with modern lightweight
  landmark detectors.
- **arXiv 2509.17498** — comparative study of YOLOv5 through YOLOv11 for
  drowsiness/eye-state detection, benchmarking detector-generation trade-offs (speed vs.
  accuracy) relevant to in-vehicle real-time constraints.

## 3. Commercial / OEM systems

| System | Approach | Claimed capability |
|---|---|---|
| **Bosch** driver drowsiness detection | Steering-behaviour algorithm, fuses ~70 vehicle signals (steering angle, lane position, time of day, trip duration) | No camera; behavioural inference |
| **Bosch** interior-sensing camera (DMS) | Eye opening, gaze direction, head posture | Camera-based driver monitoring |
| **Volvo** Driver Alert Control | Lane-tracking + steering-pattern analysis | ~97% claimed detection accuracy (manufacturer figure) |
| **Toyota** | Near-infrared steering-column-mounted camera, in production since 2006 | One of the earliest OEM camera-based systems |
| **Seeing Machines** Guardian | Fleet driver-monitoring system, eye-gaze + eyelid tracking | Deployed across 1100+ fleet operators |
| **Smart Eye AB, Continental, Panasonic Automotive** | Eye/gaze/head-pose tracking DMS suppliers | Tier-1 automotive DMS providers |

**Regulatory driver:** the **EU General Safety Regulation (GSR)** mandates Driver Drowsiness
and Attention Warning (DDAW) systems on new vehicle type-approvals, making camera- or
behaviour-based drowsiness detection a compliance requirement rather than an optional feature —
this is a strong practical motivation for low-cost, vision-only solutions like this project.

## 4. Comparison table

| Approach | Input | Real-time cost | Accuracy (reported) | Key weakness |
|---|---|---|---|---|
| PERCLOS | Video, eyelid closure over time | Low | Gold-standard reference metric | Needs long window, landmark accuracy dependent |
| EAR/MAR thresholding | Single frame + landmarks | Very low | Good in controlled conditions | Single fixed threshold fails across subjects/lighting |
| CNN eye-state classifiers | Cropped eye image | Medium (GPU helpful) | 92–99% on benchmarks | Dataset bias, poor cross-condition generalization |
| Transformer/ViT/Swin | Cropped eye or face image | High | Reported gains over CNN | Heavy, hard to run real-time on embedded/edge hardware |
| YOLOv5–v11 | Full frame, face/eye detection | Low–medium | Competitive, model-size dependent | Detection-quality dependent on lighting/occlusion |
| Bosch steering-behaviour | CAN-bus signals, no camera | Very low | Proprietary | No direct eye/face evidence, indirect signal |
| Vision DMS (Seeing Machines, Smart Eye) | Camera, gaze + eyelid | Medium | Commercial-grade | Expensive, closed-source, hardware-bound |

## 5. Research gap (this project's motivation)

**Existing solutions and research gap.** The dominant open-source pattern
(PyImageSearch-style EAR thresholding) uses a **single, globally fixed EAR threshold**
tuned on one demonstrator's face. This fails to generalize across:

1. **Inter-individual eye-shape variation** (EAR baseline differs by person even at full-eye-open).
2. **Eyeglasses**, which introduce reflections/occlusion that corrupt landmark localization.
3. **Low-light / IR conditions** typical of night driving, where raw pixel contrast is poor
   and landmark detectors lose precision.

Deep-learning classifiers improve raw accuracy but are usually evaluated on curated,
well-lit datasets and do not address the **preprocessing weaknesses** (low contrast, glare,
noise) that cause failure in the classical pipeline in the first place.

**This project's position:** rather than only swapping the classifier, this project inserts
a **DIP-heavy preprocessing/enhancement front-end** (CLAHE, gamma correction, reflection
suppression, adaptive thresholding for segmentation) ahead of an eye-state classifier trained
on the MRL Eye Dataset, and combines it with **adaptive** (not fixed) EAR thresholding plus
PERCLOS at inference time on live video — directly targeting the individual-, eyeglass-, and
lighting-variation gap identified above.

## References

1. Wierwille, W.W. et al. — PERCLOS driver fatigue metric, Carnegie Mellon Research Institute / FHWA.
2. Soukupová, T., Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks.* 21st CVWW.
3. Rosebrock, A. (2017). *Drowsiness detection with OpenCV.* PyImageSearch.
4. GitHub: SafwenNaimi/Drowsiness-detection-with-OpenCV
5. GitHub: Practical-CV/Drowsiness-detection-with-OpenCV
6. GitHub: Nisarg1112/Driver-s-Drowsiness-Detection-using-OpenCV-Python
7. MDPI *Sensors* 2024, 24(19), 6261 — CNN + MAR embedded drowsiness detection.
8. MDPI *Applied Sciences* 2023, 13(13), 7849 — CNN real-time eye-state classification.
9. Nature *Scientific Reports* 2025 — transformer/ViT/Swin drowsiness classification.
10. arXiv:2511.13618 — MediaPipe + EAR real-time drowsiness detection.
11. arXiv:2509.17498 — YOLOv5–v11 comparative drowsiness/eye-state detection.
12. Bosch Mobility — Driver Drowsiness Detection & interior-sensing camera DMS product pages.
13. Volvo Cars — Driver Alert Control.
14. Toyota — near-infrared driver-monitoring camera (production since 2006).
15. Seeing Machines — Guardian fleet driver-monitoring system.
16. EU General Safety Regulation (GSR 2019/2144) — DDAW mandate.
17. MRL Eye Dataset, VSB–TU Ostrava — https://mrl.cs.vsb.cz/data/eyedataset/
