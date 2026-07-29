# DIP Topic Mapping

Course requirement: incorporate **image enhancement, segmentation, morphological operations,
transforms, and image compression**. This document maps each mandated topic to a concrete
pipeline stage and the exact OpenCV/scikit-image functions used.

| DIP topic | Pipeline stage | Concrete functions |
|---|---|---|
| **Image enhancement** | CLAHE + gamma correction on low-light IR eye crops; reflection suppression for eyeglasses | `cv2.createCLAHE()`, `cv2.LUT()` (gamma table), `cv2.inpaint()` / specular-highlight masking for glare from eyeglasses |
| **Segmentation** | Sclera/iris/pupil isolation via adaptive + Otsu thresholding | `cv2.adaptiveThreshold()`, `cv2.threshold(..., cv2.THRESH_OTSU)`, `skimage.filters.threshold_otsu` |
| **Morphological operations** | Opening/closing to clean eye mask; hole-filling before area measurement | `cv2.morphologyEx(..., cv2.MORPH_OPEN)`, `cv2.morphologyEx(..., cv2.MORPH_CLOSE)`, `scipy.ndimage.binary_fill_holes` / `skimage.morphology.remove_small_holes` |
| **Transforms** | DFT/DCT for blur & motion detection; Hough circle transform for iris fitting | `cv2.dft()` / `np.fft.fft2`, `cv2.dct()`, `cv2.HoughCircles()` |
| **Image compression** | DCT/JPEG quality sweep quantifying accuracy-vs-bitrate trade-off for in-vehicle transmission | `cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])` sweep over `q`, `cv2.dct()` block-wise analysis |

## Where this sits in the overall pipeline

1. Raw IR eye crop (MRL dataset, or webcam eye ROI after face/eye localization)
2. **Enhancement** — CLAHE + gamma normalize illumination, suppress eyeglass glare
3. **Segmentation** — isolate sclera/iris/pupil region from eyelid/skin
4. **Morphology** — clean segmentation mask (remove speckle, fill holes) before measuring
   eyelid-opening area
5. **Transforms** — DFT/DCT sharpness check to reject motion-blurred frames; Hough circle to
   fit iris boundary for finer eye-openness estimation
6. **Compression** — evaluate JPEG quality vs. classification-accuracy trade-off, relevant to
   bandwidth-constrained in-vehicle transmission (e.g. sending frames to a fleet server)

This 5-row table is reused verbatim (condensed) in `docs/proposal/proposal.html` and in
`README.md`, since it directly answers the instructor's stated grading constraint.
