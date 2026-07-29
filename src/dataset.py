"""Week-1 dataset loading: decode MRL Eye Dataset filenames and manifest rows.

No preprocessing, no thresholds, no model here -- that is later weeks' work.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EyeSample:
    """One decoded MRL Eye Dataset image record.

    Fields mirror the 8 underscore-separated tokens in an MRL filename
    (see `parse_filename`), plus the filesystem path to the image.
    """

    path: Path
    subject_id: str
    image_id: str
    gender: int
    glasses: int
    eye_state: int
    reflections: int
    lighting: int
    sensor: str

    @property
    def is_open(self) -> bool:
        """True if eyeState == 1 (open); False if closed (0)."""
        return self.eye_state == 1

    @property
    def has_glasses(self) -> bool:
        """True if glasses == 1 (yes); False if no glasses (0)."""
        return self.glasses == 1

    @property
    def lighting_label(self) -> str:
        """'good' if lighting == 1, else 'bad' (lighting == 0)."""
        return "good" if self.lighting == 1 else "bad"


def parse_filename(name: str | Path) -> EyeSample:
    """Decode one MRL Eye Dataset filename into an EyeSample.

    Convention, quoted verbatim from data/raw/mrlEyes_2018_01/annotation.txt:

        subjectID_imageID_gender_glasses_eyeState_reflections_lightingConditions_sensorID.png

        gender:
        0 - male
        1 - famale

        glasses:
        0 - no
        1 - yes

        eye state:
        0 - close
        1 - open

        reflections:
        0 - none
        1 - low
        2 - high

        lighting conditions/image quality:
        0 - bad
        1 - good

        sensor type:
        01 - RealSense SR300 640x480
        02 - IDS Imaging, 1280x1024
        03 - Aptina Imagin 752x480

        example:
        s001_00123_0_0_0_0_0_01.png

    Splitting the basename (without extension) on '_' yields exactly 8
    fields: [subject_id, image_id, gender, glasses, eye_state, reflections,
    lighting, sensor].
    """
    path = Path(name)
    parts = path.stem.split("_")
    if len(parts) != 8:
        raise ValueError(f"expected 8 '_'-separated fields in filename, got {len(parts)}: {path.name!r}")
    subject_id, image_id, gender, glasses, eye_state, reflections, lighting, sensor = parts
    return EyeSample(
        path=path,
        subject_id=subject_id,
        image_id=image_id,
        gender=int(gender),
        glasses=int(glasses),
        eye_state=int(eye_state),
        reflections=int(reflections),
        lighting=int(lighting),
        sensor=sensor,
    )


def load_manifest(path: str | Path) -> list[EyeSample]:
    """Load EyeSample records from a data/samples/manifest.csv file.

    Each row's image path is reconstructed as
    <manifest's parent dir>/<class>/<filename>, i.e. relative to where the
    manifest itself lives (e.g. data/samples/alert/xxx.png).
    """
    manifest_path = Path(path)
    base_dir = manifest_path.parent
    samples: list[EyeSample] = []
    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = base_dir / row["class"] / row["filename"]
            samples.append(
                EyeSample(
                    path=image_path,
                    subject_id=row["subject_id"],
                    image_id=row["image_id"],
                    gender=int(row["gender"]),
                    glasses=int(row["glasses"]),
                    eye_state=int(row["eye_state"]),
                    reflections=int(row["reflections"]),
                    lighting=int(row["lighting"]),
                    sensor=row["sensor"],
                )
            )
    return samples
