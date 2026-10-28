
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bo.event_boxes import pick_centers, make_boxes

FEATHER = Path(
    "/mnt/d/A desk/ICTS PROJECT 20/DATA/O4 Unlensed/"
    "O4_unlensed_extracted/IntrinsicLVCPrior/"
    "MD_SDSS_u_1008/run_0/bilby_style_samples.feather"
)


def main():
    # Load only the two parameters needed for this test.
    df = pd.read_feather(FEATHER, columns=["chirp_mass", "mass_ratio"])

    values = df[["chirp_mass", "mass_ratio"]].to_numpy(dtype=float)

    # Remove invalid values before taking logarithms.
    valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    samples = np.log10(values[valid])

    if len(samples) < 5:
        raise ValueError(f"Not enough valid samples: {len(samples)}")

    rng = np.random.default_rng(138)
    centers = pick_centers(samples, n_centers=5, rng=rng)

    # Both parameters are handled jointly in every box.
    boxes = make_boxes(samples, centers, sort_dimension=0)

    volumes = np.prod(boxes.upper - boxes.lower, axis=1)
    retained_probability = boxes.retained_probability

    print("Input file:", FEATHER.name)
    print("Valid sample count:", len(samples))
    print("Sample dimensions:", samples.shape)
    print("Joint 2D box count:", boxes.n_boxes)
    print("Box coordinate dimensions:", boxes.lower.shape)
    print("Non-positive box volumes:", np.sum(volumes <= 0))
    print("Non-finite box volumes:", np.sum(~np.isfinite(volumes)))
    print("Retained probability:", retained_probability)

    assert boxes.lower.shape[1] == 2
    assert boxes.upper.shape[1] == 2
    assert np.all(volumes > 0)
    assert np.all(np.isfinite(volumes))
    assert 0 <= retained_probability <= 1 + 1e-10

    print("Joint 2D diagnostic: PASS")


if __name__ == "__main__":
    main()
