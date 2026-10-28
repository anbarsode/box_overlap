
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bo.event_boxes import make_event_boxes, save_event_boxes, load_event_boxes

FEATHER = Path(
    "/mnt/d/A desk/ICTS PROJECT 20/DATA/O4 Unlensed/"
    "O4_unlensed_extracted/IntrinsicLVCPrior/"
    "MD_SDSS_u_1008/run_0/bilby_style_samples.feather"
)

COLUMNS = [
    "chirp_mass",
    "mass_ratio",
    "chi_eff",
    "ra",
    "dec",
]

OUTPUT = PROJECT_ROOT / "validation/generated_boxes/feather_5d_test.boxes"


def main():
    if not FEATHER.is_file():
        raise FileNotFoundError(f"Feather file not found: {FEATHER}")

    df = pd.read_feather(FEATHER, columns=COLUMNS)
    values = df[COLUMNS].to_numpy(dtype=float)

    # All five coordinates must be finite.
    valid = np.isfinite(values).all(axis=1)

    # The two mass parameters must be positive before taking log10.
    valid &= (values[:, 0] > 0) & (values[:, 1] > 0)

    samples = values[valid].copy()
    rejected = len(values) - len(samples)

    if len(samples) < 10:
        raise ValueError(f"Not enough valid samples: {len(samples)}")

    # Transform masses only; leave chi_eff, ra and dec unchanged.
    samples[:, 0] = np.log10(samples[:, 0])
    samples[:, 1] = np.log10(samples[:, 1])

    if not np.isfinite(samples).all():
        raise ValueError("Non-finite values remain after transformation.")

    print("Input file:", FEATHER.name)
    print("Selected parameters:", COLUMNS)
    print("Total rows:", len(values))
    print("Valid rows:", len(samples))
    print("Rejected rows:", rejected)
    print("Sample shape:", samples.shape)
    print("Coordinate order:", COLUMNS)
    print("Transform: log10(chirp_mass), log10(mass_ratio)")

    # Test the existing two-block event-box generator.
    event = make_event_boxes(
        samples,
        rng=np.random.default_rng(138),
        block_dimensions=(3, 2),
        n_centers=(5, 8),
        ntrial=(2, 3),
    )

    print("Block 1 dimensions:", event.blocks[0].lower.shape[1])
    print("Block 1 boxes:", event.blocks[0].n_boxes)
    print("Block 2 dimensions:", event.blocks[1].lower.shape[1])
    print("Block 2 boxes:", event.blocks[1].n_boxes)
    print("Retained probability:", event.retained_probability)

    # Check dimensions and box geometry.
    assert len(event.blocks) == 2
    assert event.blocks[0].lower.shape[1] == 3
    assert event.blocks[1].lower.shape[1] == 2

    for block in event.blocks:
        assert block.lower.shape == block.upper.shape
        assert block.lower.shape[1] > 0
        assert np.isfinite(block.lower).all()
        assert np.isfinite(block.upper).all()
        assert np.all(block.upper > block.lower)

    assert 0 <= event.retained_probability <= 1 + 1e-10

    # Test serialization round-trip.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_event_boxes(OUTPUT, event)
    print("Saved boxes:", OUTPUT)

    restored = load_event_boxes(OUTPUT)

    assert len(restored.blocks) == 2

    for original, reloaded in zip(event.blocks, restored.blocks):
        assert original.lower.shape == reloaded.lower.shape
        assert original.upper.shape == reloaded.upper.shape
        assert np.allclose(original.lower, reloaded.lower)
        assert np.allclose(original.upper, reloaded.upper)

    print("Reloaded block dimensions:",
          [block.lower.shape[1] for block in restored.blocks])
    print("5D Feather pipeline test: PASS")


if __name__ == "__main__":
    main()
