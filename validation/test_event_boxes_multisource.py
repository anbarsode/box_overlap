"""Checks for the sampled event probability box generator.

Run from the project root with the ``gw`` environment's Python:
    python validation/test_event_boxes.py
"""

from pathlib import Path
import argparse
import os
import tarfile
import sys
import tempfile
import warnings
# Path to the O4 unlensed archive
TAR_PATH = Path(
    "/mnt/d/A desk/ICTS PROJECT 20/DATA/O4 Unlensed/O4_unlensed.tar.gz"
)

# Directory where the archive will be extracted
EXTRACT_DIR = Path(
    "/mnt/d/A desk/ICTS PROJECT 20/DATA/O4 Unlensed/O4_unlensed_extracted"
)


def extract_o4_data():
    """Safely extract the O4 archive if the destination is missing or empty."""
    if not TAR_PATH.is_file():
        raise FileNotFoundError(f"Archive not found: {TAR_PATH}")

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    if any(EXTRACT_DIR.iterdir()):
        print(f"O4 data already extracted: {EXTRACT_DIR}")
        return EXTRACT_DIR

    print(f"Extracting O4 archive to: {EXTRACT_DIR}")
    destination = EXTRACT_DIR.resolve()

    with tarfile.open(TAR_PATH, "r:gz") as tar:
        members = tar.getmembers()

        # Validate every member before extracting; this also works on Python
        # versions that do not support tarfile.extractall(filter="data").
        for member in members:
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(
                    f"Unsafe path in archive: {member.name!r}"
                )
            if member.issym() or member.islnk():
                link_target = (target.parent / member.linkname).resolve()
                if link_target != destination and destination not in link_target.parents:
                    raise ValueError(
                        f"Unsafe link in archive: {member.name!r}"
                    )

        # Extract only regular files/directories; reject special entries.
        for member in members:
            if not (member.isfile() or member.isdir()):
                raise ValueError(
                    f"Unsupported special entry in archive: {member.name!r}"
                )
        tar.extractall(path=destination, members=members)

    print("Extraction completed.")
    return EXTRACT_DIR


os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from bo import (
        load_event_boxes,
        make_boxes,
        make_event_boxes,
        pick_centers,
        save_event_boxes,
        make_box_trials,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import the box-generator module 'bo'. Run this script "
        "from the project checkout and make sure bo.py is in the project root."
    ) from exc


def reference_boxes(samples, centers, weight_function=None):
    """Slow, explicit implementation of the construction for comparison.

    This version forms every preliminary cell with ``np.ndindex``. It is kept
    here as an independent correctness check; production code creates only
    occupied cells.
    """
    edges = []
    for coordinate in range(samples.shape[1]):
        selected = np.unique(centers[:, coordinate])
        middle = selected[:-1] + (selected[1:] - selected[:-1]) / 2
        edges.append(np.r_[samples[:, coordinate].min(),
                           middle,
                           samples[:, coordinate].max()])

    lower = []
    upper = []
    counts = []
    shape = tuple(len(axis_edges) - 1 for axis_edges in edges)
    for cell in np.ndindex(shape):
        cell_lower = np.array([edges[k][cell[k]] for k in range(samples.shape[1])])
        cell_upper = np.array([edges[k][cell[k] + 1] for k in range(samples.shape[1])])
        inside = np.all(samples >= cell_lower, axis=1)
        for coordinate in range(samples.shape[1]):
            if cell[coordinate] == shape[coordinate] - 1:
                inside &= samples[:, coordinate] <= cell_upper[coordinate]
            else:
                inside &= samples[:, coordinate] < cell_upper[coordinate]
        if np.count_nonzero(inside) > 1:
            lower.append(samples[inside].min(axis=0))
            upper.append(samples[inside].max(axis=0))
            counts.append(np.count_nonzero(inside))

    lower = np.asarray(lower)
    upper = np.asarray(upper)
    density = np.asarray(counts) / (len(samples) * np.prod(upper - lower, axis=1))
    centers = lower + (upper - lower) / 2
    weight = (np.ones(len(lower)) if weight_function is None else
              np.asarray(weight_function(centers)))
    return lower, upper, density, weight


def check_density_and_disjointness(boxes):
    """Check finite densities and pairwise intersection volume."""
    volume = np.prod(boxes.upper - boxes.lower, axis=1)
    assert np.all(volume > 0)
    assert np.all(boxes.density > 0)
    assert np.all(boxes.weight == 1)

    for i in range(boxes.n_boxes):
        overlap_widths = np.minimum(boxes.upper[i], boxes.upper[i + 1:]) - np.maximum(
            boxes.lower[i], boxes.lower[i + 1:])
        assert not np.any(np.all(overlap_widths > 0, axis=1))


def test_known_grid():
    # The selected x values are 0 and 1.5, hence the separator is 0.75.
    # The selected y values are 0 and 3, hence the separator is 1.5.
    samples = np.array([[0., 0.], [0.5, 0.5], [1.5, 3.], [2., 4.]])
    boxes = make_boxes(samples, samples[[0, 2]])
    assert boxes.n_boxes == 2
    np.testing.assert_array_equal(boxes.lower, [[0., 0.], [1.5, 3.]])
    np.testing.assert_array_equal(boxes.upper, [[0.5, 0.5], [2., 4.]])
    np.testing.assert_allclose(boxes.density, [2., 1.])
    check_density_and_disjointness(boxes)
    mass = np.sum(boxes.density * np.prod(boxes.upper - boxes.lower, axis=1))
    np.testing.assert_allclose(mass, 1)

    weighted = make_boxes(samples, samples[[0, 2]], lambda x: x[:, 0] + 2 * x[:, 1])
    np.testing.assert_allclose(weighted.weight, [0.75, 8.75])


def test_against_explicit_full_grid():
    """Compare the sparse implementation with all cells of a small full grid."""
    rng = np.random.default_rng(734)
    for dimension in (1, 2, 3):
        samples = rng.normal(size=(200, dimension))
        centers = pick_centers(samples, 4, rng)
        weight_function = lambda x: np.sum(x**2, axis=1)
        actual = make_boxes(samples, centers, weight_function)
        expected = reference_boxes(samples, centers, weight_function)
        expected_rows = np.column_stack(expected)
        expected_rows = expected_rows[np.argsort(expected_rows[:, 0], kind="stable")]
        np.testing.assert_allclose(actual.rows(), expected_rows)


def test_singleton_discard():
    samples = np.array([[0., 0.], [0.5, 0.5], [2., 4.]])
    boxes = make_boxes(samples, samples[[0, 2]])
    assert boxes.n_boxes == 1
    np.testing.assert_array_equal(boxes.lower, [[0., 0.]])
    np.testing.assert_array_equal(boxes.upper, [[0.5, 0.5]])
    np.testing.assert_allclose(boxes.density, [8. / 3.])
    check_density_and_disjointness(boxes)
    np.testing.assert_allclose(np.sum(boxes.density * np.prod(boxes.upper - boxes.lower, axis=1)),
                               2. / 3.)

    all_singletons = make_boxes(samples, samples)
    assert all_singletons.n_boxes == 0
    assert all_singletons.retained_probability == 0




def load_event_samples(path, input_type="auto", feather_columns=None):
    """Load synthetic NPY or real GW Feather samples.

    Feather input supports:
      2D: log10(chirp_mass), log10(mass_ratio)
      5D: log10(chirp_mass), log10(mass_ratio), chi_eff, ra, dec

    Synthetic NPY coordinates are left unchanged.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    if input_type == "auto":
        suffix = path.suffix.lower()
        if suffix == ".npy":
            input_type = "synthetic"
        elif suffix == ".feather":
            input_type = "feather"
        else:
            raise ValueError(
                f"Cannot infer input type from {path.suffix!r}; "
                "use a .npy or .feather file."
            )

    if input_type == "synthetic":
        samples = np.asarray(np.load(path, allow_pickle=False), dtype=float)

        if samples.ndim != 2 or samples.shape[0] == 0:
            raise ValueError(f"Invalid sample array shape: {samples.shape}")
        if samples.shape[1] != 5:
            raise ValueError(
                f"Synthetic NPY input must have 5 parameters; "
                f"got shape {samples.shape}"
            )

    elif input_type == "feather":
        import pandas as pd

        columns = list(feather_columns or [
            "chirp_mass", "mass_ratio", "chi_eff", "ra", "dec"
        ])

        if len(columns) not in (2, 5):
            raise ValueError(
                "Feather input requires either 2 columns "
                "(chirp_mass,mass_ratio) or 5 columns."
            )

        frame = pd.read_feather(path)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Missing Feather columns: {missing}\n"
                f"Columns available in {path.name}: {frame.columns.tolist()}"
            )

        values = frame[columns].to_numpy(dtype=float)

        # Chirp mass and mass ratio must be positive before log10.
        valid = np.isfinite(values).all(axis=1)
        valid &= values[:, 0] > 0
        valid &= values[:, 1] > 0
        values = values[valid]

        if len(values) == 0:
            raise ValueError(f"No valid samples in {path}")

        # Both supported formats begin with chirp_mass and mass_ratio.
        transformed = [
            np.log10(values[:, 0]),
            np.log10(values[:, 1]),
        ]

        if len(columns) == 5:
            transformed.extend([
                values[:, 2],  # chi_eff
                values[:, 3],  # RA
                values[:, 4],  # declination
            ])

        samples = np.column_stack(transformed)

    else:
        raise ValueError(f"Unknown input type: {input_type}")

    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError(f"Invalid sample array shape: {samples.shape}")
    if not np.isfinite(samples).all():
        raise ValueError("Sample array contains NaN or infinite values.")

    return samples


def test_event_samples(samples, label, output_path):
    """Build, validate, save, and reload event boxes for 2D or 5D samples."""
    samples = np.asarray(samples, dtype=float)

    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError(f"Invalid sample shape: {samples.shape}")
    if samples.shape[1] not in (2, 5):
        raise ValueError(
            f"Expected 2D or 5D samples, got shape {samples.shape}"
        )

    dimension = samples.shape[1]

    print(f"\nTesting {label}")
    print("Sample shape:", samples.shape)

    if dimension == 2:
        # Split the two parameters into two one-dimensional blocks.
        block_dimensions = (1, 1)
        n_centers = (5, 8)
        ntrial = (1, 1)
    else:
        # Preserve the original five-dimensional regression-test setup.
        block_dimensions = (3, 2)
        n_centers = (5, 8)
        ntrial = (1, 1)

    event = make_event_boxes(
        samples,
        n_centers=n_centers,
        rng=np.random.default_rng(138),
        block_dimensions=block_dimensions,
        ntrial=ntrial,
    )

    blocks = list(event)
    np.testing.assert_allclose(
        event.retained_probability,
        np.prod([block.retained_probability for block in blocks]),
    )

    start = 0
    for block, dimension_block, center_count in zip(
        blocks, block_dimensions, n_centers
    ):
        block_samples = samples[:, start:start + dimension_block]
        start += dimension_block

        assert block.n_boxes <= center_count ** dimension_block
        check_density_and_disjointness(block)

        volumes = np.prod(block.upper - block.lower, axis=1)
        assert np.all(volumes > 0)

        inside = np.all(
            (block_samples[:, None, :] >= block.lower[None, :, :])
            & (block_samples[:, None, :] <= block.upper[None, :, :]),
            axis=2,
        )
        retained_fraction = np.mean(np.any(inside, axis=1))
        integrated_probability = np.sum(block.density * volumes)

        np.testing.assert_allclose(
            integrated_probability, retained_fraction
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_event_boxes(output_path, event)
    restored = load_event_boxes(output_path)

    assert len(blocks) == len(list(restored))
    for old, new in zip(blocks, restored):
        np.testing.assert_array_equal(old.rows(), new.rows())

    np.testing.assert_array_equal(
        restored.retained_probability, event.retained_probability
    )

    # Preserve the existing sorting and round-trip checks.
    if dimension == 2:
        sort_dimensions = (0, 0)
    else:
        sort_dimensions = (2, 1)

    reordered = load_event_boxes(
        output_path, sort_dimensions=sort_dimensions
    )

    for original, new_block in zip(blocks, reordered):
        assert np.all(
            np.diff(new_block.lower[:, new_block.sort_dimension]) >= 0
        )
        original_rows = original.rows()
        new_rows = new_block.rows()

        np.testing.assert_array_equal(
            original_rows[
                np.lexsort(original_rows[:, :original.dimension].T)
            ],
            new_rows[
                np.lexsort(new_rows[:, :new_block.dimension].T)
            ],
        )

    # Check deterministic center selection and box construction.
    first_dimension = block_dimensions[0]
    first_samples = samples[:, :first_dimension]
    selected = pick_centers(
        first_samples, n_centers[0], np.random.default_rng(138)
    )
    expected = make_boxes(first_samples, selected)

    np.testing.assert_array_equal(blocks[0].rows(), expected.rows())

    for index, block in enumerate(blocks):
        print(f"Block {index + 1} boxes:", block.n_boxes)

    print("Retained probability:", event.retained_probability)
    print("Saved and reloaded boxes:", output_path)
    print(f"{label}: PASS")
    return event


def test_arak_event():
    """Keep the original default ARAK sample test for compatibility."""
    path = Path(__file__).parent / "ARAK_samples" / "event_0.npy"
    if not path.is_file():
        print(f"Skipping default ARAK event; file not found: {path}")
        return
    samples = load_event_samples(path, input_type="synthetic")
    with tempfile.TemporaryDirectory() as folder:
        test_event_samples(samples, "default ARAK event", Path(folder) / "event.boxes")


def test_configurable_block_dimensions():
    rng = np.random.default_rng(514)
    for dimensions in ((2, 3), (1, 3), (2, 4)):
        samples = rng.normal(size=(2000, sum(dimensions)))
        event = make_event_boxes(samples, n_centers=(3, 3),
                                 rng=np.random.default_rng(18), block_dimensions=dimensions, ntrial=(1, 1))
        assert tuple(b.dimension for b in event.blocks) == dimensions
        choices = np.random.default_rng(18)
        d0 = dimensions[0]
        for columns, actual in zip((samples[:, :d0], samples[:, d0:]), event.blocks):
            expected = make_boxes(columns, pick_centers(columns, 3, choices))
            np.testing.assert_array_equal(actual.rows(), expected.rows())
        np.testing.assert_allclose(event.retained_probability,
            np.prod([b.retained_probability for b in event.blocks]))


def test_retained_probability_warning():
    samples = np.array([[0., 0., 0., 0., 0.],
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [2., 2., 2., 2., 2.]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        event = make_event_boxes(samples, n_centers=(2, 1), rng=np.random.default_rng(4), ntrial=(1, 1))
    np.testing.assert_allclose(event.retained_probability, 2. / 3.)
    assert len(caught) == 1
    assert "more than 0.1% was discarded" in str(caught[0].message)



def test_trial_selection():
    """Compare the winner against separately constructed random grids."""
    samples = np.random.default_rng(8).normal(size=(1800, 5))
    for dimension, trials in ((3, 5), (2, 10)):
        block = samples[:, :dimension]
        choices = np.random.default_rng(138)
        candidates, centers = [], []
        for _ in range(trials):
            centers.append(pick_centers(block, 5, choices))
            candidates.append(make_boxes(block, centers[-1]))
        volumes = np.array([np.prod(b.upper-b.lower, axis=1).sum() for b in candidates])
        result = make_box_trials(block, 5, rng=np.random.default_rng(138))
        winner = int(np.argmin(volumes))
        assert result.selected_trial == winner
        np.testing.assert_array_equal(result.hypervolumes, volumes)
        np.testing.assert_array_equal(result.boxes.rows(), candidates[winner].rows())
        np.testing.assert_array_equal(result.centers, centers[winner])
        check_density_and_disjointness(result.boxes)
        custom = make_box_trials(block, 5, ntrial=3, rng=np.random.default_rng(138))
        assert len(custom.hypervolumes) == 3
        np.testing.assert_array_equal(custom.hypervolumes, volumes[:3])
        weighted = make_box_trials(block, 5, ntrial=3, rng=np.random.default_rng(138),
                                   weight_function=lambda x: x[:, 0]**2)
        midpoint = custom.boxes.lower+(custom.boxes.upper-custom.boxes.lower)/2
        np.testing.assert_array_equal(weighted.boxes.weight, midpoint[:, 0]**2)
    event = make_event_boxes(samples, rng=np.random.default_rng(138), ntrial=(2, 3))
    rng = np.random.default_rng(138)
    for block, dimensions, ncenters, trials in zip(event, (slice(0, 3), slice(3, 5)),
                                                 (4, 8), (2, 3)):
        expected = make_box_trials(samples[:, dimensions], ncenters, trials, rng)
        np.testing.assert_array_equal(block.rows(), expected.boxes.rows())


def test_degenerate_volume_is_visible():
    """Do not silently fix a retained box with zero width in one coordinate."""
    samples = np.array([[0., 0.], [0., 1.]])
    with np.errstate(divide='ignore', invalid='ignore'):
        boxes = make_boxes(samples, samples[:1])
        assert boxes.n_boxes == 1
        assert np.isinf(boxes.density[0])
        assert np.isnan(boxes.retained_probability)



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run box-generator correctness tests and optionally test ARAK, "
            "CAMP, and/or real GW Feather event samples."
        )
    )
    parser.add_argument(
        "--arak", type=Path, default=None,
        help="Path to an ARAK five-dimensional .npy sample file.",
    )
    parser.add_argument(
        "--camp", type=Path, default=None,
        help="Path to a CAMP five-dimensional .npy sample file.",
    )
    parser.add_argument(
        "--feather", type=Path, default=None,
        help="Path to a real event .feather file.",
    )
    parser.add_argument(
        "--feather-columns", default="chirp_mass,mass_ratio,chi_eff,ra,dec",
        help=(
            "Comma-separated Feather column names in this order: "
            "chirp mass, mass ratio, chi_eff, RA, declination."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "generated_boxes",
        help="Directory in which tested event box files will be saved.",
    )
    parser.add_argument(
        "--extract-o4",
        action="store_true",
        help=(
            "Extract the O4 archive configured by TAR_PATH into EXTRACT_DIR. "
            "This is optional and is not needed when testing explicit input files."
        ),
    )
    args = parser.parse_args()

    if args.extract_o4:
        extract_o4_data()

    # These are the repository's algorithm/regression tests. Keep them running
    # regardless of which event data files are supplied.
    test_degenerate_volume_is_visible()
    test_trial_selection()
    test_known_grid()
    test_against_explicit_full_grid()
    test_singleton_discard()
    test_retained_probability_warning()
    test_configurable_block_dimensions()

    requested_inputs = []
    if args.arak is not None:
        requested_inputs.append(("ARAK", args.arak, "synthetic"))
    if args.camp is not None:
        requested_inputs.append(("CAMP", args.camp, "synthetic"))
    if args.feather is not None:
        requested_inputs.append(("Feather", args.feather, "feather"))

    # Preserve the previous no-argument behavior: test the bundled ARAK sample
    # if it exists. If the user supplied any paths, test all of those paths.
    if not requested_inputs:
        default_arak = Path(__file__).parent / "ARAK_samples" / "event_0.npy"
        if default_arak.is_file():
            requested_inputs.append(("ARAK", default_arak, "synthetic"))
        else:
            print(
                "No event data path supplied and the default ARAK sample "
                "was not found. Only algorithm/regression tests were run."
            )


    feather_columns = [
        name.strip()
        for name in args.feather_columns.split(",")
        if name.strip()
    ]

    if len(feather_columns) not in (2, 5):
        parser.error(
            "--feather-columns must contain either 2 or 5 "
            "comma-separated names."
        )

    for label, path, input_type in requested_inputs:
        samples = load_event_samples(
            path,
            input_type=input_type,
            feather_columns=feather_columns,
        )
        output_path = args.output_dir / f"{label.lower()}_{path.stem}.boxes"
        test_event_samples(samples, f"{label} ({path.name})", output_path)

    print("\nAll requested box-generator checks passed.")

if __name__ == "__main__":
    main()
