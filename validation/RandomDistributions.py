"""Deterministic analytic distributions and a reproducible random factory.

The public model classes hold only realised parameters: ``ARAK`` is an
anchored residual-autoregressive kernel, ``ARAKMixture`` is a finite bounded
mixture, and ``ARAKMixtureProduct`` is an independent Cartesian product.
``ConnectedARAKMixtureProduct`` holds several related realised distributions.
``RandomDistribution`` owns the optional construction RNG and draws those
realised parameters from qualitative controls.
"""

from __future__ import annotations

from numbers import Integral
from typing import Mapping, Optional, Sequence, Union

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import ndtr, ndtri


ArrayLike = Union[np.ndarray, Sequence[float]]


class ARAK:
    """A fixed probability density built from an autoregressive flow.

    An ARAK has no construction-time randomness. Its origin is zero. Its
    shape comes from the affine matrix and, in two or more dimensions, the
    supplied residual bases. A one-dimensional ARAK uses a positive cubic
    spline times a Gaussian instead. The factory class is useful when these
    parameters should be drawn rather than specified.

    Parameters
    ----------
    dimension:
        Number of coordinates in the PDF. The origin is always zero. Move a
        kernel by setting a component centre in ``ARAKMixture``.
    scale:
        Invertible ``(dimension, dimension)`` matrix applied after the flow.
        It sets the orientation and local width of this unit kernel. The
        factory normally calibrates it so most marginal mass lies in the
        box from -0.5 to 0.5.
    epsilon:
        Multiplier of every residual update. Zero removes the nonlinear flow.
    basis_types, degrees, frequencies, directions, interaction_directions,
    coefficients:
        Exact parameters of the multivariate residual flow. Their common
        shape is ``(layers, dimension, n_basis)``. Direction arrays have one
        extra coordinate axis. Supply all six together or omit all six. If
        they are omitted, the multivariate PDF is an affine Gaussian.
    spline_knots, spline_heights:
        Exact parameters of the one-dimensional spline factor. Knots must
        increase and heights must be positive. Supply both or neither. If
        they are omitted, the one-dimensional PDF is an affine Gaussian.
    base_scale, support_scale, support_radius:
        Calibration records from ``RandomDistribution``. They are stored for
        inspection and reproducibility. They do not change the PDF; ``scale``
        is the matrix that the PDF uses.
    run_checks:
        Run numerical identity checks during construction. This is off by
        default because it costs extra PDF and transform evaluations.
    check_samples:
        Number of deterministic points used when ``run_checks`` is true.

    Sampling uses an optional seed for that sampling call only. It never
    changes the kernel parameters.
    """

    _POLYNOMIAL = 0
    _INTERACTION = 1
    _SINE = 2
    _COSINE = 3
    _TANH = 4

    def __init__(
        self,
        dimension: int,
        scale: Optional[ArrayLike] = None,
        *,
        epsilon: float = 0.1,
        basis_types: Optional[ArrayLike] = None,
        degrees: Optional[ArrayLike] = None,
        frequencies: Optional[ArrayLike] = None,
        directions: Optional[ArrayLike] = None,
        interaction_directions: Optional[ArrayLike] = None,
        coefficients: Optional[ArrayLike] = None,
        spline_knots: Optional[ArrayLike] = None,
        spline_heights: Optional[ArrayLike] = None,
        base_scale: Optional[ArrayLike] = None,
        support_scale: Optional[ArrayLike] = None,
        support_radius: Optional[ArrayLike] = None,
        run_checks: bool = False,
        check_samples: int = 1_001,
    ) -> None:
        if not isinstance(dimension, Integral) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        self.d = int(dimension)
        self.center = np.zeros(self.d)
        self.D = self.d  # Matches the naming used by the existing distributions.

        if scale is None:
            scale = np.eye(self.d)

        effective_scale = np.asarray(scale, dtype=float)
        if effective_scale.shape != (self.d, self.d):
            raise ValueError(f"scale must have shape ({self.d}, {self.d})")
        if not np.all(np.isfinite(effective_scale)):
            raise ValueError("scale must contain only finite values")
        self.base_scale = effective_scale.copy() if base_scale is None else np.asarray(base_scale, dtype=float).copy()
        if self.base_scale.shape != (self.d, self.d):
            raise ValueError(f"base_scale must have shape ({self.d}, {self.d})")
        self._set_affine_scale(effective_scale)

        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be a finite non-negative amplitude")
        if not isinstance(check_samples, Integral) or check_samples < 3:
            raise ValueError("check_samples must be an integer of at least three")
        self.epsilon = float(epsilon)
        self.run_checks = bool(run_checks)
        self.check_samples = int(check_samples)
        if self.d == 1:
            if (spline_knots is None) != (spline_heights is None):
                raise ValueError("spline_knots and spline_heights must be supplied together")
            if spline_knots is None:
                spline_knots, spline_heights = [-5.5, -2.75, 0.0, 2.75, 5.5], np.ones(5)
            self.spline_knots = np.asarray(spline_knots, dtype=float)
            self.spline_heights = np.asarray(spline_heights, dtype=float)
            if (self.spline_knots.ndim != 1 or self.spline_knots.size < 3
                    or self.spline_heights.shape != self.spline_knots.shape
                    or not np.all(np.isfinite(self.spline_knots)) or not np.all(np.diff(self.spline_knots) > 0.0)
                    or not np.all(np.isfinite(self.spline_heights)) or np.any(self.spline_heights <= 0.0)):
                raise ValueError("spline knots must increase and spline heights must be finite and positive")
            self.n_knots = self.spline_knots.size
            self._configure_spline_1d()
        else:
            supplied = (basis_types, degrees, frequencies, directions, interaction_directions, coefficients)
            if any(value is None for value in supplied) and any(value is not None for value in supplied):
                raise ValueError("supply every residual-basis array or none of them")
            if all(value is None for value in supplied):
                shape = (1, self.d, 1)
                basis_types, degrees = np.zeros(shape, dtype=np.int8), np.full(shape, 2, dtype=np.int16)
                frequencies = coefficients = np.zeros(shape)
                directions = interaction_directions = np.zeros(shape + (self.d,))
            self.basis_types = np.asarray(basis_types, dtype=np.int8)
            self.degrees = np.asarray(degrees, dtype=np.int16)
            self.frequencies = np.asarray(frequencies, dtype=float)
            self.coefficients = np.asarray(coefficients, dtype=float)
            self.directions = np.asarray(directions, dtype=float)
            self.interaction_directions = np.asarray(interaction_directions, dtype=float)
            if (self.basis_types.ndim != 3 or self.basis_types.shape[1] != self.d
                    or self.degrees.shape != self.basis_types.shape or self.frequencies.shape != self.basis_types.shape
                    or self.coefficients.shape != self.basis_types.shape
                    or self.directions.shape != self.basis_types.shape + (self.d,)
                    or self.interaction_directions.shape != self.directions.shape):
                raise ValueError("residual-basis arrays have inconsistent shapes")
            self.K, _, self.n_basis = self.basis_types.shape
            if (self.K < 1 or self.n_basis < 1 or np.any((self.basis_types < 0) | (self.basis_types > self._TANH))
                    or np.any(self.degrees < 2) or not np.all(np.isfinite(self.frequencies))
                    or not np.all(np.isfinite(self.coefficients)) or not np.all(np.isfinite(self.directions))
                    or not np.all(np.isfinite(self.interaction_directions))):
                raise ValueError("residual-basis arrays contain invalid layer or basis information")
            self.p = int(self.degrees.max())

        self.support_scale = np.ones(self.d) if support_scale is None else np.asarray(support_scale, dtype=float)
        self.support_radius = np.full(self.d, np.nan) if support_radius is None else np.asarray(support_radius, dtype=float)
        if self.support_scale.shape != (self.d,) or self.support_radius.shape != (self.d,):
            raise ValueError("support_scale and support_radius must have shape (d,)")
        if not np.all(np.isfinite(self.support_scale)) or np.any(self.support_scale <= 0.0):
            raise ValueError("support_scale must be finite and positive")
        if support_radius is not None and (not np.all(np.isfinite(self.support_radius)) or np.any(self.support_radius <= 0.0)):
            raise ValueError("support_radius must be finite and positive when supplied")

        self.anchor = self.center.copy()

        if self.run_checks:
            self._run_numerical_checks()

    def _set_affine_scale(self, scale: np.ndarray) -> None:
        """Set the observed affine map and cache its inverse/Jacobian."""
        determinant_sign, log_abs_det = np.linalg.slogdet(scale)
        if determinant_sign == 0 or not np.isfinite(log_abs_det):
            raise ValueError("scale must be invertible")
        self.scale = scale
        self.log_abs_det_L = log_abs_det
        # A cached inverse turns affine inversion of an entire batch into one
        # matrix product rather than a solve per sample.
        self._L_inv = np.linalg.solve(scale, np.eye(self.d))
        self.local_scale = scale.copy()
        self.local_covariance = scale @ scale.T

    def _run_numerical_checks(self) -> None:
        """Run opt-in numerical identities; never called in the fast path."""
        latent = np.random.default_rng(0).normal(size=(self.check_samples, self.d))
        recovered = self.inverse_transform(self.transform(latent))
        if not np.allclose(recovered, latent, rtol=1e-9, atol=1e-9):
            raise AssertionError("transform and inverse_transform disagree")

        if self.d != 1:
            return

        spline_coordinate = np.linspace(-8.0, 8.0, self.check_samples)
        y = self._spline_coordinate_scale * (spline_coordinate - self._spline_median)
        x = self.center[0] + self.scale[0, 0] * y
        density = self.pdf(x[:, None])
        numerical_cdf = cumulative_trapezoid(density, x, initial=0.0)
        if not np.isclose(trapezoid(density, x), 1.0, atol=2e-6):
            raise AssertionError("1D spline PDF does not numerically normalize")
        if not np.allclose(numerical_cdf, self._spline_cdf(spline_coordinate), atol=3e-6):
            raise AssertionError("1D spline analytical and numerical CDFs disagree")

    def _configure_spline_1d(self) -> None:
        """Cache the exact spline-integral representation."""
        self._set_spline(self.spline_heights)
        self._spline_median = float(self._spline_ppf(np.array([0.5]))[0])
        self._spline_coordinate_scale = float(
            self._standard_normal_pdf(np.array([self._spline_median]))[0]
            * self._spline_weight(np.array([self._spline_median]))[0]
            / (self._spline_normalization * self._standard_normal_pdf(np.array([0.0]))[0])
        )

    @staticmethod
    def _standard_normal_pdf(values: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * values * values) / np.sqrt(2.0 * np.pi)

    def _set_spline(self, heights: np.ndarray) -> None:
        """Build the cubic segments and their exact Gaussian integrals once."""
        self._spline_interpolator = PchipInterpolator(self.spline_knots, heights, extrapolate=False)
        local = self._spline_interpolator.c
        origin = self.spline_knots[:-1]
        cubic, quadratic, linear, constant = local
        self._spline_poly = np.stack(
            (
                constant - origin * linear + origin**2 * quadratic - origin**3 * cubic,
                linear - 2.0 * origin * quadratic + 3.0 * origin**2 * cubic,
                quadratic - 3.0 * origin * cubic,
                cubic,
            ),
            axis=1,
        )
        self._spline_correction_poly = self._spline_poly.copy()
        self._spline_correction_poly[:, 0] -= 1.0
        moments = self._gaussian_moments(self.spline_knots[:-1], self.spline_knots[1:])
        self._spline_normalization = float(1.0 + np.einsum("ij,ij->", self._spline_correction_poly, moments))

    def _gaussian_moments(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Integrals of x**k * phi(x), k=0..3, between matching bounds."""
        phi_lower = self._standard_normal_pdf(lower)
        phi_upper = self._standard_normal_pdf(upper)
        moment_0 = ndtr(upper) - ndtr(lower)
        moment_1 = phi_lower - phi_upper
        moment_2 = moment_0 + lower * phi_lower - upper * phi_upper
        moment_3 = 2.0 * moment_1 + lower**2 * phi_lower - upper**2 * phi_upper
        return np.stack((moment_0, moment_1, moment_2, moment_3), axis=-1)

    def _spline_weight(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        weights = np.ones_like(values)
        inside = (values >= self.spline_knots[0]) & (values <= self.spline_knots[-1])
        weights[inside] = self._spline_interpolator(values[inside])
        return weights

    def _spline_cdf(self, values: np.ndarray) -> np.ndarray:
        """Analytic CDF of the normalized spline-modulated Gaussian."""
        values = np.asarray(values, dtype=float)
        cumulative = ndtr(values)
        knots = self.spline_knots
        for left, right, correction_polynomial in zip(knots[:-1], knots[1:], self._spline_correction_poly):
            upper = np.clip(values, left, right)
            correction = np.einsum("j,...j->...", correction_polynomial, self._gaussian_moments(left, upper))
            cumulative += correction
        return cumulative / self._spline_normalization

    def _spline_ppf(self, probabilities: np.ndarray) -> np.ndarray:
        """Invert the analytic spline CDF with a batched bisection solve."""
        probabilities = np.asarray(probabilities, dtype=float)
        lower = np.full_like(probabilities, -10.0)
        upper = np.full_like(probabilities, 10.0)
        for _ in range(52):
            midpoint = 0.5 * (lower + upper)
            lower = np.where(self._spline_cdf(midpoint) < probabilities, midpoint, lower)
            upper = np.where(self._spline_cdf(midpoint) >= probabilities, midpoint, upper)
        return 0.5 * (lower + upper)

    def _as_batch(self, values: ArrayLike, name: str) -> tuple[np.ndarray, bool]:
        array = np.asarray(values, dtype=float)
        scalar_input = array.ndim == 1
        if scalar_input:
            if array.shape != (self.d,):
                raise ValueError(f"{name} must have shape ({self.d},) or (n, {self.d})")
            array = array[None, :]
        elif array.ndim != 2 or array.shape[1] != self.d:
            raise ValueError(f"{name} must have shape ({self.d},) or (n, {self.d})")
        return array, scalar_input

    def _basis_values(self, previous: np.ndarray, layer: int, coordinate: int) -> np.ndarray:
        """Evaluate all anchored bases for one triangular output coordinate."""
        directions = self.directions[layer, coordinate, :, :coordinate]
        projection = previous @ directions.T
        saturated_projection = np.tanh(projection)
        kind = self.basis_types[layer, coordinate]
        values = np.empty_like(projection)

        polynomial = kind == self._POLYNOMIAL
        if np.any(polynomial):
            values[:, polynomial] = saturated_projection[:, polynomial] ** self.degrees[layer, coordinate, polynomial]

        interaction = kind == self._INTERACTION
        if np.any(interaction):
            second_projection = previous @ self.interaction_directions[layer, coordinate, interaction, :coordinate].T
            values[:, interaction] = saturated_projection[:, interaction] * np.tanh(second_projection)

        frequency = self.frequencies[layer, coordinate]
        sine = kind == self._SINE
        if np.any(sine):
            argument = frequency[sine] * saturated_projection[:, sine]
            values[:, sine] = np.sin(argument) - argument

        cosine = kind == self._COSINE
        if np.any(cosine):
            argument = frequency[cosine] * saturated_projection[:, cosine]
            values[:, cosine] = np.cos(argument) - 1.0

        tanh_residual = kind == self._TANH
        if np.any(tanh_residual):
            argument = frequency[tanh_residual] * saturated_projection[:, tanh_residual]
            values[:, tanh_residual] = np.tanh(argument) - argument
        return values

    def _forward_layer(self, values: np.ndarray, layer: int) -> np.ndarray:
        source = values
        transformed = source.copy()
        for coordinate in range(1, self.d):
            bases = self._basis_values(source[:, :coordinate], layer, coordinate)
            transformed[:, coordinate] += self.epsilon * (bases @ self.coefficients[layer, coordinate])
        return transformed

    def _inverse_layer(self, values: np.ndarray, layer: int) -> np.ndarray:
        source = np.empty_like(values)
        source[:, 0] = values[:, 0]
        for coordinate in range(1, self.d):
            bases = self._basis_values(source[:, :coordinate], layer, coordinate)
            source[:, coordinate] = values[:, coordinate] - self.epsilon * (bases @ self.coefficients[layer, coordinate])
        return source

    def _deform_batch(self, values: np.ndarray) -> np.ndarray:
        """Apply the nonlinear deformation to an already-validated batch."""
        if self.d == 1:
            return (
                self._spline_coordinate_scale
                * (self._spline_ppf(ndtr(values[:, 0])) - self._spline_median)
            )[:, None]
        for layer in range(self.K):
            values = self._forward_layer(values, layer)
        return values

    def _inverse_deform_batch(self, values: np.ndarray) -> np.ndarray:
        """Invert the nonlinear deformation for an already-validated batch."""
        if self.d == 1:
            spline_coordinate = self._spline_median + values[:, 0] / self._spline_coordinate_scale
            return ndtri(self._spline_cdf(spline_coordinate))[:, None]
        for layer in range(self.K - 1, -1, -1):
            values = self._inverse_layer(values, layer)
        return values

    def deform(self, z: ArrayLike) -> np.ndarray:
        """Apply the nonlinear deformation ``F(z)``."""
        values, scalar_input = self._as_batch(z, "z")
        transformed = self._deform_batch(values)
        return transformed[0] if scalar_input else transformed

    def inverse_deform(self, y: ArrayLike) -> np.ndarray:
        """Apply the exact triangular inverse ``F^{-1}(y)``."""
        values, scalar_input = self._as_batch(y, "y")
        transformed = self._inverse_deform_batch(values)
        return transformed[0] if scalar_input else transformed

    def transform(self, z: ArrayLike) -> np.ndarray:
        """Map latent standard-normal points to unbounded observations."""
        z_batch, scalar_input = self._as_batch(z, "z")
        y = self._deform_batch(z_batch)
        x = self.center + y @ self.scale.T
        return x[0] if scalar_input else x

    def _invert_observation_batch(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Undo the affine map for an observation batch."""
        affine_coordinate = (values - self.center) @ self._L_inv.T
        return affine_coordinate, np.zeros(len(values)), np.ones(len(values), dtype=bool)

    def inverse_transform(self, x: ArrayLike) -> np.ndarray:
        """Map observations to latent standard-normal coordinates.

        This ARAK is unbounded. Global bounds, when needed, belong to an
        ``ARAKMixture``.
        """
        x_batch, scalar_input = self._as_batch(x, "x")
        y, _, _ = self._invert_observation_batch(x_batch)
        z = self._inverse_deform_batch(y)
        return z[0] if scalar_input else z

    def logpdf(self, x: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate the normalized log density at one point or a batch.

        The ARAK itself is unbounded. A bounded mixture applies the boundary
        treatment and returns ``-inf`` outside its global bounds.
        """
        x_batch, scalar_input = self._as_batch(x, "x")
        y, inverse_log_det, inside = self._invert_observation_batch(x_batch)
        if self.d == 1:
            standardized = self._spline_median + y[:, 0] / self._spline_coordinate_scale
            log_density = -0.5 * standardized * standardized - 0.5 * np.log(2.0 * np.pi)
            log_density += np.log(self._spline_weight(standardized)) - np.log(self._spline_normalization)
            log_density -= np.log(self._spline_coordinate_scale)
            log_density -= self.log_abs_det_L
            log_density += inverse_log_det
            log_density[~inside] = -np.inf
            return float(log_density[0]) if scalar_input else log_density
        z_batch = self._inverse_deform_batch(y)
        log_density = -0.5 * np.einsum("ij,ij->i", z_batch, z_batch)
        log_density -= 0.5 * self.d * np.log(2.0 * np.pi) + self.log_abs_det_L
        log_density += inverse_log_det
        log_density[~inside] = -np.inf
        return float(log_density[0]) if scalar_input else log_density

    def pdf(self, x: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate the normalized density at one point or a batch."""
        return np.exp(self.logpdf(x))

    def rvs(
        self,
        size: int = 1,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> np.ndarray:
        """Draw samples from this deterministic kernel.

        Parameters
        ----------
        size:
            Non-negative number of rows in the returned ``(size, dimension)``
            array.
        random_state:
            Optional seed or generator for this draw only.  It changes samples
            but never changes the kernel PDF or its realized parameters.
        """
        if not isinstance(size, Integral) or size < 0:
            raise ValueError("size must be a non-negative integer")
        generator = np.random.default_rng(random_state)
        if self.d == 1:
            uniforms = generator.uniform(0.0, 1.0, size=int(size))
            # The spline CDF is analytical.  Form its interpolation grid only
            # for this draw, rather than retaining an inverse-CDF cache on
            # every kernel instance.
            grid = np.linspace(-10.0, 10.0, max(32_769, 2_048 * (self.n_knots - 1)))
            cdf = self._spline_cdf(grid)
            cdf[0], cdf[-1] = 0.0, 1.0
            strictly_increasing = np.r_[True, np.diff(cdf) > 0.0]
            standardized = np.interp(uniforms, cdf[strictly_increasing], grid[strictly_increasing])
            y = self._spline_coordinate_scale * (standardized - self._spline_median)
            return self.center + y[:, None] @ self.scale.T
        return self.transform(generator.normal(size=(int(size), self.d)))

    sample = rvs


class ARAKMixture:
    """A fixed weighted mixture of ARAK kernels inside global bounds.

    The mixture first evaluates unit kernels at their shifted and scaled raw
    coordinates, then sums them using the given weights. ``boundary_treatment``
    chooses either an analytical global compactification or a hard truncation
    with a supplied empirical normalization estimate.

    Parameters
    ----------
    component_centers:
        Exact raw-space centres, shape ``(n_components, dimension)``. These
        positions are set before the global compactification.
    component_scales:
        Positive raw-space scale of every mode and coordinate, with the same
        shape as ``component_centers``. These scales set mode size.
    bounds:
        Final absolute lower and upper limits, shape ``(dimension, 2)``. The
        entire raw mixture is mapped into this box.
    components:
        One realised origin-centred ARAK per mode. All components must have
        the same dimension.
    weights:
        Positive mode probabilities that sum to one. For one component this
        defaults to one. For several components, provide the weights.
    arak_parameters:
        Exact ARAK constructor parameters for the one-component shortcut.
        If ``components`` is omitted, this builds one deterministic ARAK.
    boundary_treatment:
        ``'compactify'`` applies the analytical global map. ``'truncate'``
        sets the PDF to zero outside ``bounds`` and divides the raw PDF by
        ``truncation_mass`` inside those bounds.
    boundary_hardness, compactification:
        Positive transition width and ``'tanh'``/``'sigmoid'`` profile used
        only for ``boundary_treatment='compactify'``.
    truncation_mass, truncation_mass_error, truncation_samples:
        Empirical probability that a raw draw from the final retained mixture
        lies inside ``bounds``, its binomial standard error, and the number of
        final draws used to estimate it. These are required for truncation.

    Sampling can take a seed, but the mixture itself has no random state.
    """

    def __init__(
        self,
        *,
        component_centers: ArrayLike,
        component_scales: ArrayLike,
        bounds: ArrayLike,
        components: Optional[Sequence[ARAK]] = None,
        weights: Optional[ArrayLike] = None,
        arak_parameters: Optional[Mapping[str, object]] = None,
        boundary_treatment: str = "compactify",
        boundary_hardness: Optional[Union[float, ArrayLike]] = None,
        compactification: str = "tanh",
        truncation_mass: Optional[float] = None,
        truncation_mass_error: Optional[float] = None,
        truncation_samples: Optional[int] = None,
    ) -> None:
        """Create a mixture from fully realised component parameters.

        ``weights`` must already be normalized, ``component_centers`` has
        shape ``(m, d)``, and positive ``component_scales`` has the same
        shape.  Random weights, scales, centres, and kernels belong in
        :meth:`RandomDistribution.mixture`.
        """
        centers = np.asarray(component_centers, dtype=float)
        if centers.ndim != 2 or centers.shape[0] < 1:
            raise ValueError("component_centers must have shape (n_components, d)")
        if components is None:
            parameters = {} if arak_parameters is None else dict(arak_parameters)
            parameters.setdefault("dimension", centers.shape[1])
            self.components = (ARAK(**parameters),)
        else:
            self.components = tuple(components)
        if not self.components:
            raise ValueError("components must be non-empty")
        if not all(isinstance(component, ARAK) for component in self.components):
            raise TypeError("components must contain only ARAK instances")
        self.d = self.components[0].d
        if any(component.d != self.d for component in self.components):
            raise ValueError("all components must have the same dimension")
        self.D = self.d
        self.n_components = len(self.components)
        self.bounds = self._validate_bounds(bounds)
        if weights is None:
            if self.n_components != 1:
                raise ValueError("weights are required for a multi-component mixture")
            weights = [1.0]
        self.weights = np.asarray(weights, dtype=float)
        self.component_centers = centers
        self.component_scales = np.asarray(component_scales, dtype=float)
        if self.weights.shape != (self.n_components,) or not np.all(np.isfinite(self.weights)) or np.any(self.weights <= 0.0):
            raise ValueError("weights must be finite and positive with shape (n_components,)")
        if not np.isclose(self.weights.sum(), 1.0, rtol=1e-12, atol=1e-14):
            raise ValueError("weights must sum to one")
        if self.component_centers.shape != (self.n_components, self.d) or not np.all(np.isfinite(self.component_centers)):
            raise ValueError("component_centers must be finite with shape (n_components, d)")
        if (self.component_scales.shape != (self.n_components, self.d)
                or not np.all(np.isfinite(self.component_scales)) or np.any(self.component_scales <= 0.0)):
            raise ValueError("component_scales must be positive with shape (n_components, d)")
        self.center = np.average(self.component_centers, axis=0, weights=self.weights)
        if boundary_treatment not in {"compactify", "truncate"}:
            raise ValueError("boundary_treatment must be 'compactify' or 'truncate'")
        self.boundary_treatment = boundary_treatment
        if self.boundary_treatment == "compactify":
            self.compactification = self._validate_compactification(compactification)
            self.boundary_hardness = self._validate_boundary_hardness(boundary_hardness)
            self._compact_softness = self.boundary_hardness / 2.0 if self.compactification == "tanh" else self.boundary_hardness
            self.truncation_mass = None
            self.truncation_mass_error = None
            self.truncation_samples = None
            self.truncation_correction = None
            self.truncation_correction_error = None
        else:
            if truncation_mass is None or truncation_mass_error is None or truncation_samples is None:
                raise ValueError("truncation requires mass, mass error, and sample count")
            if not 0.0 < truncation_mass <= 1.0 or not np.isfinite(truncation_mass):
                raise ValueError("truncation_mass must lie in (0, 1]")
            if truncation_mass_error < 0.0 or not np.isfinite(truncation_mass_error):
                raise ValueError("truncation_mass_error must be finite and non-negative")
            if not isinstance(truncation_samples, Integral) or truncation_samples < 1:
                raise ValueError("truncation_samples must be a positive integer")
            self.compactification = None
            self.boundary_hardness = None
            self.truncation_mass = float(truncation_mass)
            self.truncation_mass_error = float(truncation_mass_error)
            self.truncation_samples = int(truncation_samples)
            self.truncation_correction = 1.0 / self.truncation_mass
            self.truncation_correction_error = self.truncation_mass_error / self.truncation_mass**2

    def _validate_bounds(self, bounds: ArrayLike) -> np.ndarray:
        array = np.asarray(bounds, dtype=float)
        if array.shape != (self.d, 2):
            raise ValueError(f"bounds must have shape ({self.d}, 2)")
        if not np.all(np.isfinite(array)) or np.any(array[:, 0] >= array[:, 1]):
            raise ValueError("each bound must be finite and satisfy lower < upper")
        return array

    def _validate_boundary_hardness(self, boundary_hardness: Optional[Union[float, ArrayLike]]) -> np.ndarray:
        if boundary_hardness is None:
            array = np.full(self.d, 0.1)
        else:
            array = np.asarray(boundary_hardness, dtype=float)
            if array.ndim == 0:
                array = np.full(self.d, float(array))
        if array.shape != (self.d,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
            raise ValueError(f"boundary_hardness must be positive with shape ({self.d},)")
        return array

    @staticmethod
    def _validate_compactification(compactification: str) -> str:
        if compactification == "logit":
            return "sigmoid"
        if compactification not in {"tanh", "sigmoid"}:
            raise ValueError("compactification must be 'tanh', 'sigmoid', or 'logit'")
        return compactification

    @staticmethod
    def _softplus(values: np.ndarray) -> np.ndarray:
        return np.logaddexp(0.0, values)

    @staticmethod
    def _inverse_softplus(values: np.ndarray) -> np.ndarray:
        return values + np.log(-np.expm1(-values))

    @staticmethod
    def _log_sigmoid_from_softplus(values: np.ndarray) -> np.ndarray:
        return np.log(-np.expm1(-values))

    def _as_batch(self, values: ArrayLike, name: str) -> tuple[np.ndarray, bool]:
        array = np.asarray(values, dtype=float)
        scalar_input = array.ndim == 1
        if scalar_input:
            if array.shape != (self.d,):
                raise ValueError(f"{name} must have shape ({self.d},) or (n, {self.d})")
            array = array[None, :]
        elif array.ndim != 2 or array.shape[1] != self.d:
            raise ValueError(f"{name} must have shape ({self.d},) or (n, {self.d})")
        return array, scalar_input

    def _compactify_global(self, values: np.ndarray) -> np.ndarray:
        """Map raw mixture coordinates into the absolute global box."""
        lower, upper = self.bounds.T
        softness = self._compact_softness
        transformed = lower + softness * self._softplus((values - lower) / softness)
        capped = upper - softness * self._softplus((upper - transformed) / softness)
        lower_cap = upper - softness * self._softplus((upper - lower) / softness)
        factor = (upper - lower) / (upper - lower_cap)
        return lower + factor * (capped - lower_cap)

    def _invert_global_compactification(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Invert the global map and return inverse log-Jacobian and support mask."""
        lower, upper = self.bounds.T
        softness = self._compact_softness
        inside = np.all((values > lower) & (values < upper), axis=1)
        lower_cap = upper - softness * self._softplus((upper - lower) / softness)
        factor = (upper - lower) / (upper - lower_cap)
        capped = lower_cap + (values - lower) / factor
        valid = (capped > lower_cap) & (capped < upper)
        safe_capped = np.where(valid, capped, 0.5 * (lower_cap + upper))

        upper_softplus = (upper - safe_capped) / softness
        upper_argument = self._inverse_softplus(upper_softplus)
        after_lower = upper - softness * upper_argument
        lower_softplus = (after_lower - lower) / softness
        lower_argument = self._inverse_softplus(lower_softplus)
        preimage = lower + softness * lower_argument

        inverse_log_det = -np.sum(np.log(factor))
        inverse_log_det -= np.sum(self._log_sigmoid_from_softplus(upper_softplus), axis=1)
        inverse_log_det -= np.sum(self._log_sigmoid_from_softplus(lower_softplus), axis=1)
        return preimage, inverse_log_det, inside

    def _raw_logpdf(self, values: np.ndarray) -> np.ndarray:
        """Evaluate the unbounded mixture before global compactification."""
        return np.logaddexp.reduce(self._raw_component_logpdf(values), axis=1)

    def _raw_component_logpdf(self, values: np.ndarray) -> np.ndarray:
        """Evaluate unbounded weighted component log PDFs, one column per mode."""
        log_density = np.empty((len(values), self.n_components))
        for index, (weight, center, scale, component) in enumerate(
            zip(self.weights, self.component_centers, self.component_scales, self.components)
        ):
            local = (values - center) / scale
            log_density[:, index] = np.log(weight) - np.log(scale).sum() + component.logpdf(local)
        return log_density

    def component_logpdf(self, x: ArrayLike) -> np.ndarray:
        """Evaluate weighted observed-PDF contributions, one column per mode.

        Contributions include the outer mode scale and the selected global
        boundary treatment. They sum to :meth:`pdf` at every point.
        """
        values, scalar_input = self._as_batch(x, "x")
        if self.boundary_treatment == "compactify":
            preimage, inverse_log_det, inside = self._invert_global_compactification(values)
            log_density = self._raw_component_logpdf(preimage) + inverse_log_det[:, None]
        else:
            inside = np.all((values > self.bounds[:, 0]) & (values < self.bounds[:, 1]), axis=1)
            log_density = self._raw_component_logpdf(values) - np.log(self.truncation_mass)
        log_density[~inside] = -np.inf
        return log_density[0] if scalar_input else log_density

    def component_pdf(self, x: ArrayLike) -> np.ndarray:
        """Evaluate weighted observed-PDF contributions, one final axis per mode."""
        return np.exp(self.component_logpdf(x))

    def logpdf(self, x: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate the normalized PDF after the selected boundary treatment."""
        values, scalar_input = self._as_batch(x, "x")
        if self.boundary_treatment == "compactify":
            preimage, inverse_log_det, inside = self._invert_global_compactification(values)
            log_density = self._raw_logpdf(preimage) + inverse_log_det
        else:
            inside = np.all((values > self.bounds[:, 0]) & (values < self.bounds[:, 1]), axis=1)
            log_density = self._raw_logpdf(values) - np.log(self.truncation_mass)
        log_density[~inside] = -np.inf
        return float(log_density[0]) if scalar_input else log_density

    def pdf(self, x: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate the exact normalized PDF of the finite mixture."""
        return np.exp(self.logpdf(x))

    def rvs(
        self,
        size: int = 1,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> np.ndarray:
        """Draw samples from the bounded mixture.

        ``size`` is the number of ``(d,)`` samples.  ``random_state`` affects
        only this draw, including mode assignments and component samples; it
        does not alter weights, centres, scales, components, or bounds.
        """
        if not isinstance(size, Integral) or size < 0:
            raise ValueError("size must be a non-negative integer")
        generator = np.random.default_rng(random_state)
        if self.boundary_treatment == "compactify":
            return self._compactify_global(self._rvs_raw(int(size), generator))
        accepted = []
        remaining = int(size)
        while remaining:
            draw_count = max(remaining, int(np.ceil(1.2 * remaining / self.truncation_mass)))
            raw = self._rvs_raw(draw_count, generator)
            raw = raw[np.all((raw >= self.bounds[:, 0]) & (raw <= self.bounds[:, 1]), axis=1)]
            accepted.append(raw[:remaining])
            remaining -= min(remaining, len(raw))
        return np.concatenate(accepted, axis=0) if accepted else np.empty((0, self.d))

    def _rvs_raw(self, size: int, generator: np.random.Generator) -> np.ndarray:
        """Draw from the unbounded scaled-and-shifted mixture."""
        component_index = generator.choice(self.n_components, size=int(size), p=self.weights)
        samples = np.empty((int(size), self.d))
        for index, component in enumerate(self.components):
            selected = component_index == index
            count = selected.sum()
            if count:
                local = component.rvs(
                    count,
                    random_state=int(generator.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)),
                )
                samples[selected] = self.component_centers[index] + self.component_scales[index] * local
        return samples

    sample = rvs


class ARAKMixtureProduct:
    """An independent product of fixed ARAK mixtures.

    Parameters
    ----------
    mixtures:
        Non-empty sequence of realised mixtures. Their coordinate blocks are
        joined in this order. The joint PDF is the product of block PDFs and
        the final bounds are the joined block bounds.

    Sampling draws each block independently. A sampling seed affects that
    draw only and does not change any mixture.
    """

    def __init__(self, mixtures: Sequence[ARAKMixture]) -> None:
        if not isinstance(mixtures, Sequence) or isinstance(mixtures, (str, bytes)) or not mixtures:
            raise TypeError("mixtures must be a non-empty sequence of ARAKMixture instances")
        if not all(isinstance(mixture, ARAKMixture) for mixture in mixtures):
            raise TypeError("mixtures must contain only ARAKMixture instances")
        self.mixtures = tuple(mixtures)
        self.d = sum(mixture.d for mixture in self.mixtures)
        self.D = self.d
        self.bounds = np.concatenate([mixture.bounds for mixture in self.mixtures], axis=0)
        self._offsets = np.concatenate(([0], np.cumsum([mixture.d for mixture in self.mixtures])))

    def _as_batch(self, values: ArrayLike, name: str) -> tuple[np.ndarray, bool]:
        array = np.asarray(values, dtype=float)
        scalar_input = array.ndim == 1
        if scalar_input:
            if array.shape != (self.d,):
                raise ValueError(f"{name} must have shape ({self.d},) or (n, {self.d})")
            array = array[None, :]
        elif array.ndim != 2 or array.shape[1] != self.d:
            raise ValueError(f"{name} must have shape ({self.d},) or (n, {self.d})")
        return array, scalar_input

    def logpdf(self, x: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate the exact log PDF of the independent product."""
        values, scalar_input = self._as_batch(x, "x")
        log_density = np.zeros(len(values))
        for start, stop, mixture in zip(self._offsets[:-1], self._offsets[1:], self.mixtures):
            log_density += mixture.logpdf(values[:, start:stop])
        return float(log_density[0]) if scalar_input else log_density

    def pdf(self, x: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate the exact normalized PDF of the independent product."""
        return np.exp(self.logpdf(x))

    def rvs(
        self,
        size: int = 1,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> np.ndarray:
        """Draw independent product samples.

        ``size`` gives the number of rows.  ``random_state`` only seeds this
        sampling call; it is split between deterministic mixture blocks and
        cannot change the realized product PDF.
        """
        if not isinstance(size, Integral) or size < 0:
            raise ValueError("size must be a non-negative integer")
        generator = np.random.default_rng(random_state)
        blocks = [
            mixture.rvs(
                int(size),
                random_state=int(generator.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)),
            )
            for mixture in self.mixtures
        ]
        return np.concatenate(blocks, axis=1)

    sample = rvs


class ConnectedARAKMixtureProduct:
    """A group of related, independently realised distributions.

    This is the result of :meth:`RandomDistribution.camp`.  It is a small
    container rather than one PDF: each item in ``distributions`` is its own
    normalized :class:`ARAKMixtureProduct`. The distributions use the same input
    settings, but their random kernels and modes are different.

    Parameters
    ----------
    reference_distribution:
        The temporary distribution from which connection anchors were drawn. It is
        kept so the construction can be inspected, but it is not one of the
        final distributions.
    anchors:
        ``(n_distributions, dimension)`` points drawn from
        ``reference_distribution``. Anchor ``i`` is the reference centre
        supplied to final distribution ``i``.
    distributions:
        Final independently realised distributions. Their order matches
        ``anchors``.

    :class:`ARAKMixtureProduct` with its usual ``pdf`` and ``rvs`` methods.
    Iteration and indexing return final distributions, so ``group[i]`` is an
    :class:`ARAKMixtureProduct` with its usual ``pdf`` and ``rvs`` methods.
    :class:`ARAKMixtureProduct` with its usual ``pdf`` and ``rvs`` methods.
    """

    def __init__(
        self,
        reference_distribution: ARAKMixtureProduct,
        anchors: ArrayLike,
        distributions: Sequence[ARAKMixtureProduct],
    ) -> None:
        if not isinstance(reference_distribution, ARAKMixtureProduct):
            raise TypeError("reference_distribution must be an ARAKMixtureProduct")
        if not isinstance(distributions, Sequence) or isinstance(distributions, (str, bytes)):
            raise TypeError("distributions must be a sequence of ARAKMixtureProduct instances")
        if not all(isinstance(distribution, ARAKMixtureProduct) for distribution in distributions):
            raise TypeError("distributions must contain only ARAKMixtureProduct instances")
        locations = np.asarray(anchors, dtype=float)
        if locations.ndim != 2 or locations.shape != (len(distributions), reference_distribution.d):
            raise ValueError("anchors must have shape (len(distributions), reference_distribution.d)")
        if not np.all(np.isfinite(locations)):
            raise ValueError("anchors must be finite")
        if any(distribution.d != reference_distribution.d for distribution in distributions):
            raise ValueError("every final distribution must have the reference distribution dimension")
        self.reference_distribution = reference_distribution
        self.anchors = locations
        self.distributions = tuple(distributions)
        self.d = reference_distribution.d
        self.D = self.d

    def __len__(self) -> int:
        """Return the number of final connected distributions."""
        return len(self.distributions)

    def __getitem__(self, index: int) -> ARAKMixtureProduct:
        """Return one final distribution by index."""
        return self.distributions[index]

    def __iter__(self):
        """Iterate over the final distributions in anchor order."""
        return iter(self.distributions)


class RandomDistribution:
    """Build random ARAKs, mixtures, and products from qualitative settings.

    This class is where random choices live. A returned ARAK, mixture, or
    product is fixed. Reusing the same factory seed and making the same calls
    gives the same realised PDFs.

    The ``arak`` method draws spline or residual-flow parameters and can
    calibrate the result to a unit box. The ``mixture`` method draws mode
    shapes, later-mode weights and scales, and later-mode centres. It then
    returns a deterministic mixture with the requested global bounds. The
    ``product`` method joins independently generated mixture blocks. The
    ``camp`` method makes several fresh distributions around samples from one
    temporary reference distribution.
    """

    def __init__(self, random_state: Optional[Union[int, np.random.Generator]] = None) -> None:
        """Create a random-distribution factory.

        Parameters
        ----------
        random_state:
            Optional integer seed or ``numpy.random.Generator``.  The factory
            consumes this stream while realizing parameters.  Equal seeds and
            equal method-call order produce equal PDFs.  ``None`` obtains a
            fresh generator, so each construction is different.  This state
            is never stored by returned ARAKs, mixtures, or products.
        """
        self.rng = np.random.default_rng(random_state)

    def _unit_vectors(self, count: int, dimension: int) -> np.ndarray:
        vectors = self.rng.normal(size=(count, dimension))
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def _aligned_scale(self, d: int, alignment: float) -> np.ndarray:
        if not np.isfinite(alignment) or not 0.0 <= alignment <= 1.0:
            raise ValueError("alignment must be a finite number in [0, 1]")
        scale = np.eye(d)
        if alignment == 0.0:
            return scale
        if d == 1:
            raise ValueError("alignment has no meaning for a one-dimensional kernel")
        row, column = np.tril_indices(d, k=-1)
        direction = self.rng.normal(size=len(row))
        direction /= np.linalg.norm(direction)
        scale[row, column] = 0.75 * alignment * np.sqrt(d - 1) * direction
        return scale

    def _random_spline(self, n_knots: int, knot_jitter: float, coefficient_scale: float) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(n_knots, Integral) or n_knots < 5 or n_knots % 2 == 0:
            raise ValueError("n_knots must be an odd integer of at least 5")
        if not np.isfinite(knot_jitter) or knot_jitter < 0.0:
            raise ValueError("knot_jitter must be finite and non-negative")
        if not np.isfinite(coefficient_scale) or coefficient_scale < 0.0:
            raise ValueError("coefficient_scale must be finite and non-negative")
        half, tail = (n_knots - 1) // 2, 5.5

        def side() -> np.ndarray:
            values = np.linspace(0.35, tail, half)
            if half > 1 and knot_jitter:
                values[:-1] = np.clip(
                    values[:-1] + self.rng.normal(0.0, knot_jitter * tail / half, size=half - 1),
                    0.08, tail - 0.08,
                )
                values[:-1].sort()
            return values

        negative, positive = side(), side()
        knots = np.concatenate((-negative[::-1], [0.0], positive))
        left = np.clip(np.exp(self.rng.normal(0.0, coefficient_scale, size=half - 1)), 0.12, 4.0)
        right = np.clip(np.exp(self.rng.normal(0.0, coefficient_scale, size=half - 1)), 0.12, 4.0)
        middle = np.clip(np.exp(self.rng.normal(0.0, coefficient_scale)), 0.12, 4.0)
        return knots, np.concatenate(([1.0], left[::-1], [middle], right, [1.0]))

    def _random_bases(
        self, d: int, K: int, p: int, n_basis: int, frequency: Union[float, Sequence[float]], coefficient_scale: float
    ) -> dict[str, np.ndarray]:
        if not isinstance(K, Integral) or K < 1 or not isinstance(p, Integral) or p < 2 or not isinstance(n_basis, Integral) or n_basis < 1:
            raise ValueError("K, p, and n_basis must be positive integers (with p >= 2)")
        values = np.asarray(frequency, dtype=float)
        if values.ndim == 0:
            low = high = float(values)
        elif values.shape == (2,):
            low, high = values
        else:
            raise ValueError("frequency must be a positive number or a (min, max) pair")
        if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or high < low:
            raise ValueError("frequency values must satisfy 0 < min <= max")
        if not np.isfinite(coefficient_scale) or coefficient_scale < 0.0:
            raise ValueError("coefficient_scale must be finite and non-negative")
        shape = (int(K), d, int(n_basis))
        basis_types = np.zeros(shape, dtype=np.int8)
        degrees = np.full(shape, 2, dtype=np.int16)
        frequencies = np.zeros(shape)
        directions = np.zeros(shape + (d,))
        interactions = np.zeros_like(directions)
        coefficients = np.zeros(shape)
        for layer in range(K):
            for coordinate in range(1, d):
                directions[layer, coordinate, :, :coordinate] = self._unit_vectors(n_basis, coordinate)
                interactions[layer, coordinate, :, :coordinate] = self._unit_vectors(n_basis, coordinate)
                basis_types[layer, coordinate] = self.rng.integers(ARAK._POLYNOMIAL, ARAK._TANH + 1, size=n_basis)
                degrees[layer, coordinate] = self.rng.integers(2, p + 1, size=n_basis)
                frequencies[layer, coordinate] = self.rng.uniform(low, high, size=n_basis)
                coefficients[layer, coordinate] = self.rng.normal(0.0, coefficient_scale, size=n_basis)
        return dict(basis_types=basis_types, degrees=degrees, frequencies=frequencies,
                    directions=directions, interaction_directions=interactions, coefficients=coefficients)

    def arak(
        self,
        dimension: int,
        scale: Optional[ArrayLike] = None,
        *,
        alignment: float = 0.0,
        K: int = 1,
        epsilon: float = 0.1,
        p: int = 3,
        n_basis: int = 4,
        n_knots: Optional[int] = None,
        knot_jitter: float = 0.2,
        frequency: Union[float, Sequence[float]] = (0.5, 2.0),
        coefficient_scale: float = 0.1,
        normalize_support: bool = True,
        support_mass: float = 0.999,
        calibration_samples: int = 1_000,
        support_width: float = 1.0,
        run_checks: bool = False,
        check_samples: int = 1_001,
        **parameters: object,
    ) -> ARAK:
        """Build one random, origin-centred ARAK.

        Parameters
        ----------
        dimension:
            Number of coordinates.
        scale:
            Fixed base affine matrix. If absent, the factory draws one from
            ``alignment``. Do not set both a scale and nonzero alignment.
        alignment:
            Tilt strength from 0 (axes aligned) to 1 (stronger random tilt).
            It matters only when ``scale`` is absent and dimension exceeds 1.
        K:
            Number of residual flow layers in dimensions two and above.
        epsilon:
            Size of every residual update. Higher values make the flow less
            like a Gaussian, for fixed basis parameters.
        p:
            Largest polynomial degree available to a random polynomial basis.
        n_basis:
            Number of random bases per coordinate and layer. In 1D it also
            sets the default number of spline knots.
        n_knots:
            Odd number of knots in a random 1D spline. It is ignored above 1D.
        knot_jitter:
            Random displacement of interior 1D knots. Higher values make
            their spacing less regular.
        frequency:
            One frequency or a low/high interval for random sine and cosine
            bases in dimensions two and above.
        coefficient_scale:
            Standard deviation of random residual coefficients, or log-height
            variation of the random 1D spline.
        normalize_support:
            Rescale the realised kernel to a central box after drawing it.
        support_mass:
            Marginal mass used to estimate that box. The default is 0.999.
        calibration_samples:
            Number of pilot samples used for this estimate.
        support_width:
            Width of the target central box. One gives roughly [-0.5, 0.5].
        run_checks, check_samples:
            Request deterministic construction checks and set their size.
        **parameters:
            Exact ARAK basis or spline arrays. They replace the corresponding
            random draw, but the optional support calibration still applies.
        """
        if not isinstance(dimension, Integral) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        d = int(dimension)
        if scale is None:
            scale = self._aligned_scale(d, alignment)
        elif alignment != 0.0:
            raise ValueError("alignment may not be combined with an explicit scale")
        base_scale = np.asarray(scale, dtype=float)
        if base_scale.shape != (d, d):
            raise ValueError(f"scale must have shape ({d}, {d})")

        exact_keys = {"basis_types", "degrees", "frequencies", "directions", "interaction_directions", "coefficients", "spline_knots", "spline_heights"}
        kernel_parameters = {key: parameters.pop(key) for key in list(parameters) if key in exact_keys}
        if parameters:
            raise TypeError(f"unexpected exact parameter(s): {', '.join(parameters)}")
        if d == 1:
            if not kernel_parameters:
                knots = max(5, 2 * n_basis + 1) if n_knots is None else n_knots
                kernel_parameters["spline_knots"], kernel_parameters["spline_heights"] = self._random_spline(knots, knot_jitter, coefficient_scale)
        elif not kernel_parameters:
            kernel_parameters = self._random_bases(d, K, p, n_basis, frequency, coefficient_scale)

        common = dict(dimension=d, scale=base_scale, epsilon=epsilon, **kernel_parameters)
        if not normalize_support:
            return ARAK(**common, run_checks=run_checks, check_samples=check_samples)
        if not 0.0 < support_mass < 1.0 or not isinstance(calibration_samples, Integral) or calibration_samples < 1:
            raise ValueError("support_mass must lie in (0, 1) and calibration_samples must be positive")
        if not np.isfinite(support_width) or support_width <= 0.0:
            raise ValueError("support_width must be positive")
        raw = ARAK(**common)
        radius = np.quantile(np.abs(raw.rvs(int(calibration_samples), random_state=self.rng)), support_mass, axis=0)
        if not np.all(np.isfinite(radius)) or np.any(radius <= 0.0):
            raise RuntimeError("support calibration produced invalid radii")
        support_scale = 0.5 * support_width / radius
        common["scale"] = support_scale[:, None] * base_scale
        return ARAK(**common, base_scale=base_scale, support_scale=support_scale, support_radius=radius,
                    run_checks=run_checks, check_samples=check_samples)

    def _relative_values(self, n_components: int, decay: float) -> np.ndarray:
        if not isinstance(n_components, Integral) or n_components < 1 or not np.isfinite(decay) or decay < 0.0:
            raise ValueError("n_components must be positive and decay must be finite and non-negative")
        values = np.ones(int(n_components))
        if n_components > 1 and decay:
            values[1:] = np.sort(self.rng.uniform(size=n_components - 1) ** decay)[::-1]
        return values

    def _uniform_or_constant(
        self, value: Union[float, ArrayLike], shape: tuple[int, ...], name: str,
        uniform_ranges: Optional[Mapping[str, ArrayLike]],
    ) -> np.ndarray:
        """Return a fixed broadcast value or independent draws from a range."""
        if uniform_ranges is not None and name in uniform_ranges:
            limits = np.asarray(uniform_ranges[name], dtype=float)
            if limits.shape != shape + (2,) or not np.all(np.isfinite(limits)) or np.any(limits[..., 0] > limits[..., 1]):
                raise ValueError(f"uniform_ranges[{name!r}] must have shape {shape + (2,)} with low <= high")
            return self.rng.uniform(limits[..., 0], limits[..., 1])
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(shape, float(array))
        else:
            array = np.broadcast_to(array, shape).copy()
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        return array

    def mixture(
        self,
        n_components: Optional[int] = None,
        bounds: Optional[ArrayLike] = None,
        *,
        dimension: Optional[int] = None,
        reference_center: Optional[ArrayLike] = None,
        components: Optional[Sequence[ARAK]] = None,
        weights: Optional[ArrayLike] = None,
        component_centers: Optional[ArrayLike] = None,
        component_scales: Optional[ArrayLike] = None,
        weight_decay: float = 1.0,
        scale_decay: float = 1.0,
        mode_scale: Union[float, ArrayLike] = 1.0,
        center_scale: Optional[Union[float, ArrayLike]] = None,
        component_scale: Union[float, ArrayLike] = 1.0,
        uniform_ranges: Optional[Mapping[str, ArrayLike]] = None,
        boundary_treatment: str = "compactify",
        boundary_hardness: Optional[Union[float, ArrayLike]] = None,
        compactification: str = "tanh",
        truncation_pilot_samples: int = 1_000,
        truncation_normalization_samples: int = 100_000,
        **arak_settings: object,
    ) -> ARAKMixture:
        """Build one random mixture, or assemble an exact one.

        Parameters
        ----------
        n_components:
            Number of modes to draw. Mode zero is the reference mode.
        bounds:
            Final absolute lower and upper limits for every coordinate.
        dimension:
            Dimension of each random ARAK mode.
        reference_center:
            Raw-space centre of mode zero. It is the starting point for all
            later centre draws. It is not the centre of ``bounds`` unless you
            choose it to be.
        components, weights, component_centers, component_scales:
            Exact mixture inputs. Supplying them bypasses random construction.
        weight_decay:
            Power in ``U**weight_decay`` for later relative weights. Higher
            values usually make later modes carry less probability.
        scale_decay:
            Same power-law draw for later relative mode sizes.
        mode_scale:
            First-mode outer size in each coordinate. Default one keeps the
            unit ARAK size.
        center_scale:
            Standard deviation of later raw-space centre offsets. It controls
            separation between modes and defaults to one.
        component_scale:
            Extra scale multiplier for each mode and coordinate. It defaults
            to one and multiplies ``mode_scale`` and ``scale_decay``.
        uniform_ranges:
            Optional low/high arrays from which to draw selected scales,
            alignments, or local bounds. See the class documentation for
            their shapes.
        boundary_treatment:
            ``'compactify'`` uses the analytical global map. ``'truncate'``
            keeps raw in-bounds samples and estimates its normalization.
        boundary_hardness, compactification:
            Final analytical compactification settings. They cannot be used
            with truncation.
        truncation_pilot_samples:
            Per-mode pilot draw count for truncation. A mode is retained only
            if at least one of these raw samples lies inside ``bounds``.
        truncation_normalization_samples:
            Number of raw draws from the retained, reweighted mixture used to
            estimate the in-bounds mass and its binomial uncertainty.
        **arak_settings:
            Settings passed to every random ARAK mode. ``alignment`` can be a
            scalar or one value per mode. Individual relative bounds are not
            supported; all boundary treatment is global at mixture level.
        """
        if components is not None:
            if weights is None or component_centers is None or component_scales is None or bounds is None:
                raise ValueError("explicit components require weights, centers, scales, and bounds")
            return ARAKMixture(components=components, weights=weights, component_centers=component_centers,
                               component_scales=component_scales, bounds=bounds,
                               boundary_treatment=boundary_treatment, boundary_hardness=boundary_hardness,
                               compactification=compactification)
        if n_components is None or bounds is None or dimension is None:
            raise ValueError("random mixture requires n_components, bounds, and dimension")
        if not isinstance(dimension, Integral) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        d = int(dimension)
        reference_center = np.zeros(d) if reference_center is None else np.asarray(reference_center, dtype=float)
        if reference_center.shape != (d,) or not np.all(np.isfinite(reference_center)):
            raise ValueError("reference_center must be finite with shape (dimension,)")
        mode = self._uniform_or_constant(mode_scale, (d,), "mode_scale", uniform_ranges)
        separation = self._uniform_or_constant(1.0 if center_scale is None else center_scale, (d,), "center_scale", uniform_ranges)
        component_factor = self._uniform_or_constant(component_scale, (n_components, d), "component_scale", uniform_ranges)
        if np.any(mode <= 0.0) or np.any(separation <= 0.0) or np.any(component_factor <= 0.0):
            raise ValueError("mode_scale, center_scale, and component_scale must be positive with their documented shapes")
        relative_weights = self._relative_values(n_components, weight_decay)
        relative_scales = self._relative_values(n_components, scale_decay)
        centres = np.empty((n_components, d))
        centres[0] = reference_center
        if n_components > 1:
            centres[1:] = reference_center + self.rng.normal(size=(n_components - 1, d)) * separation
        alignment = arak_settings.pop("alignment", 0.0)
        alignments = self._uniform_or_constant(alignment, (n_components,), "alignment", uniform_ranges)
        if np.any((alignments < 0.0) | (alignments > 1.0)):
            raise ValueError("alignment must lie in [0, 1]")
        parts = [self.arak(dimension=d, alignment=alignments[index], **arak_settings) for index in range(n_components)]
        scales = relative_scales[:, None] * mode * component_factor
        base_weights = relative_weights / relative_weights.sum()
        if boundary_treatment == "compactify":
            mixture = ARAKMixture(components=parts, weights=base_weights, component_centers=centres,
                                  component_scales=scales, bounds=bounds, boundary_treatment="compactify",
                                  boundary_hardness=boundary_hardness, compactification=compactification)
            mixture.relative_weights = relative_weights
            mixture.relative_scales = relative_scales
            return mixture
        if boundary_treatment != "truncate":
            raise ValueError("boundary_treatment must be 'compactify' or 'truncate'")
        if boundary_hardness is not None:
            raise ValueError("boundary_hardness cannot be used with truncation")
        if not isinstance(truncation_pilot_samples, Integral) or truncation_pilot_samples < 1:
            raise ValueError("truncation_pilot_samples must be a positive integer")
        if not isinstance(truncation_normalization_samples, Integral) or truncation_normalization_samples < 1:
            raise ValueError("truncation_normalization_samples must be a positive integer")
        limits = np.asarray(bounds, dtype=float)
        keep = np.zeros(n_components, dtype=bool)
        for index, part in enumerate(parts):
            pilot = centres[index] + scales[index] * part.rvs(int(truncation_pilot_samples), random_state=self.rng)
            keep[index] = np.any(np.all((pilot >= limits[:, 0]) & (pilot <= limits[:, 1]), axis=1))
        if not np.any(keep):
            raise RuntimeError("no mode had an in-bounds truncation pilot sample")
        retained_parts = [part for part, selected in zip(parts, keep) if selected]
        retained_centres = centres[keep]
        retained_scales = scales[keep]
        retained_weights = base_weights[keep]
        retained_weights /= retained_weights.sum()
        provisional = ARAKMixture(components=retained_parts, weights=retained_weights,
                                  component_centers=retained_centres, component_scales=retained_scales,
                                  bounds=bounds, boundary_treatment="compactify")
        raw = provisional._rvs_raw(int(truncation_normalization_samples), self.rng)
        inside_count = int(np.count_nonzero(np.all((raw >= limits[:, 0]) & (raw <= limits[:, 1]), axis=1)))
        mass = inside_count / int(truncation_normalization_samples)
        if mass == 0.0:
            raise RuntimeError("truncation normalization found no in-bounds samples")
        mass_error = np.sqrt(mass * (1.0 - mass) / int(truncation_normalization_samples))
        mixture = ARAKMixture(components=retained_parts, weights=retained_weights,
                              component_centers=retained_centres, component_scales=retained_scales,
                              bounds=bounds, boundary_treatment="truncate", truncation_mass=mass,
                              truncation_mass_error=mass_error, truncation_samples=int(truncation_normalization_samples))
        mixture.relative_weights = relative_weights[keep]
        mixture.relative_scales = relative_scales[keep]
        mixture.truncation_pilot_samples = int(truncation_pilot_samples)
        mixture.truncation_pilot_kept = keep
        mixture.truncation_inside_count = inside_count
        return mixture

    def product(
        self,
        mixture_settings: Optional[Sequence[Mapping[str, object]]] = None,
        *,
        mixtures: Optional[Sequence[ARAKMixture]] = None,
    ) -> ARAKMixtureProduct:
        """Realize an independent product of mixtures.

        Parameters
        ----------
        mixture_settings:
            Sequence of keyword dictionaries accepted by :meth:`mixture`.
            Each dictionary realizes one coordinate block, in the same order
            in which its coordinates appear in the returned product.
        mixtures:
            Exact realized mixtures.  Supplying this bypasses random
            construction and returns their deterministic Cartesian product.
            Exactly one of ``mixture_settings`` and ``mixtures`` is required.
        """
        if mixtures is not None:
            return ARAKMixtureProduct(mixtures)
        if mixture_settings is None:
            raise ValueError("provide mixture_settings or mixtures")
        return ARAKMixtureProduct([self.mixture(**dict(settings)) for settings in mixture_settings])

    @staticmethod
    def _centred_block_settings(
        mixture_settings: Sequence[Mapping[str, object]], center: ArrayLike,
    ) -> list[dict[str, object]]:
        """Copy block settings after placing each block's reference at ``center``."""
        location = np.asarray(center, dtype=float)
        if location.ndim != 1 or not np.all(np.isfinite(location)):
            raise ValueError("center must be a finite one-dimensional coordinate")
        copied: list[dict[str, object]] = []
        offset = 0
        for block in mixture_settings:
            if not isinstance(block, Mapping):
                raise TypeError("mixture_settings must contain keyword mappings")
            settings = dict(block)
            if "dimension" in settings:
                dimension = settings["dimension"]
                if not isinstance(dimension, Integral) or dimension < 1:
                    raise ValueError("each mixture setting dimension must be a positive integer")
                d = int(dimension)
            elif "component_centers" in settings:
                centres = np.asarray(settings["component_centers"], dtype=float)
                if centres.ndim != 2 or centres.shape[0] < 1:
                    raise ValueError("component_centers must have shape (n_components, dimension)")
                d = centres.shape[1]
            else:
                raise ValueError("each CAMP mixture setting needs dimension or component_centers")
            target = location[offset:offset + d]
            if target.shape != (d,):
                raise ValueError("center length must equal the total product dimension")
            if "component_centers" in settings:
                centres = np.asarray(settings["component_centers"], dtype=float)
                settings["component_centers"] = centres + target - centres[0]
            else:
                settings["reference_center"] = target
            copied.append(settings)
            offset += d
        if offset != len(location):
            raise ValueError("center length must equal the total product dimension")
        return copied

    def camp(
        self,
        n_distributions: int,
        mixture_settings: Sequence[Mapping[str, object]],
        *,
        center: Optional[ArrayLike] = None,
    ) -> ConnectedARAKMixtureProduct:
        """Build connected, independently realised ARAK mixture distributions.

        CAMP first builds one temporary distribution from ``mixture_settings``
        at ``center`` and draws ``n_distributions`` anchor points from it. It
        then builds one new distribution around each anchor, using the same settings but
        fresh random ARAKs, mode weights, scales, and offsets. An anchor is a
        point with probability under the temporary distribution, so the final
        distributions are placed in a common populated region rather than at
        unrelated random locations.

        Parameters
        ----------
        n_distributions:
            Number of final distributions. It must be a positive integer.
        mixture_settings:
            One keyword dictionary for each distribution coordinate block, using
            the arguments of :meth:`mixture`. Random blocks need ``dimension``.
            Exact blocks may instead provide ``component_centers``. Settings
            are copied and never modified. The same block settings are used
            for the temporary distribution and every final distribution.
        center:
            Full coordinate of the temporary distribution's first mode in every
            block. Its length is the sum of block dimensions. If omitted, a
            random block uses its usual zero ``reference_center``; an exact
            block keeps its supplied first component centre. Each final
            distribution receives one sampled anchor as its corresponding centre.

        Returns
        -------
        ConnectedARAKMixtureProduct
            A container with ``reference_distribution``, sampled ``anchors``, and
            the requested final ``distributions``. Indexing or iterating it yields
            normal deterministic :class:`ARAKMixtureProduct` objects.
        """
        if not isinstance(n_distributions, Integral) or n_distributions < 1:
            raise ValueError("n_distributions must be a positive integer")
        if not isinstance(mixture_settings, Sequence) or isinstance(mixture_settings, (str, bytes)) or not mixture_settings:
            raise TypeError("mixture_settings must be a non-empty sequence of keyword mappings")

        seed_settings = [dict(settings) for settings in mixture_settings]
        if center is not None:
            seed_settings = self._centred_block_settings(seed_settings, center)
        reference_distribution = self.product(seed_settings)
        anchors = reference_distribution.rvs(int(n_distributions), random_state=self.rng)
        distributions = [
            self.product(self._centred_block_settings(mixture_settings, anchor))
            for anchor in anchors
        ]
        return ConnectedARAKMixtureProduct(reference_distribution, anchors, distributions)


__all__ = [
    "ARAK",
    "ARAKMixture",
    "ARAKMixtureProduct",
    "ConnectedARAKMixtureProduct",
    "RandomDistribution",
]
