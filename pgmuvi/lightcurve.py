import contextlib
from pathlib import Path
from typing import ClassVar
import numpy as np
import torch
import gpytorch
from .gps import (
    SpectralMixtureLinearMeanGPModel,
    SpectralMixtureLinearMeanKISSGPModel,
    TwoDSpectralMixtureLinearMeanGPModel,
    TwoDSpectralMixtureLinearMeanKISSGPModel,
    SpectralMixtureGPModel,
    SpectralMixtureKISSGPModel,
    TwoDSpectralMixtureGPModel,
    TwoDSpectralMixtureKISSGPModel,
    QuasiPeriodicGPModel,
    MaternGPModel,
    PeriodicPlusStochasticGPModel,
    SeparableGPModel,
    AchromaticGPModel,
    WavelengthDependentGPModel,
    LinearMeanQuasiPeriodicGPModel,
    TwoDSpectralMixturePowerLawMeanGPModel,
    TwoDSpectralMixturePowerLawMeanKISSGPModel,
    TwoDSpectralMixtureDustMeanGPModel,
    TwoDSpectralMixtureDustMeanKISSGPModel,
    DustMeanGPModel,
    PowerLawMeanGPModel,
)
import matplotlib.pyplot as plt
from .trainers import train
from gpytorch.constraints import Interval, GreaterThan, LessThan, Positive  # noqa: F401
from .constraints import get_constraint_set
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior  # noqa: F401
from .priors import (
    LogNormalFrequencyPrior,
    LogNormalPeriodPrior,
    NormalFrequencyPrior,
    NormalPeriodPrior,
    get_prior_set,
)
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from inspect import isclass
import xarray as xr
import arviz as az
import warnings
import dataclasses
import json
import math


def _reraise_with_note(e, note):
    """Reraise an exception with a note added to the message

    This function is to provide a way to add a note to an exception, without
    losing the traceback, and without requiring python 3.11, which has
    added notes. It is based on this answer on stackoverflow:
    https://stackoverflow.com/a/75549200/16164384

    Parameters
    ----------
    e : Exception
        The exception to reraise
    note : str
        The note to add to the exception message
    """
    try:
        e.add_note(note)
    except AttributeError:
        args = e.args
        arg0 = f"{args[0]}\n{note}" if args else note
        e.args = (arg0, *args[1:])
    raise e


# Function to walk through nested dict and yield all values
# Taken from https://stackoverflow.com/a/12507546/16164384
def dict_walk_generator(indict, pre=None):
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                yield from dict_walk_generator(value, [*pre, key])
            elif isinstance(value, list | tuple):
                for v in value:
                    yield from dict_walk_generator(v, [*pre, key])
            else:
                yield [*pre, key, value]
    else:
        yield [*pre, indict]


def _convert_time_to_days(xdata, time_units):
    """Convert the time axis of xdata to days.

    Parameters
    ----------
    xdata : torch.Tensor, numpy.ndarray, or array-like
        The independent variable data.  For 1-D light curves this is a
        1-D (or single-column) array of time values.  For 2-D (multi-band)
        light curves this is a 2-D array of shape ``(N, 2)`` where column 0
        is time and column 1 is wavelength/band; only the time column is
        converted.  Non-tensor inputs are coerced to a ``torch.float32``
        tensor automatically.
    time_units : str, astropy.units.UnitBase, or None
        Units of the time values.  Any string accepted by
        ``astropy.units.Unit`` (e.g. ``'s'``, ``'hr'``, ``'yr'``,
        ``'days'``) and any ``astropy.units`` unit object are supported.
        If *None* the data are assumed to already be in days and are
        returned unchanged.

    Returns
    -------
    torch.Tensor
        ``xdata`` with the time axis expressed in days.

    Raises
    ------
    ValueError
        If *time_units* cannot be converted to days (e.g. it is a unit of
        length rather than time).
    """
    if time_units is None:
        return xdata

    import astropy.units as u

    if isinstance(time_units, str):
        unit = u.Unit(time_units)
    else:
        unit = time_units

    try:
        conversion_factor = float(unit.to(u.day))
    except u.UnitConversionError as e:
        raise ValueError(
            f"Cannot convert time_units '{time_units}' to days: {e}"
        ) from e

    # Coerce to tensor so .dim() / .shape are always available, regardless of
    # whether the caller passed a list, NumPy array, or torch.Tensor.
    if not isinstance(xdata, torch.Tensor):
        xdata = torch.as_tensor(xdata, dtype=torch.float32)

    if xdata.dim() <= 1 or xdata.shape[1] == 1:
        # 1-D light curve: all values are time
        return xdata * conversion_factor
    else:
        # 2-D light curve: column 0 is time, column 1 is wavelength
        xdata = xdata.clone()
        xdata[:, 0] = xdata[:, 0] * conversion_factor
        return xdata


class Transformer(torch.nn.Module):
    def __init__(self):
        """Baseclass for data transformers

        This is a baseclass for data transformers, which are used to transform
        data before it is passed to the GP.

        Parameters
        ----------

        Examples
        --------

        Notes
        -----
        The baseclass has no implementation, and should not be used directly.
        `__init__` is only implemented to allow the class to be subclassed and
        ensure that `nn.Module` stuff is setup correctly. Subclasses should
        implement the `transform` and `inverse` methods."""
        super().__init__()

    def transform(self, data, **kwargs):
        """Transform some data and return it, storing the parameters required
        to repeat or reverse the transform

        This is a baseclass with no implementations, your subclass should
        implement the transform itself
        """
        raise NotImplementedError

    def inverse(self, data, shift=True, **kwargs):
        """Invert a transform based on saved parameters

        This is a baseclass with no implementation, your subclass should
        implement the inverse transform itself
        """
        raise NotImplementedError


class MinMax(Transformer):
    def transform(self, data, dim=0, apply_to=None, recalc=False, shift=True, **kwargs):
        """Perform a MinMax transformation

        Transform the data such that each dimension is rescaled to the [0,1]
        interval. It stores the min and range of the data for the inverse
        transformation.

        Parameters
        ----------
        data : Tensor of floats
            The data to be transformed
        apply_to : tensor of ints or slice objects, optional
            Which dimensions to apply the transform to. If None, apply to all
        recalc : bool, default False
            Should the min and range of the transform be recalculated, or
            reused from previously?
        shift : bool, default True
            Should the data be shifted such that the minimum value is 0?
            This is mainly included so that data or parameters can be
            transformed when they apply to a single period - in this case,
            only the range needs to be applied.
        """
        if recalc or not hasattr(self, "min"):
            self.register_buffer("min", torch.min(data, dim=dim, keepdim=True)[0])
            self.register_buffer(
                "range", torch.max(data, dim=dim, keepdim=True)[0] - self.min
            )
            shift = True  # if we're recalculating, we need to shift
        if apply_to is not None:
            return (data - (shift * self.min[apply_to])) / self.range[apply_to]
        return (data - (shift * self.min)) / self.range

    def inverse(self, data, shift=True, **kwargs):
        """Invert a MinMax transformation based on saved state

        Invert the transformation of the data from  the [0,1] interval.
        It used the stored min and range of the data for the inverse
        transformation.

        Parameters
        ----------
        data : Tensor of floats
            The data to be reverse-transformed
        """
        return (data * self.range) + (shift * self.min)


class ZScore(Transformer):
    def transform(self, data, dim=0, apply_to=None, recalc=False, shift=True, **kwargs):
        """Perform a z-score transformation

        Transform the data such that each dimension is rescaled such that
        its mean is 0 and its standard deviation is 1.

        Parameters
        ----------
        data : Tensor of floats
            The data to be transformed
        apply_to : int or tensor of ints, optional
            Which dimensions to apply the transform to. If None, apply to all
        recalc : bool, default False
            Should the parameters of the transform be recalculated, or reused
            from previously?
        shift : bool, default True
            Should the data be shifted such that the mean value is 0?
            This is mainly included so that data or parameters can be
            transformed when they apply to a single period - in this case,
            only the standard deviation needs to be applied.
        """
        if recalc or not hasattr(self, "mean"):
            mean = torch.mean(data, dim=dim, keepdim=True)[0]
            self.register_buffer("mean", mean)
            sd = torch.std(data, dim=dim, keepdim=True)[0]
            self.register_buffer("sd", sd)
            shift = True  # if we're recalculating, we need to shift
        if apply_to is not None:
            return (data - (shift * self.mean[apply_to])) / self.sd[apply_to]
        return (data - shift * self.mean) / self.sd

    def inverse(self, data, shift=True, **kwargs):
        """Invert a z-score transformation based on saved state

        Invert the z-scoring of the data based on the saved mean and standard
        deviation

        Parameters
        ----------
        data : Tensor of floats
            The data to be reverse-transformed
        """
        return (data * self.sd) + (self.mean * shift)


class RobustZScore(Transformer):
    def transform(self, data, dim=0, apply_to=None, recalc=False, shift=True, **kwargs):
        """Perform a robust z-score transformation

        Transform the data such that each dimension is rescaled such that
        its median is 0 and its median absolute deviation is 1.

        Parameters
        ----------
        data : Tensor of floats
            The data to be transformed
        apply_to : int or tensor of ints, optional
            Which dimensions to apply the transform to. If None, apply to all
        recalc : bool, default False
            Should the parameters of the transform be recalculated, or reused
            from previously?
        shift : bool, default True
            Should the data be shifted such that the median value is 0?
            This is mainly included so that data or parameters can be
            transformed when they apply to a single period - in this case,
            only the median absolute deviation needs to be applied.
        """
        if recalc or not hasattr(self, "mad"):
            median = torch.median(data, dim=dim, keepdim=True)[0]
            self.register_buffer("median", median)
            mad = torch.median(torch.abs(data - median), dim=dim, keepdim=True)[0]
            self.register_buffer("mad", mad)
            shift = True  # if we're recalculating, we need to shift
        if apply_to is not None:
            return (data - shift * self.median[apply_to]) / self.mad[apply_to]
        return (data - shift * self.median) / self.mad

    def inverse(self, data, shift=True, **kwargs):
        """Invert a robust z-score transformation based on saved state

        Invert the robust z-scoring of the data based on the saved median and
        median absolute deviation.

        Parameters
        ----------
        data : Tensor of floats
            The data to be reverse-transformed
        """
        return (data * self.mad) + (self.median * shift)


def minmax(data, dim=0):
    m = torch.min(data, dim=dim, keepdim=True)
    r = torch.max(data, dim=dim, keepdim=True) - m
    return (data - m) / r, m, r


class InputHelpers:
    """Mixin class providing helper methods for reading data from various input formats.

    This class provides classmethods for instantiating a :class:`Lightcurve`
    from different input formats, with flexible column name detection.
    :class:`Lightcurve` inherits from this class so all methods are available
    directly on :class:`Lightcurve`.

    Attributes
    ----------
    _X_COLUMN_NAMES : list of str
        Candidate column names used for auto-detecting the time (independent
        variable) column, checked case-insensitively in order.
    _Y_COLUMN_NAMES : list of str
        Candidate column names used for auto-detecting the dependent variable
        (y) column, checked case-insensitively in order.
    _YERR_COLUMN_NAMES : list of str
        Candidate column names used for auto-detecting the uncertainty column,
        checked case-insensitively in order.
    _WAVELENGTH_COLUMN_NAMES : list of str
        Candidate column names used for auto-detecting a *numeric* wavelength
        column, checked case-insensitively in order.  When such a column is
        found and contains more than one unique value, the data are loaded as
        a 2-D lightcurve whose ``xdata`` has shape ``(N, 2)`` with the time
        values in column 0 and the wavelength values in column 1.
    _WAVELENGTH_ID_COLUMN_NAMES : list of str
        Candidate column names used for auto-detecting a *string* band
        identifier column (e.g. ``"V"``, ``"R"``, ``"W1"``), checked
        case-insensitively in order.  When such a column is found it is
        ingested as the ``band`` attribute of the resulting
        :class:`Lightcurve`.
    """

    _X_COLUMN_NAMES: ClassVar[list[str]] = [
        "x", "time", "t", "jd", "mjd", "date", "hjd", "bjd", "epoch"
    ]
    _Y_COLUMN_NAMES: ClassVar[list[str]] = [
        "y", "magnitude", "mag", "flux", "value", "data"
    ]
    _YERR_COLUMN_NAMES: ClassVar[list[str]] = [
        "yerr",
        "uncertainty",
        "error",
        "err",
        "unc",
        "sigma",
        "e_magnitude",
        "e_mag",
        "e_flux",
        "flux_error",
        "mag_error",
        "magnitude_error",
        "value_error",
        "data_error",
        "y_error",
    ]
    _WAVELENGTH_COLUMN_NAMES: ClassVar[list[str]] = [
        "wavelength",
        "wave",
        "wl",
        "lambda",
        "freq",
        "frequency",
        "channel",
    ]
    _WAVELENGTH_ID_COLUMN_NAMES: ClassVar[list[str]] = [
        "band",
        "filter",
        "filtername",
        "filter_name",
    ]

    @classmethod
    def _find_column(
        cls, columns: list[str], candidates: list[str]
    ) -> str | None:
        """Find the first matching column name from a list of candidates.

        Matching is case-insensitive.

        Parameters
        ----------
        columns : list of str
            The available column names.
        candidates : list of str
            Candidate column names to search for, in priority order.

        Returns
        -------
        str or None
            The matched column name (preserving the original capitalisation
            from *columns*), or ``None`` if no candidate was found.
        """
        columns_lower = {c.lower(): c for c in columns}
        for candidate in candidates:
            if candidate.lower() in columns_lower:
                return columns_lower[candidate.lower()]
        return None

    @staticmethod
    def _drop_nonfinite_rows(x, y, yerr):
        """Drop rows containing non-finite (NaN or Inf) values from data arrays.

        Parameters
        ----------
        x : torch.Tensor
            Independent variable tensor of shape ``(N,)`` or ``(N, D)``.
        y : torch.Tensor
            Dependent variable tensor of shape ``(N,)``.
        yerr : torch.Tensor or None
            Uncertainty tensor of shape ``(N,)``, or ``None``.

        Returns
        -------
        x : torch.Tensor
            Filtered independent variable tensor.
        y : torch.Tensor
            Filtered dependent variable tensor.
        yerr : torch.Tensor or None
            Filtered uncertainty tensor, or ``None`` if it was ``None`` on
            input.

        Notes
        -----
        A ``UserWarning`` is emitted when one or more rows are dropped.
        A ``ValueError`` is raised when no valid rows remain after filtering.
        """
        valid_mask = torch.isfinite(y)
        if x.dim() > 1:
            valid_mask &= torch.isfinite(x).all(dim=1)
        else:
            valid_mask &= torch.isfinite(x)
        if yerr is not None:
            valid_mask &= torch.isfinite(yerr)
        n_dropped = int((~valid_mask).sum().item())
        if n_dropped > 0:
            warnings.warn(
                f"Dropped {n_dropped} row(s) containing non-finite "
                "(NaN or Inf) values.",
                stacklevel=3,
            )
            x = x[valid_mask]
            y = y[valid_mask]
            if yerr is not None:
                yerr = yerr[valid_mask]
        if y.numel() == 0:
            raise ValueError(
                "No valid data rows remain after dropping non-finite rows."
            )
        elif y.numel() < 10 and n_dropped > 0:
            warnings.warn(
                f"Fewer than 10 elements remain after dropping {n_dropped} rows,"
                " take care interpreting results!",
                stacklevel=3,
            )
        return x, y, yerr

    @staticmethod
    def _drop_nan_rows(x, y, yerr):
        """Drop rows that contain NaN in any of the data arrays.

        .. deprecated:: 0.3.0
            Use :meth:`_drop_nonfinite_rows` instead, which also handles
            infinite values.
        """
        return Lightcurve._drop_nonfinite_rows(x, y, yerr)

    @classmethod
    def from_csv(
        cls,
        filepath: str | Path,
        xcol: str | list[str] | None = None,
        ycol: str | None = None,
        yerrcol: str | None = None,
        wavelcol: str | None = None,
        **kwargs,
    ) -> "Lightcurve":
        """Instantiate a Lightcurve from a CSV file.

        The file must have a header line whose entries are used to identify
        the relevant data columns.  Column names are matched
        case-insensitively.

        **1-D lightcurves** (single time series)
            When only a time column and a flux/magnitude column are present,
            or when all observations share the same wavelength/band, the
            resulting ``xdata`` is a 1-D tensor of shape ``(N,)``.

        **2-D (multiband) lightcurves**
            When the CSV contains a *numeric* wavelength column with more than
            one unique value, the resulting ``xdata`` has shape ``(N, 2)``
            where column 0 holds the time values and column 1 holds the
            numeric wavelength values.  The ``ydata`` (and optional ``yerr``)
            remain 1-D tensors of shape ``(N,)``.

            The numeric wavelength column is selected in one of three ways:

            1. *Explicit ``xcol`` list*: pass ``xcol`` as a list of two
               column names, e.g. ``xcol=["time", "wavelength"]``.  The first
               element is the time column and the second is the wavelength
               column.  All subsequent x-axis columns are stacked in the
               order given.
            2. *Explicit ``wavelcol``*: pass the column name as a separate
               ``wavelcol`` keyword argument.
            3. *Auto-detection*: if neither an iterable ``xcol`` nor a
               ``wavelcol`` is supplied, the method searches for a column
               whose name matches one of the entries in
               :attr:`_WAVELENGTH_COLUMN_NAMES` (e.g. ``"wavelength"``,
               ``"wl"``).  If such a column is found and contains more than
               one unique value, a 2-D lightcurve is returned automatically.

        **Band labels**
            String band-identifier columns (e.g. one named ``"band"`` or
            ``"filter"`` containing values like ``"V"``, ``"R"``) are
            resolved *independently* of the numeric wavelength column.  When
            the CSV contains a string-typed column whose name matches one of
            the entries in :attr:`_WAVELENGTH_ID_COLUMN_NAMES` **and** the
            resulting lightcurve is 2-D, the unique labels are stored
            automatically in :attr:`Lightcurve.band`.  For 1-D lightcurves,
            ``band`` is left as ``None`` unless supplied explicitly via
            ``**kwargs``.  Numeric wavelength columns are used directly and
            :attr:`Lightcurve.band` is left as ``None`` unless the caller
            provides ``band=`` explicitly in ``**kwargs``.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the CSV file.
        xcol : str or list of str or None, optional
            Name of the column containing the time (independent variable)
            data, or a list of column names to stack as the x-axis (first
            element is time, subsequent elements are additional dimensions
            such as wavelength).  If not provided, auto-detection is
            attempted using :attr:`_X_COLUMN_NAMES` for the time column.
        ycol : str or None, optional
            Name of the column containing the dependent variable (y) data.
            If not provided, auto-detection is attempted using
            :attr:`_Y_COLUMN_NAMES`.
        yerrcol : str or None, optional
            Name of the column containing the uncertainties on the dependent
            variable.  If not provided, auto-detection is attempted using
            :attr:`_YERR_COLUMN_NAMES`.  If no matching column is found,
            ``yerr`` is set to ``None``.
        wavelcol : str or None, optional
            Name of the column containing wavelength or band values.  When
            provided, the time and wavelength columns are stacked to form a
            2-D ``xdata``.  Ignored when ``xcol`` is a list.
        **kwargs
            Additional keyword arguments passed to the Lightcurve constructor.
            If ``band`` is not provided and the wavelength/band column is
            string-typed, it is populated automatically.

        Returns
        -------
        Lightcurve
            A 1-D lightcurve when a single time column is used (or when the
            wavelength/band column has only one unique value), or a 2-D
            lightcurve when multiple wavelengths/bands are present.

        Raises
        ------
        ValueError
            If a required column cannot be auto-detected and was not specified
            explicitly, or if an explicitly specified column name is not
            present in the file.
        """
        filepath = Path(filepath)
        # Use dtype=None so that NumPy auto-detects each column's type.
        # This allows string/bytes band columns (e.g. "V", "R") to be read
        # as-is rather than being silently coerced to NaN.
        data = np.genfromtxt(
            filepath, delimiter=",", names=True, dtype=None, encoding=None
        )
        columns = list(data.dtype.names)

        # ------------------------------------------------------------------
        # Helper: is a structured-array column string-typed?
        # ------------------------------------------------------------------
        def _is_str_col(col_name):
            dt = data.dtype[col_name]
            return np.issubdtype(dt, np.str_) or np.issubdtype(dt, np.bytes_)

        # ------------------------------------------------------------------
        # Resolve all column names (no tensor building yet)
        # ------------------------------------------------------------------

        # Resolve x / time column
        if not isinstance(xcol, list):
            if xcol is None:
                xcol = cls._find_column(columns, cls._X_COLUMN_NAMES)
                if xcol is None:
                    raise ValueError(
                        f"Could not auto-detect x column. "
                        f"Available columns: {columns}. "
                        "Please specify xcol explicitly."
                    )
            elif xcol not in columns:
                raise ValueError(
                    f"Column '{xcol}' not found in CSV. "
                    f"Available columns: {columns}"
                )

        # Resolve y column
        if ycol is None:
            ycol = cls._find_column(columns, cls._Y_COLUMN_NAMES)
            if ycol is None:
                raise ValueError(
                    f"Could not auto-detect y column. "
                    f"Available columns: {columns}. "
                    "Please specify ycol explicitly."
                )
        elif ycol not in columns:
            raise ValueError(
                f"Column '{ycol}' not found in CSV. Available columns: {columns}"
            )

        # Resolve yerr column
        if yerrcol is None:
            yerrcol = cls._find_column(columns, cls._YERR_COLUMN_NAMES)
        elif yerrcol not in columns:
            raise ValueError(
                f"Column '{yerrcol}' not found in CSV. Available columns: {columns}"
            )

        # ------------------------------------------------------------------
        # Resolve the x (time + optional numeric wavelength) columns.
        # The string band-ID column is resolved independently below.
        # ------------------------------------------------------------------
        band_id_col = None  # set in the else branch when applicable
        if isinstance(xcol, list):
            # Explicit multi-column x specification
            for col in xcol:
                if col not in columns:
                    raise ValueError(
                        f"Column '{col}' not found in CSV. "
                        f"Available columns: {columns}"
                    )
            # Build NaN mask across all columns before stacking
            xcol_names = xcol
        else:
            # Resolve numeric wavelength column for xdata[:, 1].
            # Only _WAVELENGTH_COLUMN_NAMES is consulted; string band-ID
            # columns are handled separately and independently.
            if wavelcol is not None:
                # Explicit: validate it exists before proceeding.
                if wavelcol not in columns:
                    raise ValueError(
                        f"Column '{wavelcol}' not found in CSV. "
                        f"Available columns: {columns}"
                    )
            else:
                wavelcol = cls._find_column(columns, cls._WAVELENGTH_COLUMN_NAMES)

            # Independently resolve string band-ID column for lc.band.
            # This is always attempted, regardless of whether a numeric
            # wavelength column was found.
            band_id_col = cls._find_column(columns, cls._WAVELENGTH_ID_COLUMN_NAMES)

            xcol_names = [xcol] + ([wavelcol] if wavelcol is not None else [])

        # ------------------------------------------------------------------
        # Build NaN / validity mask from ALL relevant columns.
        # With dtype=None, integer and string columns cannot be NaN, so only
        # check floating-point columns for non-finite values.
        # ------------------------------------------------------------------
        relevant_cols = xcol_names + [ycol] + ([yerrcol] if yerrcol else [])
        valid_mask = np.ones(len(data), dtype=bool)
        for col in relevant_cols:
            col_dtype = data.dtype[col]
            if np.issubdtype(col_dtype, np.floating):
                valid_mask &= ~np.isnan(data[col])
            elif _is_str_col(col):
                # Treat empty strings as missing; convert once outside the
                # per-element comparison.
                valid_mask &= np.asarray(data[col], dtype=np.str_) != ""
            # Integer columns cannot contain NaN; no filtering needed.

        n_dropped = int((~valid_mask).sum())
        if n_dropped > 0:
            warnings.warn(
                f"Dropped {n_dropped} row(s) containing NaN values.",
                stacklevel=2,
            )
        if valid_mask.sum() == 0:
            raise ValueError(
                "No valid data rows remain after dropping NaN-containing rows."
            )

        # Apply mask to get clean data
        clean = data[valid_mask]

        # ------------------------------------------------------------------
        # Helper: convert a structured-array column to a float32 tensor.
        # Boolean-indexing a structured array with dtype=None can produce
        # non-contiguous strides; np.array() (not np.asarray) forces a copy.
        # ------------------------------------------------------------------
        def _to_float_tensor(arr):
            return torch.as_tensor(
                np.array(arr, dtype=np.float64), dtype=torch.float32
            )

        # ------------------------------------------------------------------
        # Helper: map string band labels to float indices and record them.
        # Returns (wave_tensor, unique_labels_array).
        # ------------------------------------------------------------------
        def _str_col_to_wave(arr):
            str_vals = np.asarray(arr, dtype=np.str_)
            # Preserve first-appearance order via dict.fromkeys.
            unique_labels = list(dict.fromkeys(str_vals.tolist()))
            label_to_idx = {lbl: float(i) for i, lbl in enumerate(unique_labels)}
            indices = np.array([label_to_idx[v] for v in str_vals], dtype=np.float64)
            return (
                torch.as_tensor(indices, dtype=torch.float32),
                np.array(unique_labels, dtype=np.str_),
            )

        # ------------------------------------------------------------------
        # Build tensors from clean data
        # ------------------------------------------------------------------
        if isinstance(xcol, list):
            x_tensors = []
            for col in xcol:
                if _is_str_col(col):
                    wave_t, _unique = _str_col_to_wave(clean[col])
                    x_tensors.append(wave_t)
                    if "band" not in kwargs:
                        kwargs["band"] = np.asarray(clean[col], dtype=np.str_)
                else:
                    x_tensors.append(_to_float_tensor(clean[col]))
            x = torch.stack(x_tensors, dim=1) if len(x_tensors) > 1 else x_tensors[0]
        else:
            time_tensor = _to_float_tensor(clean[xcol])
            if wavelcol is not None:
                if _is_str_col(wavelcol):
                    # Explicitly-provided string wavelcol: map labels → indices.
                    wave_tensor, _unique = _str_col_to_wave(clean[wavelcol])
                    if "band" not in kwargs:
                        kwargs["band"] = np.asarray(clean[wavelcol], dtype=np.str_)
                else:
                    wave_tensor = _to_float_tensor(clean[wavelcol])
                if wave_tensor.unique().numel() > 1:
                    # Multiple wavelengths → 2-D lightcurve
                    x = torch.stack([time_tensor, wave_tensor], dim=1)
                else:
                    # Single wavelength → treat as 1-D
                    x = time_tensor
            else:
                x = time_tensor

            # Independently populate band from the string band-ID column.
            # Only auto-populate when xdata is 2-D, matching from_table
            # behaviour (band labels per-row are meaningless for 1-D).
            if x.dim() == 2 and "band" not in kwargs and band_id_col is not None:
                if _is_str_col(band_id_col):
                    kwargs["band"] = np.asarray(clean[band_id_col], dtype=np.str_)

        y = _to_float_tensor(clean[ycol])
        yerr = _to_float_tensor(clean[yerrcol]) if yerrcol else None

        return cls(xdata=x, ydata=y, yerr=yerr, **kwargs)


# Spectral-mixture model names that support MLS-based initialisation in fit().
_SM_MODELS: frozenset[str] = frozenset(
    {
        "2D",
        "1D",
        "1DLinear",
        "2DLinear",
        "2DPowerLaw",
        "2DDust",
        "1DSKI",
        "2DSKI",
        "1DLinearSKI",
        "2DLinearSKI",
        "2DPowerLawSKI",
        "2DDustSKI",
    }
)


@dataclasses.dataclass(frozen=True)
class PeriodPeakResult:
    """A single PSD peak from :meth:`Lightcurve.get_period_summary`."""

    rank: int = 1
    frequency: float = float("nan")
    period: float = float("nan")
    height: float = float("nan")
    prominence: float = float("nan")
    area_fraction: float = float("nan")
    interval_frequency: tuple = (float("nan"), float("nan"))
    interval_period: tuple = (float("nan"), float("nan"))
    period_ratio_to_primary: float = 1.0
    is_candidate_lsp: bool = False
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "frequency": self.frequency,
            "period": self.period,
            "height": self.height,
            "prominence": self.prominence,
            "area_fraction": self.area_fraction,
            "interval_frequency": list(self.interval_frequency),
            "interval_period": list(self.interval_period),
            "period_ratio_to_primary": self.period_ratio_to_primary,
            "is_candidate_lsp": self.is_candidate_lsp,
            "notes": self.notes,
        }


class ComponentDiagnosticsResult:
    """Kernel-component diagnostic information for a spectral-mixture GP.

    These values are extracted directly from GP hyperparameters and are
    provided for diagnostic purposes only.  They must **not** be interpreted
    as independent physical periods.  The literature-comparable period
    estimates are the summed-PSD peaks stored in
    :attr:`PeriodSummaryResult.peaks`.

    Attributes
    ----------
    component_periods : numpy.ndarray
        Centre period of each mixture component (1/frequency).
    component_frequencies : numpy.ndarray
        Centre frequency of each mixture component.
    component_weights : numpy.ndarray
        Relative amplitude weight of each mixture component.
    component_period_scales : numpy.ndarray
        Width (sigma) of each Gaussian component in period units.
    component_frequency_scales : numpy.ndarray
        Width (sigma) of each Gaussian component in frequency units.
    n_components : int
        Number of mixture components.
    kernel_family : str
        Name of the spectral-mixture kernel family.
    notes : str
        Diagnostic notes for this component set.
    component_labels : list of str
        Human-readable label for each component,
        e.g. ``["SM component 1", "SM component 2"]``.
    """

    def __init__(
        self,
        component_periods=None,
        component_frequencies=None,
        component_weights=None,
        component_period_scales=None,
        component_frequency_scales=None,
        n_components=0,
        kernel_family="",
        notes="",
        component_labels=None,
    ):
        self.component_periods = (
            component_periods
            if component_periods is not None
            else np.array([])
        )
        self.component_frequencies = (
            component_frequencies
            if component_frequencies is not None
            else np.array([])
        )
        self.component_weights = (
            component_weights
            if component_weights is not None
            else np.array([])
        )
        self.component_period_scales = (
            component_period_scales
            if component_period_scales is not None
            else np.array([])
        )
        self.component_frequency_scales = (
            component_frequency_scales
            if component_frequency_scales is not None
            else np.array([])
        )
        self.n_components = n_components
        self.kernel_family = kernel_family
        self.notes = notes
        self.component_labels = component_labels or [
            f"SM component {i + 1}" for i in range(n_components)
        ]

    def as_dict(self) -> dict:
        """Return a plain-dict representation of this diagnostics object."""
        return {
            "n_components": self.n_components,
            "kernel_family": self.kernel_family,
            "notes": self.notes,
            "component_labels": self.component_labels,
            "component_periods": self.component_periods,
            "component_frequencies": self.component_frequencies,
            "component_weights": self.component_weights,
            "component_period_scales": self.component_period_scales,
            "component_frequency_scales": self.component_frequency_scales,
        }


class PeriodSummaryResult:
    """Structured result from :meth:`Lightcurve.get_period_summary`."""

    def __init__(
        self,
        method="",
        model_name="",
        n_peaks_detected=0,
        n_peaks_analyzed=0,
        n_peaks_requested=None,
        dominant_period=None,
        dominant_frequency=None,
        peaks=None,
        freq_grid=None,
        psd=None,
        notes="",
        component_diagnostics=None,
        interval_definition="peak_centered_68pct_mass_interval",
        backend="",
        kernel_family="",
        time_kernel_family="",
        has_stochastic_background=False,
        q_factor=None,
    ):
        self.method = method
        self.model_name = model_name
        self.backend = backend
        self.kernel_family = kernel_family
        self.time_kernel_family = time_kernel_family
        self.has_stochastic_background = has_stochastic_background
        self.n_peaks_detected = n_peaks_detected
        self.n_peaks_analyzed = n_peaks_analyzed
        self.n_peaks_requested = n_peaks_requested
        self.dominant_period = dominant_period
        self.dominant_frequency = dominant_frequency
        self.q_factor = q_factor
        # Sort peaks once by descending significance so that peaks[0] is
        # always the highest-significance peak regardless of insertion order.
        # Sort key: (1) descending area_fraction (NaN sorts last),
        #           (2) descending height (NaN sorts last),
        #           (3) ascending original rank as tie-breaker.
        # After sorting, ranks are reassigned sequentially (1, 2, 3 …) so
        # that peak.rank reliably reflects position in the sorted list.
        _raw_peaks = peaks if peaks is not None else []

        def _significance_key(p):
            af = p.area_fraction if np.isfinite(p.area_fraction) else -np.inf
            h = p.height if np.isfinite(p.height) else -np.inf
            return (-af, -h, p.rank)

        _sorted = sorted(_raw_peaks, key=_significance_key)
        # Reassign ranks sequentially and update period_ratio_to_primary so
        # that the new rank-1 peak always has ratio=1.0 and the other peaks
        # are relative to it.
        _primary_period = _sorted[0].period if _sorted else 1.0
        self.peaks = [
            dataclasses.replace(
                p,
                rank=i + 1,
                period_ratio_to_primary=(
                    p.period / _primary_period
                    if _primary_period > 0 and np.isfinite(p.period)
                    else float("nan")
                ),
            )
            for i, p in enumerate(_sorted)
        ]
        self.freq_grid = freq_grid
        self.psd = psd
        self.notes = notes
        self.interval_definition = interval_definition
        self.component_diagnostics = component_diagnostics

    def as_dict(self) -> dict:
        # Derive primary-peak quantities once so they can be reused for
        # the backward-compatible alias keys without repeating the logic.
        primary = self.get_primary_peak()
        primary_interval = primary.interval_period if primary is not None else None
        primary_area = (
            primary.area_fraction if primary is not None else float("nan")
        )
        # Prefer primary-peak values for dominant_period/frequency so that
        # all reported quantities (period, interval, area) consistently
        # describe the same peak.  Fall back to the constructor-provided
        # values if no peaks are present (non-periodic / explicit-period
        # backends that set dominant_period directly).
        dominant_period = (
            primary.period if primary is not None else self.dominant_period
        )
        dominant_frequency = (
            primary.frequency if primary is not None else self.dominant_frequency
        )
        _sig_peaks = self.get_significant_peaks()
        significant_periods = np.array([p.period for p in _sig_peaks])

        return {
            "component_diagnostics": (
                self.component_diagnostics.as_dict()
                if self.component_diagnostics is not None
                else None
            ),
            "freq_grid": self.freq_grid,
            "psd": self.psd,
            "dominant_frequency": dominant_frequency,
            "dominant_period": dominant_period,
            # Backward-compatible interval keys (both alias the same value).
            "period_interval_fwhm_like": primary_interval,
            "period_interval": primary_interval,
            "interval_definition": self.interval_definition,
            "q_factor": self.q_factor,
            "peak_fraction": primary_area,
            "n_peaks": len(self.peaks),
            "n_peaks_detected": self.n_peaks_detected,
            "n_significant_peaks": len(_sig_peaks),
            "significant_periods": significant_periods,
            "peaks": [p.as_dict() for p in self.peaks],
            "method": self.method,
            "notes": self.notes,
            # Kernel-dispatch metadata
            "backend": self.backend,
            "kernel_family": self.kernel_family,
            "time_kernel_family": self.time_kernel_family,
            "has_stochastic_background": self.has_stochastic_background,
        }

    def __getitem__(self, key):
        return self.as_dict()[key]

    def __contains__(self, key):
        return key in self.as_dict()

    def get(self, key, default=None):
        return self.as_dict().get(key, default)

    def keys(self):
        return self.as_dict().keys()

    def items(self):
        return self.as_dict().items()

    def values(self):
        return self.as_dict().values()

    # ------------------------------------------------------------------
    # Multi-peak accessors
    # ------------------------------------------------------------------

    def get_primary_peak(self):
        """Return the primary (rank-1) peak, or ``None`` if none exist.

        Returns
        -------
        PeriodPeakResult or None
            The first entry in :attr:`peaks` (sorted by ascending rank,
            so rank 1 is always first), or ``None`` when :attr:`peaks`
            is empty.
        """
        return self.peaks[0] if self.peaks else None

    def get_top_n_peaks(self, n):
        """Return up to *n* peaks in ascending rank order.

        Parameters
        ----------
        n : int
            Maximum number of peaks to return.

        Returns
        -------
        list of PeriodPeakResult
            A slice of :attr:`peaks` of length ``min(n, len(peaks))``.
        """
        return self.peaks[:n]

    def get_significant_peaks(self, threshold=0.68):
        """Return peaks whose area fraction meets or exceeds *threshold*.

        Parameters
        ----------
        threshold : float, optional
            Minimum ``area_fraction`` to qualify as significant.
            Default is ``0.68`` (~1 sigma).

        Returns
        -------
        list of PeriodPeakResult
            Peaks from :attr:`peaks` (in rank order) for which
            ``peak.area_fraction >= threshold``.  Peaks with NaN area
            fraction are excluded.
        """
        return [
            p for p in self.peaks
            if np.isfinite(p.area_fraction) and p.area_fraction >= threshold
        ]

    def to_table(self) -> list:
        return [
            {
                "peak_rank": p.rank,
                "period": p.period,
                "frequency": p.frequency,
                "height": p.height,
                "prominence": p.prominence,
                "area_fraction": p.area_fraction,
                "period_interval_lo": p.interval_period[0],
                "period_interval_hi": p.interval_period[1],
                "period_ratio_to_primary": p.period_ratio_to_primary,
                "is_candidate_lsp": p.is_candidate_lsp,
                "notes": p.notes,
            }
            for p in self.peaks
        ]

    def to_text(
        self,
        include_components=True,
        include_peaks=True,
        include_psd_info=False,
        max_peaks_to_show=3,
    ) -> str:
        """Return a human-readable text summary of this period result.

        The text is plain UTF-8 text, suitable for writing to a ``.txt``
        file, reading in a terminal, or storing alongside analysis outputs.
        It clearly separates **analyzed peak results** (the
        literature-comparable outputs) from **kernel component diagnostics**
        (internal quantities derived directly from GP hyperparameters).

        Parameters
        ----------
        include_components : bool, optional
            If ``True`` (default), include a section listing the kernel
            component periods, frequencies, and weights.  These are
            **diagnostic quantities** and should not be cited as final
            period determinations.
        include_peaks : bool, optional
            If ``True`` (default), include one block per analyzed peak.
        include_psd_info : bool, optional
            If ``True``, include a short summary of the PSD grid
            (existence, length, frequency range, PSD range).  The full
            arrays are never dumped.  Default is ``False``.
        max_peaks_to_show : int, optional
            Maximum number of peaks to show in detail.  The primary peak
            is always shown first; up to ``max_peaks_to_show - 1``
            additional peaks follow.  If more peaks exist, a count line
            is appended.  Default is ``3``.

        Returns
        -------
        str
            Formatted text summary.
        """

        def _fmt(v, precision=6):
            """Format a scalar value for display."""
            if v is None:
                return "N/A"
            try:
                if np.isnan(v):
                    return "nan"
                if np.isinf(v):
                    return "inf" if v > 0 else "-inf"
            except (TypeError, ValueError):
                pass
            try:
                return f"{v:.{precision}g}"
            except (TypeError, ValueError):
                return str(v)

        def _fmt_interval(pair, precision=6):
            """Format a (lo, hi) interval pair."""
            if pair is None:
                return "N/A"
            lo, hi = pair
            return f"[{_fmt(lo, precision)}, {_fmt(hi, precision)}]"

        def _arr_summary(arr, label, precision=6):
            """One-line summary of a 1-D array."""
            if arr is None or len(arr) == 0:
                return f"  {label}: (none)"
            vals = ", ".join(_fmt(v, precision) for v in arr)
            return f"  {label}: {vals}"

        lines = []

        # ------------------------------------------------------------------
        # Header
        # ------------------------------------------------------------------
        # Use the same dominant-period/frequency logic as as_dict(): prefer
        # the primary peak's values so that to_text() and as_dict() always
        # describe the same dominant peak.
        _primary = self.get_primary_peak()
        _display_period = (
            _primary.period if _primary is not None else self.dominant_period
        )
        _display_frequency = (
            _primary.frequency
            if _primary is not None
            else self.dominant_frequency
        )

        lines.append("PERIOD SUMMARY")
        lines.append("==============")
        lines.append(f"  Model name          : {self.model_name or 'N/A'}")
        lines.append(f"  Method              : {self.method or 'N/A'}")
        lines.append(f"  Backend             : {self.backend or 'N/A'}")
        lines.append(
            f"  Kernel family       : {self.kernel_family or 'N/A'}"
        )
        _tkf = self.time_kernel_family or "N/A"
        lines.append(f"  Time-kernel family  : {_tkf}")
        _hsb = str(self.has_stochastic_background)
        lines.append(f"  Stochastic bg       : {_hsb}")
        lines.append(
            f"  Interval definition : {self.interval_definition or 'N/A'}"
        )
        lines.append(f"  Dominant period     : {_fmt(_display_period)}")
        lines.append(
            f"  Dominant frequency  : {_fmt(_display_frequency)}"
        )
        lines.append(f"  Peaks detected      : {self.n_peaks_detected}")
        lines.append(f"  Peaks analyzed      : {self.n_peaks_analyzed}")
        _req = (
            str(self.n_peaks_requested)
            if self.n_peaks_requested is not None
            else "N/A"
        )
        lines.append(f"  Peaks requested     : {_req}")
        if self.notes:
            lines.append(f"  Notes               : {self.notes}")
        lines.append("")

        # ------------------------------------------------------------------
        # Analyzed peaks (literature-comparable outputs)
        # ------------------------------------------------------------------
        if include_peaks and self.peaks:
            primary = self.peaks[0]

            # ---- Primary peak (full detail) ------------------------------
            lines.append("PRIMARY PEAK  (literature-comparable output)")
            lines.append("=" * 44)
            lines.append(
                f"    Period                     : {_fmt(primary.period)}"
            )
            lines.append(
                f"    Frequency                  : {_fmt(primary.frequency)}"
            )
            lines.append(
                f"    Height                     : {_fmt(primary.height)}"
            )
            lines.append(
                f"    Prominence                 : {_fmt(primary.prominence)}"
            )
            lines.append(
                f"    Area fraction              : {_fmt(primary.area_fraction)}"
            )
            lines.append(
                f"    Interval (frequency)       : "
                f"{_fmt_interval(primary.interval_frequency)}"
            )
            lines.append(
                f"    Interval (period)          : "
                f"{_fmt_interval(primary.interval_period)}"
            )
            lines.append(
                f"    LSP candidate              : {primary.is_candidate_lsp}"
            )
            if primary.notes:
                lines.append(
                    f"    Notes                      : {primary.notes}"
                )
            lines.append("")

            # ---- Additional peaks (compact) ------------------------------
            extra_peaks = self.peaks[1:]
            if extra_peaks:
                n_to_show = max(0, max_peaks_to_show - 1)
                shown = extra_peaks[:n_to_show]
                n_hidden = len(extra_peaks) - len(shown)

                if shown:
                    lines.append("ADDITIONAL PEAKS")
                    lines.append("=" * 16)
                    for pk in shown:
                        _int_str = _fmt_interval(pk.interval_period)
                        lines.append(
                            f"  #{pk.rank}  period={_fmt(pk.period)}"
                            f"  freq={_fmt(pk.frequency)}"
                            f"  area={_fmt(pk.area_fraction)}"
                            f"  interval={_int_str}"
                        )
                    if n_hidden > 0:
                        lines.append(
                            f"  (+{n_hidden} additional peak"
                            f"{'s' if n_hidden != 1 else ''} not shown)"
                        )
                    lines.append("")

        # ------------------------------------------------------------------
        # Kernel component diagnostics (NOT final periods)
        # ------------------------------------------------------------------
        if include_components and self.component_diagnostics is not None:
            diag = self.component_diagnostics
            lines.append(
                "KERNEL COMPONENT DIAGNOSTICS  "
                "(internal quantities -- not final periods)"
            )
            lines.append("=" * 60)
            lines.append(
                "  These values are derived directly from GP kernel"
                " hyperparameters."
            )
            lines.append(
                "  They are provided for diagnostics only and should not"
                " be cited"
            )
            lines.append(
                "  as literature-comparable period determinations."
            )
            lines.append("")
            lines.append(
                _arr_summary(diag.component_periods, "Component periods")
            )
            lines.append(
                _arr_summary(
                    diag.component_frequencies,
                    "Component frequencies",
                )
            )
            lines.append(
                _arr_summary(diag.component_weights, "Component weights")
            )
            lines.append(
                _arr_summary(
                    diag.component_period_scales,
                    "Component period scales",
                )
            )
            lines.append(
                _arr_summary(
                    diag.component_frequency_scales,
                    "Component frequency scales",
                )
            )
            lines.append("")

        # ------------------------------------------------------------------
        # Optional PSD grid summary (never dumps full arrays)
        # ------------------------------------------------------------------
        if include_psd_info:
            lines.append("PSD GRID INFORMATION")
            lines.append("====================")
            has_freq = self.freq_grid is not None
            has_psd = self.psd is not None
            lines.append(
                f"  Frequency grid present : {has_freq}"
            )
            lines.append(f"  PSD array present      : {has_psd}")
            if has_freq:
                try:
                    lines.append(
                        f"  Grid length            : {len(self.freq_grid)}"
                    )
                    lines.append(
                        f"  Frequency min          : "
                        f"{_fmt(float(self.freq_grid[0]))}"
                    )
                    lines.append(
                        f"  Frequency max          : "
                        f"{_fmt(float(self.freq_grid[-1]))}"
                    )
                except Exception:
                    pass
            if has_psd:
                try:
                    _psd_min = float(np.min(self.psd))
                    _psd_max = float(np.max(self.psd))
                    lines.append(f"  PSD min                : {_fmt(_psd_min)}")
                    lines.append(f"  PSD max                : {_fmt(_psd_max)}")
                except Exception:
                    pass
            lines.append("")

        return "\n".join(lines)

    def write_text(
        self,
        filename,
        include_components=True,
        include_peaks=True,
        include_psd_info=False,
    ):
        """Write a human-readable text summary to *filename*.

        Calls :meth:`to_text` and writes the result to disk.

        Parameters
        ----------
        filename : str or Path-like
            Destination file path.  The file is created or overwritten.
        include_components : bool, optional
            Forwarded to :meth:`to_text`.  Default is ``True``.
        include_peaks : bool, optional
            Forwarded to :meth:`to_text`.  Default is ``True``.
        include_psd_info : bool, optional
            Forwarded to :meth:`to_text`.  Default is ``False``.

        Returns
        -------
        pathlib.Path
            The path to the file that was written, constructed from
            *filename* via :class:`pathlib.Path`.  If *filename* is a
            relative path, the returned value is also relative.
        """
        from pathlib import Path

        path = Path(filename)
        text = self.to_text(
            include_components=include_components,
            include_peaks=include_peaks,
            include_psd_info=include_psd_info,
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def _json_serialize(self, obj):
        """Recursively convert *obj* to a JSON-serializable Python object.

        Handles nested dicts, lists/tuples, numpy arrays and scalars, and
        the standard JSON primitives.  Raises ``TypeError`` for any
        unrecognised type so that serialization bugs are caught immediately
        rather than silently corrupted via ``str()``.
        """
        if obj is None or isinstance(obj, (bool, str)):
            return obj
        if isinstance(obj, int):
            return obj
        if isinstance(obj, float):
            return None if not math.isfinite(obj) else obj
        if isinstance(obj, dict):
            return {k: self._json_serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_serialize(item) for item in obj]
        if isinstance(obj, np.ndarray):
            return self._json_serialize(obj.tolist())
        if isinstance(obj, np.floating):
            scalar = obj.item()
            return None if not math.isfinite(scalar) else scalar
        if isinstance(obj, np.integer):
            return obj.item()
        raise TypeError(
            f"Cannot JSON-serialize object of type {type(obj).__name__}"
        )

    def write_json(self, filename, include_psd=False):
        d = self.as_dict()
        # Handle freq_grid/psd before general serialization: omit them
        # unless the caller explicitly requests PSD data.
        if not include_psd or d.get("freq_grid") is None:
            d = {**d, "freq_grid": None, "psd": None}
        data = self._json_serialize(d)
        with open(filename, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, allow_nan=False)


class Lightcurve(InputHelpers, gpytorch.Module):
    """A class for storing, manipulating and fitting light curves

    This class is designed to be a convenient way to store and manipulate
    light curve data, and to fit Gaussian Processes to that data. It is
    designed to be used with the GPyTorch library, and to be compatible with
    the Pyro library for MCMC fitting.

    Parameters
    ----------
    xdata : Tensor of floats
        The independent variable data
    ydata : Tensor of floats
        The dependent variable data
    yerr : Tensor of floats, optional
        The uncertainties on the dependent variable data, by default None
    xtransform : str, optional
        The transform to apply to the x data, by default None
    ytransform : str, optional
        The transform to apply to the y data, by default None
    time_units : str, astropy.units.UnitBase, or None, optional
        Units of the time axis.  Time values are converted to days
        internally.  If *None* (default) the data are assumed to already
        be in days.
    band : array-like of str or None, optional
        Optional per-row band labels for a 2-D light curve.  Each element
        must be a string identifier (e.g. ``"V"``, ``"R"``, ``"W1"``), and
        there must be exactly one label per observation row — i.e.
        ``len(band) == len(xdata)`` for 2-D data.  For 1-D light curves a
        single-element array ``["V"]`` is accepted.  ``None`` (default)
        means no band labels are stored.

    Attributes
    ----------
    band : numpy.ndarray of str or None
        Per-row string labels aligned with ``xdata``, or ``None``
        if no labels were provided.


    Examples
    --------


    Notes
    -----
    """

    def __init__(
        self,
        xdata,
        ydata,
        yerr=None,
        xtransform=None,
        ytransform=None,
        name=None,
        time_units=None,
        max_samples: int | None = None,
        subsample_seed: int | None = None,
        check_sampling: bool = False,
        sampling_kwargs: dict | None = None,
        check_variability: bool = False,
        variability_kwargs: dict | None = None,
        band=None,
        **kwargs,
    ):
        """Initialize a Lightcurve.

        Parameters
        ----------
        xdata : torch.Tensor
            The independent variable data (time, or time + wavelength for 2-D
            light curves).
        ydata : torch.Tensor
            The dependent variable data
        yerr : torch.Tensor, optional
            The uncertainties on the dependent variable data, by default None
        xtransform : str or Transformer, optional
            The transform to apply to the x data, by default None
        ytransform : str or Transformer, optional
            The transform to apply to the y data, by default None
        name : str, optional
            A name for this light curve, by default 'Lightcurve'
        time_units : str, astropy.units.UnitBase, or None, optional
            Units of the time axis in *xdata*.  The time values will be
            converted to days internally.  Accepts any string recognised by
            ``astropy.units`` (e.g. ``'s'``, ``'hr'``, ``'yr'``, ``'days'``)
            or an ``astropy.units`` unit object.  If *None* (default) the
            data are assumed to already be in days and no conversion is
            performed.
        max_samples : int or None, optional
            Maximum number of observations to retain.  When the lightcurve
            contains more than *max_samples* points a gap-preserving random
            subsample of *max_samples* points is drawn and stored permanently
            (see :func:`~pgmuvi.preprocess.subsample_lightcurve`).  Set to
            ``None`` (default) to keep all observations.  A
            :class:`UserWarning` is issued whenever subsampling occurs.
        subsample_seed : int or None, optional
            Random seed for the subsampler.  Provide an integer for
            reproducible results; ``None`` (default) gives a non-deterministic
            subsample.  Only used when *max_samples* is set.
        check_sampling : bool, optional
            If ``True``, assess temporal sampling quality after storing the
            data.  For 1-D lightcurves a :class:`ValueError` is raised if
            sampling is poor.  For 2-D (multiband) lightcurves each band is
            checked independently: bands that fail are removed from the stored
            data with a :class:`UserWarning`, and a :class:`ValueError` is
            raised only if no bands pass.  Default is ``False``.
        sampling_kwargs : dict or None, optional
            Keyword arguments forwarded to the sampling quality gates
            (``min_points``, ``max_gap_fraction``, ``min_baseline_factor``,
            ``min_snr``, ``min_fraction_good_snr``).  Only used when
            *check_sampling* is ``True``.
        check_variability : bool, optional
            If ``True``, verify that the lightcurve shows significant
            variability after storing the data.  Raises :class:`ValueError`
            if not variable.  Only supported for 1-D lightcurves.  Default
            is ``False``.
        variability_kwargs : dict or None, optional
            Keyword arguments forwarded to the variability tests (``alpha``,
            ``fvar_min``, ``stetson_k_min``).  Only used when
            *check_variability* is ``True``.
        band : array-like of str or None, optional
            Optional per-row labels for a 2-D light curve.  Each element
            should be a string identifier (e.g. ``"V"``, ``"R"``,
            ``"W1"``).  The length must match the number of observation rows
            (``len(band) == len(xdata)`` for 2-D data, or 1 for 1-D data).
            ``None`` (default) means no band labels are stored.
        """
        super().__init__()

        transform_dic = {
            "minmax": MinMax,
            "zscore": ZScore,
            "robust_score": RobustZScore,
        }

        if xtransform is None or isinstance(xtransform, Transformer):
            self.xtransform = xtransform
        else:
            self.xtransform = transform_dic[xtransform]()

        if ytransform is None or isinstance(ytransform, Transformer):
            self.ytransform = ytransform
        else:
            self.ytransform = transform_dic[ytransform]()

        # Convert time units and coerce to tensors before non-finite filtering.
        xdata = _convert_time_to_days(xdata, time_units)
        xdata = self._ensure_tensor(xdata)
        ydata = self._ensure_tensor(ydata)
        if yerr is not None:
            yerr = self._ensure_tensor(yerr)

        # Drop rows that contain NaN or Inf in any of the data arrays so that
        # all subsequent operations (transforms, GP training, LS) see only
        # finite values.  Only applied when ydata is 1-D (the standard case
        # for all supported GP models).  Non-standard multi-dimensional ydata
        # (e.g. legacy test fixtures with shape (D, N)) bypass this step;
        # those cases rely on the existing per-setter NaN validation.
        if ydata.dim() == 1:
            xdata, ydata, yerr = self._drop_nonfinite_rows(xdata, ydata, yerr)

        self.xdata = xdata
        self.ydata = ydata
        if yerr is not None:
            self.yerr = yerr

        self.name = "Lightcurve" if name is None else name

        # ------------------------------------------------------------------
        # Band labels
        # ------------------------------------------------------------------
        if band is None:
            self.band = None
        else:
            band_arr = np.asarray(band, dtype=np.str_)
            if band_arr.ndim != 1:
                raise ValueError(
                    f"'band' must be a 1-D array-like of strings (shape (n,)); "
                    f"got shape {band_arr.shape}."
                )
            # Determine the expected length: one label per observation row for
            # 2-D data, or 1 for 1-D data (single-band lightcurve).
            if self.ndim > 1:
                n_rows = len(self._xdata_raw)
            else:
                n_rows = 1
            if len(band_arr) != n_rows:
                raise ValueError(
                    f"Length of 'band' ({len(band_arr)}) does not match the "
                    f"expected number of rows ({n_rows})."
                )
            self.band = band_arr

        self.__SET_LIKELIHOOD_CALLED = False
        self.__SET_MODEL_CALLED = False
        self.__CONTRAINTS_SET = False
        self.__PRIORS_SET = False
        self.__FITTED_MAP = False
        self.__FITTED_MCMC = False

        # ------------------------------------------------------------------
        # Sampling quality check
        # ------------------------------------------------------------------
        if check_sampling:
            sk = sampling_kwargs or {}
            if self.ndim > 1:
                xdata_raw = self._xdata_raw
                if xdata_raw.dim() != 2 or xdata_raw.shape[1] != 2:
                    raise ValueError(
                        "For 2D/multiband light curves, xdata must have shape "
                        "(N, 2) with wavelength values in column 1. Received "
                        f"shape {tuple(xdata_raw.shape)}. Please ensure "
                        "that your input is not transposed or otherwise "
                        "malformed."
                    )
                results = self.assess_sampling_quality_per_band(
                    verbose=False, **sk
                )
                failing = results["summary"]["failing_wavelengths"]
                passing = results["summary"]["passing_wavelengths"]

                for wl in failing:
                    diag = results[float(wl)]
                    warnings_str = ", ".join(diag["warnings"])
                    warnings.warn(
                        f"Skipping band \u03bb={wl} due to poor "
                        f"temporal sampling: {warnings_str}",
                        UserWarning,
                        stacklevel=2,
                    )

                if not passing:
                    raise ValueError(
                        "No wavelength bands passed sampling quality checks. "
                        "GP fitting is not recommended.\n"
                        "To force fitting anyway, use: "
                        "Lightcurve(..., check_sampling=False)"
                    )

                if failing:
                    n_pass = len(passing)
                    n_total = results["summary"]["n_bands"]
                    skipped = [round(w, 4) for w in failing]
                    _msg = (
                        f"Retaining {n_pass}/{n_total} wavelength bands after "
                        f"sampling-quality filtering (skipping \u03bb = "
                        f"{skipped})."
                    )
                    warnings.warn(_msg, UserWarning, stacklevel=2)
                    keep_mask = torch.isin(
                        xdata_raw[:, 1],
                        torch.tensor(
                            passing,
                            dtype=xdata_raw.dtype,
                            device=xdata_raw.device,
                        ),
                    )
                    self.xdata = xdata_raw[keep_mask].clone()
                    self.ydata = self._ydata_raw[keep_mask].clone()
                    if hasattr(self, "_yerr_raw"):
                        self.yerr = self._yerr_raw[keep_mask].clone()
            else:
                from pgmuvi.preprocess.quality import assess_sampling_quality

                t = self._xdata_raw.detach().cpu().numpy()
                if t.ndim > 1:
                    t = t[:, 0]
                y_np = (
                    self._ydata_raw.detach().cpu().numpy()
                    if hasattr(self, "_ydata_raw")
                    else None
                )
                yerr_np = (
                    self._yerr_raw.detach().cpu().numpy()
                    if hasattr(self, "_yerr_raw")
                    else None
                )
                passes, diag = assess_sampling_quality(
                    t, y_np, yerr_np, verbose=False, **sk
                )
                if not passes:
                    warnings_str = "\n".join(
                        f"  \u2022 {w}" for w in diag["warnings"]
                    )
                    raise ValueError(
                        f"Lightcurve has poor temporal sampling:\n"
                        f"{warnings_str}\n\n"
                        f"Recommendation: {diag['recommendation']}\n"
                        "GP fitting not recommended for poorly sampled data.\n"
                        "To force fitting anyway, use: "
                        "Lightcurve(..., check_sampling=False)"
                    )

        # ------------------------------------------------------------------
        # Variability check
        # ------------------------------------------------------------------
        if check_variability:
            from pgmuvi.preprocess.variability import is_variable

            if self.ndim > 1:
                raise ValueError(
                    "check_variability=True is not supported for multiband "
                    "(ndim > 1) lightcurves, because pooling bands may produce "
                    "misleading variability results. Use "
                    "check_variability_per_band() or filter_variable_bands() "
                    "to assess each band independently."
                )

            vkwargs = variability_kwargs or {}
            y_v, yerr_v = self._get_variability_arrays()
            is_var, diag = is_variable(y_v, yerr_v, **vkwargs)

            if not is_var:
                raise ValueError(
                    f"Lightcurve shows NO significant variability:\n"
                    f"  p-value: {diag['p_value']:.4f} "
                    f"[{'PASS' if diag['tests_passed']['chi2_test'] else 'FAIL'}]\n"
                    f"  F_var: {diag['fvar']:.4f} "
                    f"[{'PASS' if diag['tests_passed']['fvar_test'] else 'FAIL'}]\n"
                    f"  Stetson K: {diag['stetson_k']:.3f} "
                    f"[{'PASS' if diag['tests_passed']['stetson_test'] else 'FAIL'}]\n"
                    f"Decision: {diag['decision']}\n\n"
                    "GP fitting not recommended for non-variable sources.\n"
                    "To force fitting anyway, use: "
                    "Lightcurve(..., check_variability=False)"
                )

        # ------------------------------------------------------------------
        # Subsampling: permanently reduce the stored data to at most
        # max_samples observations while preserving the temporal baseline
        # and the max-gap constraint.
        # ------------------------------------------------------------------
        if max_samples is not None:
            from pgmuvi.preprocess import subsample_lightcurve

            n_total = self._xdata_raw.shape[0]
            # Subsampling is only valid for the standard (N,) or (N,2) shapes
            # where observations lie along dimension 0.  Non-standard
            # multi-dimensional ydata (e.g. shape (D, N)) would subsample the
            # wrong axis; raise a clear error rather than silently misbehaving.
            if self._ydata_raw.dim() != 1:
                raise ValueError(
                    "max_samples is only supported for standard 1-D "
                    "ydata (shape (N,)). The supplied ydata has shape "
                    f"{tuple(self._ydata_raw.shape)}."
                )
            if n_total > max_samples:
                t_np = self._xdata_raw.detach().cpu().numpy()
                if t_np.ndim > 1:
                    t_np = t_np[:, 0]
                mgf = (sampling_kwargs or {}).get("max_gap_fraction", 0.3)
                idx = subsample_lightcurve(
                    t_np,
                    max_samples=max_samples,
                    max_gap_fraction=mgf,
                    random_seed=subsample_seed,
                )
                warnings.warn(
                    f"Lightcurve has {n_total} points, which exceeds "
                    f"max_samples={max_samples}. Retaining a random subsample "
                    f"of {len(idx)} points. "
                    "Set max_samples=None to disable subsampling.",
                    UserWarning,
                    stacklevel=2,
                )
                idx_t = torch.as_tensor(
                    idx,
                    dtype=torch.long,
                    device=self._xdata_raw.device,
                )
                _buffer_names = (
                    "_xdata_raw",
                    "_xdata_transformed",
                    "_ydata_raw",
                    "_ydata_transformed",
                    "_yerr_raw",
                    "_yerr_transformed",
                )
                for bname in _buffer_names:
                    if (
                        hasattr(self, bname)
                        and getattr(self, bname) is not None
                    ):
                        self.register_buffer(
                            bname, getattr(self, bname)[idx_t]
                        )

    @classmethod
    def from_table(
        cls,
        tab,
        file_format="votable",
        xcol="x",
        ycol="y",
        yerrcol="yerr",
        bandcol=None,
        **kwargs,
    ):
        """Instantiate a Lightcurve object with
        data read in from a VOTable.

        Parameters
        ----------
        tab: astropy.table.Table object or str or pathlib.Path instance
            Table containing the input data. If str, name (with extension) of
            file containing the input data. In this case, the file_format
            keyword must be set accordingly.
        file_format: str
            Format of file containing input data. Must be a format supported
            by Table.read. Only required if type(tab) is str.
        xcol: str
            Name of column in table that contains the x data
        ycol: str
            Name of column in table that contains the y data
        yerrcol: str
            Name of column in table that contains the yerr data
        bandcol: str or None, optional
            Name of the column containing string band labels (e.g. ``"V"``,
            ``"R"``, ``"W1"``).  When provided, the per-row labels are read
            from this column and stored in :attr:`Lightcurve.band`.  If
            ``None`` (default), the method attempts to auto-detect a
            string-typed column whose name matches one of the entries in
            :attr:`_WAVELENGTH_ID_COLUMN_NAMES` (e.g. ``"band"``,
            ``"filter"``); if found it is used as the band-label column.
            The ``band`` kwarg in ``kwargs`` always takes precedence.
        kwargs:
            Arguments to be passed to the Lightcurve constructor, including
            ``time_units`` (str or ``astropy.units`` unit, default *None*).
            If ``time_units`` is provided, the time axis read from *xcol* will
            be converted to days before being stored.

        Returns
        ----------
        Lightcurve object
        """
        from pathlib import Path
        from astropy.table import Table

        if isinstance(tab, str) or isinstance(tab, Path):
            data = Table.read(tab, format=file_format)
        elif isinstance(tab, Table):
            data = tab
        else:
            raise ValueError(
                "Input tab must be an instance of str, pathlib.Path, "
                "or astropy.table.Table!"
            )
        c = data.colnames
        if xcol not in c:
            raise ValueError(f"Table does not have column '{xcol}'")
        if ycol not in c:
            raise ValueError(f"Table does not have column '{ycol}'")

        ndim = len(data[xcol].squeeze().shape)
        x = torch.Tensor(data[xcol]).squeeze()
        y = torch.Tensor(data[ycol]).squeeze()
        if (ndim == 1) or (ndim == 2):
            if yerrcol not in c:
                yerr = None
            else:
                yerr = torch.Tensor(data[yerrcol]).squeeze()
        else:
            mesg = f"Column '{xcol}' must have shape (1, nsamples) or (1, 2, nsamples)"
            raise ValueError(mesg)

        # Ensure x is shaped (N, D) rather than (D, N) before NaN filtering,
        # since some table column shapes can squeeze to (D, N).
        if x.dim() == 2 and y.dim() >= 1:
            nsamples = y.shape[0]
            if x.shape[0] != nsamples and x.shape[1] == nsamples:
                x = x.transpose(0, 1)
        if yerr is not None and yerr.dim() == 2:
            nsamples = y.shape[0]
            if yerr.shape[0] != nsamples and yerr.shape[1] == nsamples:
                yerr = yerr.transpose(0, 1)

        # Compute the finite-row mask before calling _drop_nonfinite_rows so
        # that we can apply the same filter to the ancillary band column below.
        _valid = torch.isfinite(y)
        if x.dim() > 1:
            _valid &= torch.isfinite(x).all(dim=1)
        else:
            _valid &= torch.isfinite(x)
        if yerr is not None:
            _valid &= torch.isfinite(yerr)
        _valid_np = _valid.numpy().astype(bool)

        x, y, yerr = cls._drop_nonfinite_rows(x, y, yerr)

        # ------------------------------------------------------------------
        # Band labels: only relevant when xdata is already 2-D (multiband).
        # For 1-D lightcurves there is no wavelength axis, so band labels
        # from an ancillary string column would be meaningless.
        # One label per row is stored (same length as xdata).
        # ------------------------------------------------------------------
        if "band" not in kwargs and x.dim() == 2:
            # Prefer the explicit bandcol; fall back to auto-detection.
            if bandcol is None:
                bandcol = cls._find_column(c, cls._WAVELENGTH_ID_COLUMN_NAMES)
            if bandcol is not None and bandcol in c:
                col_data = np.asarray(data[bandcol])
                col_dtype = col_data.dtype
                if (
                    np.issubdtype(col_dtype, np.str_)
                    or np.issubdtype(col_dtype, np.bytes_)
                    or col_dtype.kind == "O"
                ):
                    # String column found — store per-row labels (filtered to
                    # the same valid rows as x/y/yerr).
                    kwargs["band"] = np.array(
                        col_data[_valid_np].astype(str), dtype=np.str_
                    )

        return cls(x, y, yerr, **kwargs)

    @property
    def ndim(self):
        return self.xdata.shape[-1] if self.xdata.dim() > 1 else 1

    @property
    def magnitudes(self):
        pass

    @magnitudes.setter
    def magnitudes(self, value):
        pass

    @property
    def xdata(self):
        """The independent variable data

        :getter: Returns the independent variable data in its raw
        (untransformed) state
        :setter: Takes the input data and transforms it as requested by the
        user
        :type: torch.Tensor
        """
        return self._xdata_raw

    @xdata.setter
    def xdata(self, values):
        # first, check that the input is a tensor
        # and modifiy it if necessary
        values = self._ensure_tensor(values)
        # check that the input has more than one element
        # and raise an exception if not
        values = self._ensure_dim(values)
        # check if there are any NaNs in the inputs
        if torch.isnan(values).any():
            errmsg = f"The x values contain {torch.isnan(values).sum()} NaNs."
            raise ValueError(errmsg)
        # then, store the raw data internally
        self.register_buffer("_xdata_raw", values)
        # then, apply the transformation to the values, so it can be used to
        # train the GP
        if self.xtransform is None:
            self.register_buffer("_xdata_transformed", values)
        elif isinstance(self.xtransform, Transformer):
            self.register_buffer(
                "_xdata_transformed", self.xtransform.transform(values)
            )

    @property
    def ydata(self):
        """The dependent variable data

        :getter: Returns the dependent variable data in its raw
        (untransformed) state
        :setter: Takes the input data and transforms it as requested by the
        user
        :type: torch.Tensor"""
        return self._ydata_raw

    @ydata.setter
    def ydata(self, values):
        # first, check that the input is a tensor
        # and modifiy it if necessary
        values = self._ensure_tensor(values)
        # then, store the raw data internally
        # check if there are any NaNs in the inputs
        if torch.isnan(values).any():
            errmsg = f"The y values contain {torch.isnan(values).sum()} NaNs."
            raise ValueError(errmsg)
        self.register_buffer("_ydata_raw", values)
        # then, apply the transformation to the values
        if self.ytransform is None:
            self.register_buffer("_ydata_transformed", values)
        elif isinstance(self.ytransform, Transformer):
            self.register_buffer(
                "_ydata_transformed", self.ytransform.transform(values)
            )

    @property
    def yerr(self):
        """The uncertainties on the dependent variable data

        :getter: Returns the uncertainties on the dependent variable data in
        its raw (untransformed) state
        :setter: Takes the input data and transforms it as requested by the
        user
        :type: torch.Tensor
        """
        return self._yerr_raw

    @yerr.setter
    def yerr(self, values):
        # first, check that the input is a tensor
        # and modifiy it if necessary
        values = self._ensure_tensor(values)
        # check if there are any NaNs in the inputs
        if torch.isnan(values).any():
            errmsg = f"The y uncertainties contain {torch.isnan(values).sum()} NaNs."
            raise ValueError(errmsg)
        # then, store the raw data internally
        self.register_buffer("_yerr_raw", values)
        # now apply the same transformation that was applied to the ydata
        if self.ytransform is None:
            self.register_buffer("_yerr_transformed", values)
        elif isinstance(self.ytransform, Transformer):
            self.register_buffer("_yerr_transformed", self.ytransform.transform(values))

    def _ensure_tensor(self, values):
        # Ensures that the input data has type torch.Tensor
        # Transforms the data if necessary
        if not isinstance(values, torch.Tensor):
            warnings.warn(
                (
                    "The function expects a torch.Tensor as input."
                    "Your data will be converted to a tensor."
                ),
                stacklevel=2,
            )
            values = torch.as_tensor(values, dtype=torch.float32)
        return values

    def _ensure_dim(self, values):
        # Ensures that the input data has more than one element
        # Returns an exception if not
        if values.numel() == 1:
            raise ValueError("The input data must have more than one element.")
        # elif values.numel() < threshold:
        #    warnings.warn(('The input data has less than threshold elements.'
        #        'This may lead to poor performance.'),
        #        stacklevel=2)
        return values

    def append_data(self, new_values_x, new_values_y):
        pass

    def select_bands(
        self, bands: list | tuple | np.ndarray
    ) -> "Lightcurve":
        """Return a new Lightcurve containing only the requested bands.

        Parameters
        ----------
        bands : list, tuple, or numpy.ndarray
            Selection criteria.  Each element may be:

            * A **string** — matched against :attr:`band` (the per-row string
              label array).  Requires that :attr:`band` is not ``None``.
            * A **float** or **int** — matched against ``xdata[:, 1]`` (the
              numeric wavelength column).  Exact equality is used.  ``NaN``
              values are not accepted.

            Mixed inputs (some strings, some floats) are supported; the row
            mask is the logical OR of all individual matches.

        Returns
        -------
        Lightcurve
            A new :class:`Lightcurve` object built from the subset of rows
            that match at least one of the requested *bands*.  The
            :attr:`name`, :attr:`xtransform`, and :attr:`ytransform`
            attributes are inherited from the original light curve.

        Raises
        ------
        ValueError
            If the light curve is 1-D (no wavelength axis).
        ValueError
            If any string selector is requested but :attr:`band` is ``None``.
        ValueError
            If a numeric selector is ``NaN``.
        TypeError
            If *bands* is a bare string rather than a sequence of selectors.
        TypeError
            If any element of *bands* is neither a string nor a number.
        """
        if isinstance(bands, str):
            raise TypeError(
                "'bands' must be a sequence of selectors (list, tuple, or "
                "numpy.ndarray), not a bare string. "
                "To select a single band wrap it in a list: "
                f"select_bands([{bands!r}])"
            )

        if self.ndim < 2:
            raise ValueError(
                "select_bands requires a 2-D light curve "
                "(xdata must have shape (N, 2) with a wavelength column)."
            )

        str_vals = []
        float_vals = []
        for b in bands:
            if isinstance(b, str):
                str_vals.append(b)
            elif isinstance(b, (int, float, np.floating, np.integer)):
                fv = float(b)
                if np.isnan(fv):
                    raise ValueError(
                        "NaN is not a valid wavelength selector in 'bands'."
                    )
                float_vals.append(fv)
            else:
                raise TypeError(
                    f"Each element of 'bands' must be a string or a number; "
                    f"got {type(b).__name__!r}."
                )

        xdata_raw = self._xdata_raw
        n = xdata_raw.shape[0]
        mask = torch.zeros(n, dtype=torch.bool, device=xdata_raw.device)

        if str_vals:
            if self.band is None:
                raise ValueError(
                    "String band selectors require the 'band' attribute to be "
                    "set, but this Lightcurve has band=None."
                )
            for s in str_vals:
                mask |= torch.as_tensor(
                    self.band == s, dtype=torch.bool, device=xdata_raw.device
                )

        if float_vals:
            wl_col = xdata_raw[:, 1]
            for fv in float_vals:
                mask |= wl_col == fv

        new_x = xdata_raw[mask]
        new_y = self._ydata_raw[mask]
        new_yerr = (
            self._yerr_raw[mask] if hasattr(self, "_yerr_raw") else None
        )
        new_band = (
            self.band[mask.cpu().numpy().astype(bool)]
            if self.band is not None
            else None
        )

        return Lightcurve(
            new_x,
            new_y,
            yerr=new_yerr,
            xtransform=self.xtransform,
            ytransform=self.ytransform,
            name=self.name,
            band=new_band,
        )

    def transform_x(self, values):
        if self.xtransform is None:
            return values
        elif isinstance(self.xtransform, Transformer):
            return self.xtransform.transform(values)

    def transform_y(self, values):
        if self.ytransform is None:
            return values
        elif isinstance(self.xtransform, Transformer):
            return self.xtransform.transform(values)

    def set_likelihood(self, likelihood=None, variance=False, **kwargs):
        """Set the likelihood function for the model

        Parameters
        ----------
        likelihood : string, None or instance of
                     gpytorch.likelihoods.likelihood.Likelihood or Constraint,
                     optional
            The likelihood function to use for the GP, by default None.

            If ``likelihood`` is ``None`` and per-point uncertainties on the
            data are available (i.e. ``yerr`` has been set), a
            :class:`gpytorch.likelihoods.FixedNoiseGaussianLikelihood` is
            constructed and given a tensor of noise *variances* derived from
            those uncertainties. If ``likelihood`` is ``None`` and no
            uncertainties are available, a standard
            :class:`gpytorch.likelihoods.GaussianLikelihood` is used.

            If a string, it must be ``'learn'``. When ``'learn'`` is used and
            uncertainties are available, a
            :class:`gpytorch.likelihoods.FixedNoiseGaussianLikelihood` is
            created with the same noise variance tensor and with
            ``learn_additional_noise=True``. If ``'learn'`` is used and no
            uncertainties are available, a
            :class:`gpytorch.likelihoods.GaussianLikelihood` is created with
            ``learn_additional_noise=True``.

            If an instance of a :class:`~gpytorch.likelihoods.likelihood.Likelihood`
            object is passed, that object is used directly. If a Constraint
            object is passed, a :class:`gpytorch.likelihoods.GaussianLikelihood`
            is constructed with the constraint passed as the
            ``noise_constraint`` argument; this overrides any other keyword
            arguments to the likelihood.

            You can also provide a likelihood *class* (rather than an
            instance), in which case the class will be instantiated with the
            provided ``kwargs`` under the assumption that it is a
            :class:`~gpytorch.likelihoods.likelihood.Likelihood` subclass. If
            per-point uncertainties are available, a first positional argument
            will also be passed containing the noise tensor. This tensor is in
            units of variance by default (see the ``variance`` parameter
            below).
        variance : bool, optional
            Controls how stored per-point uncertainties are interpreted when
            constructing the noise tensor passed to likelihoods that require
            per-observation noise (e.g. :class:`gpytorch.likelihoods.
            FixedNoiseGaussianLikelihood` or user-supplied likelihood classes
            that accept a noise argument).

            If ``False`` (default), the uncertainties stored in the lightcurve
            are assumed to be errors (standard deviations). They are squared
            to produce noise *variances* before being passed to the likelihood.

            If ``True``, the stored uncertainties are assumed to already be
            variances and are passed through unchanged as the noise tensor.
        """

        # Prepare the noise tensor: gpytorch likelihoods expect variances.
        # By default (variance=False) we square the stored errors; if the
        # caller has already supplied variances, we use them as-is.
        _has_noise = hasattr(self, "_yerr_transformed")
        if _has_noise:
            noise = (
                self._yerr_transformed
                if variance
                else self._yerr_transformed ** 2
            )

        if _has_noise and likelihood is None:
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise
            )
        elif _has_noise and likelihood == "learn":
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise,
                learn_additional_noise=True,
            )
        elif likelihood == "learn":
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                learn_additional_noise=True
            )
        elif "Constraint" in [t.__name__ for t in type(likelihood).__mro__]:
            # In this case, the likelihood has been passed a constraint, which
            # means we want a constrained GaussianLikelihood
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=likelihood
            )
        elif likelihood is None:
            # We're just going to make the simplest assumption
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # Also add a case for if it is a Likelihood object
        elif isinstance(likelihood, gpytorch.likelihoods.likelihood.Likelihood):
            self.likelihood = likelihood
        elif isclass(likelihood):
            if _has_noise:
                self.likelihood = likelihood(noise, **kwargs)
            else:
                self.likelihood = likelihood(**kwargs)
        else:
            raise ValueError(
                f"""Expected a string, a constraint, a Likelihood
                              instance or a class to be instantiated as a
                              Likelihood instance, but got {type(likelihood)}.
                              Please provide a suitable likelihood input."""
            )
        self.__SET_LIKELIHOOD_CALLED = True

    def set_model(
        self, model=None, likelihood=None, num_mixtures=None, variance=False, **kwargs
    ):
        """Set the model for the lightcurve

        Parameters
        ----------
        model : string or instance of gpytorch.models.GP, optional
            The model to use for the GP, by default None. If None, an
            error will be raised. If a string, it must be one of the
            following:

            Spectral mixture models (default):
                '1D': SpectralMixtureGPModel
                '2D': TwoDSpectralMixtureGPModel
                '1DLinear': SpectralMixtureLinearMeanGPModel
                '2DLinear': TwoDSpectralMixtureLinearMeanGPModel
                '1DSKI': SpectralMixtureKISSGPModel
                '2DSKI': TwoDSpectralMixtureKISSGPModel
                '1DLinearSKI': SpectralMixtureLinearMeanKISSGPModel
                '2DLinearSKI': TwoDSpectralMixtureLinearMeanKISSGPModel
                '2DPowerLaw': TwoDSpectralMixturePowerLawMeanGPModel
                '2DPowerLawSKI': TwoDSpectralMixturePowerLawMeanKISSGPModel
                '2DDust': TwoDSpectralMixtureDustMeanGPModel
                '2DDustSKI': TwoDSpectralMixtureDustMeanKISSGPModel

            Alternative 1D models:
                '1DQuasiPeriodic': QuasiPeriodicGPModel
                '1DMatern': MaternGPModel
                '1DPeriodicStochastic': PeriodicPlusStochasticGPModel
                '1DLinearQuasiPeriodic': LinearMeanQuasiPeriodicGPModel

            Separable 2D models:
                '2DSeparable': SeparableGPModel
                '2DAchromatic': AchromaticGPModel
                '2DWavelengthDependent': WavelengthDependentGPModel
                '2DDustMean': DustMeanGPModel
                '2DPowerLawMean': PowerLawMeanGPModel


            If an instance of a GP class, that object will be used.
            _description_, by default None
        likelihood : string, None or instance of
                     gpytorch.likelihoods.likelihood.Likelihood or Constraint,
                     optional
            If likelihood is passed, it will be passed along to `set_likelihood()`
            and used to set the likelihood function for the model. For details, see
            the documentation for `set_likelihood()`.
        num_mixtures : int, optional
            The number of mixtures to use in the spectral mixture kernel, by
            default None. If None, a default value will be used. This value
            is passed to the constructor for the model if a string is passed
            as the model argument.
        variance : bool, optional
            Passed to `set_likelihood()`.  If False (default), stored
            uncertainties are treated as errors and squared before being used
            as noise variances.  Set to True if the stored uncertainties
            already represent variances.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the model constructor.
        """
        self.__SET_MODEL_CALLED = True
        if isinstance(model, str):
            self._model_str = model
            self._model_instance = None
        elif "GP" in [t.__name__ for t in type(model).__mro__]:
            # GP instance provided directly — store it so it can be rebound
            # to new training data after band filtering.
            self._model_instance = model
            self._model_str = None
        else:
            self._model_str = None
            self._model_instance = None
        self._model_num_mixtures = num_mixtures
        self._fit_num_mixtures_effective = num_mixtures
        self._fit_num_mixtures_requested = num_mixtures
        model_dic_1 = {
            "2D": TwoDSpectralMixtureGPModel,
            "1D": SpectralMixtureGPModel,
            "1DLinear": SpectralMixtureLinearMeanGPModel,
            "2DLinear": TwoDSpectralMixtureLinearMeanGPModel,
            "2DPowerLaw": TwoDSpectralMixturePowerLawMeanGPModel,
            "2DDust": TwoDSpectralMixtureDustMeanGPModel,
        }

        model_dic_2 = {
            "1DSKI": SpectralMixtureKISSGPModel,
            "2DSKI": TwoDSpectralMixtureKISSGPModel,
            "1DLinearSKI": SpectralMixtureLinearMeanKISSGPModel,
            "2DLinearSKI": TwoDSpectralMixtureLinearMeanKISSGPModel,
            "2DPowerLawSKI": TwoDSpectralMixturePowerLawMeanKISSGPModel,
            "2DDustSKI": TwoDSpectralMixtureDustMeanKISSGPModel,
        }

        # Alternative kernel models — do not require num_mixtures
        model_dic_alt = {
            "1DQuasiPeriodic": QuasiPeriodicGPModel,
            "1DMatern": MaternGPModel,
            "1DPeriodicStochastic": PeriodicPlusStochasticGPModel,
            "1DLinearQuasiPeriodic": LinearMeanQuasiPeriodicGPModel,
            "2DSeparable": SeparableGPModel,
            "2DAchromatic": AchromaticGPModel,
            "2DWavelengthDependent": WavelengthDependentGPModel,
            "2DDustMean": DustMeanGPModel,
            "2DPowerLawMean": PowerLawMeanGPModel,
        }

        if not hasattr(self, "likelihood"):
            self.set_likelihood(likelihood, variance=variance, **kwargs)
        elif not self.__SET_LIKELIHOOD_CALLED and likelihood is None:
            # if no likelihood is passed, we only want to set the likelihood
            # if it hasn't already been set
            self.set_likelihood(likelihood, variance=variance, **kwargs)
        elif likelihood is not None:
            self.set_likelihood(likelihood, variance=variance, **kwargs)

        if "GP" in [t.__name__ for t in type(model).__mro__]:
            # check if it is or inherets from a GPyTorch model
            self.model = model
        elif model in model_dic_1:
            self.model = model_dic_1[model](
                self._xdata_transformed,
                self._ydata_transformed,
                self.likelihood,
                num_mixtures=num_mixtures,
                **kwargs,
            )
        elif model in model_dic_2:
            self.model = model_dic_2[model](
                self._xdata_transformed,
                self._ydata_transformed,
                self.likelihood,
                num_mixtures=num_mixtures,
                **kwargs,
            )
        elif model in model_dic_alt:
            self.model = model_dic_alt[model](
                self._xdata_transformed,
                self._ydata_transformed,
                self.likelihood,
                num_mixtures=num_mixtures,
                **kwargs,
            )
        else:
            raise ValueError("Insert a valid model")

        # now we've got a model set up, we're going to make some handy lookups
        # for the parameters and modules that we'll need to access later
        self._make_parameter_dict()
        # self.set_default_constraints()

    def _make_parameter_dict(self):
        """Make a dictionary of the model parameters

        This function is used to make a dictionary of the model parameters,
        providing a convenient way to access them. The dictionary is stored
        in the _model_pars attribute.
        """
        self._model_pars = {}
        # there are a few parameters that we want to make sure we expose a
        # direct link to if we need them!
        _special_pars = ["noise", "mixture_means", "mixture_scales", "mixture_weights"]

        for param_name, param in self.model.named_parameters():
            comps = list(param_name.split("."))
            pn_root = comps[-1]
            param_dict = {
                "full_name": param_name,
                "root_name": pn_root,
                "chain": [],
                "constrained": False,
            }
            if "raw" in param_name:
                # This is a constrained parameter, so we need to get the
                # unconstrained value
                pn_const = comps[-1].lstrip("raw_")
                param_dict["constrained"] = True
                param_dict["constrained_name"] = pn_const
                pn = ".".join([c.lstrip("raw_") for c in comps])
                param_dict["constrained_full_name"] = pn
            tmp = self.model.__getattr__(comps[0])
            param_dict["chain"].append(tmp)
            for i in range(1, len(comps)):
                c = comps[i] if "raw" not in comps[i] else comps[i].lstrip("raw_")
                try:
                    tmp = tmp.__getattr__(c)
                except AttributeError:
                    tmp = tmp.__getattribute__(c)
                param_dict["chain"].append(tmp)
            param_dict["module"] = param_dict["chain"][-2]
            if param_dict["constrained"]:
                param_dict["param"] = param_dict["chain"][-1]
                try:
                    param_dict["raw_param"] = param_dict["chain"][-2].__getattr__(
                        comps[-1]
                    )
                except AttributeError:
                    param_dict["raw_param"] = param_dict["chain"][-2].__getattribute__(
                        comps[-1]
                    )
            else:
                param_dict["param"] = param_dict["chain"][-1]
                param_dict["raw_param"] = param_dict["param"]
            if any(s in pn_root for s in _special_pars):
                # it's a special parameter that we want extra easy access to!
                param_dict["special"] = True
                j = np.argmax([s in pn_root for s in _special_pars])
                self._model_pars[_special_pars[j]] = param_dict

            #     pars[pn] = tmp.data
            # else:
            #     # Either we actually want the raw values, or it's not a
            #     # constrained parameter
            #     pars[param_name] = param.data
            self._model_pars[param_name] = param_dict
            if param_dict["constrained"]:
                self._model_pars[pn] = param_dict  # so we also alias the full
                # name for the constrained
                # parameter

    def set_prior(self, prior=None, **kwargs):
        """Set the prior for the model parameters

        Parameters
        ----------
        prior : dict, optional
            A dictionary of the priors to use for the model parameters. The
            keys should be the names of the parameters, and the values should
            be instances of gpytorch.priors.Prior. If None, no priors will be
            used. If a prior is passed for a parameter that is not a model
            parameter, it will be ignored.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the Prior constructors.
        """
        self.__PRIORS_SET = True
        pass

    def set_constraint(self, constraint, debug=False, **kwargs):
        """Set the constraint for the model parameters

        Parameters
        ----------
        constraint : dict, optional
            A dictionary of the constraints to use for the model parameters.
            The keys should be the names of the parameters, and the values
            should be instances of gpytorch.constraints.Constraint. If None, no
            constraints will be used. If a constraint is passed for a parameter
            that is not a model parameter, it will be ignored.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the Constraint
            constructors.
        """
        # which paramaters need to have their constraints transformed? and how?
        pars_to_transform = {
            "x": ["mixture_means", "mixture_scales"],
            "y": ["noise", "mean_module"],
        }

        for key in constraint:
            if key in self._model_pars:
                if debug:
                    print(f"Found parameter {key} in model parameters")
                    print(f"Parameter {key} will have constraint: {constraint[key]}")
                    print("which may be transformed")
                # constraints must be registered to raw parameters!
                k = key.split(".")[-1] if "raw_" in key else f"raw_{key.split('.')[-1]}"
                if all(
                    p not in key
                    for p in pars_to_transform["y"] + pars_to_transform["x"]
                ):  # no transform needed!
                    self._model_pars[key]["module"].register_constraint(
                        k, constraint[key]
                    )
                elif any(p in key for p in pars_to_transform["x"]):
                    # now apply the x transform
                    # remember that the means and scales are in fourier space
                    # so we need to transform them back to real space
                    # before applying the transform
                    # and then transform them back to fourier space
                    # luckily, when the shift is removed from the transform,
                    # the factors of 2pi cancel out for the scales
                    # so we can just do 1/ for both means and scales
                    if self.xtransform is not None:
                        # now things get complicated...
                        # if we have gotten to here, we know that the parameter
                        # is a mixture mean or scale, so we need to transform
                        # it to real space, apply the constraint, and then
                        # transform it back to fourier space
                        # luckily, when the shift is removed from the transform,
                        # the factors of 2pi cancel out for the scales
                        # so we can just do 1/ for both means and scales
                        if debug:
                            print(constraint[key])
                        if constraint[key].lower_bound not in [
                            torch.tensor(0),
                            torch.tensor(-torch.inf),
                        ]:
                            # we need to transform the lower bound
                            # NOTE: For 2D data, GPyTorch constraints are scalar
                            # and apply element-wise to all parameter elements
                            # (time and wavelength).
                            # We transform using dimension 0 (time) as it's
                            # typically the primary independent variable.
                            # Users setting manual constraints should be aware
                            # that the same constraint applies to both
                            # dimensions.
                            transformed_bound = 1.0 / self.xtransform.transform(
                                1.0 / constraint[key].lower_bound, shift=False
                            )
                            # Handle both 1D and 2D cases
                            if transformed_bound.numel() > 1:
                                # For 2D case, use the first dimension's
                                # transformation. Take element [0, 0] to get
                                # a scalar
                                transformed_bound = transformed_bound.flatten()[0]
                            if debug:
                                print(f"Transformed lower bound: {transformed_bound}")
                            constraint[key].lower_bound = torch.tensor(
                                transformed_bound.item()
                            )
                            if debug:
                                print(constraint[key].lower_bound)
                                print(constraint[key])
                        if constraint[key].upper_bound not in [
                            torch.tensor(0),
                            torch.tensor(torch.inf),
                        ]:
                            # we need to transform the upper bound
                            # (Same dimension-0 transformation logic as
                            # lower_bound above)
                            transformed_bound = 1.0 / self.xtransform.transform(
                                1.0 / constraint[key].upper_bound, shift=False
                            )
                            # Handle both 1D and 2D cases
                            if transformed_bound.numel() > 1:
                                # For 2D case, use the first dimension's
                                # transformation. Take element [0, 0] to get
                                # a scalar
                                transformed_bound = transformed_bound.flatten()[0]
                            constraint[key].upper_bound = torch.tensor(
                                transformed_bound.item()
                            )
                            if debug:
                                print(constraint[key].upper_bound)
                                print(constraint[key])
                        if debug:
                            print(constraint[key])
                    self._model_pars[key]["module"].register_constraint(
                        k, constraint[key]
                    )
                elif any(p in key for p in pars_to_transform["y"]):
                    if self.ytransform is not None:
                        if debug:
                            print(constraint[key])
                        if isinstance(constraint[key], Positive) and (
                            isinstance(self.ytransform, ZScore | RobustZScore)
                        ):
                            # convert constraint to an interval with minimum equal to
                            # what zero is in the untransformed space
                            constraint[key] = Interval(
                                self.ytransform.transform(0), torch.inf
                            )
                            if debug:
                                print(constraint[key])

                        elif constraint[key].lower_bound not in [
                            torch.tensor(0),
                            torch.tensor(-torch.inf),
                        ]:
                            # we need to transform the lower bound
                            constraint[key].lower_bound = torch.tensor(
                                self.ytransform.transform(
                                    constraint[key].lower_bound
                                ).item()
                            )
                            if debug:
                                print(constraint[key].lower_bound)
                                print(constraint[key])
                        if constraint[key].upper_bound not in [
                            torch.tensor(0),
                            torch.tensor(torch.inf),
                        ]:
                            # we need to transform the upper bound
                            constraint[key].upper_bound = torch.tensor(
                                self.ytransform.transform(
                                    constraint[key].upper_bound
                                ).item()
                            )
                            if debug:
                                print(constraint[key].upper_bound)
                                print(constraint[key])
                    if debug:
                        print(constraint[key])
                    self._model_pars[key]["module"].register_constraint(
                        k, constraint[key]
                    )
                if debug:
                    try:
                        print(f"Registered constraint {constraint[key]}")
                    except TypeError:
                        print("Registered constraint")
                        print(constraint[key])
                    print(f"to parameter {key}")
            else:
                print(f"Parameter {key} not found in model parameters,")
                print("this constraint will be ignored.")
                print("Available parameters are:")
                print(self._model_pars.keys())
                print("(Beware, several of these are aliases!)")

    def set_default_priors(self, prior_set=None, **kwargs):
        """Set the default priors for the model and likelihood parameters

        The default priors are as follows:
            - Parameters that must be positive are given LogNormal, HalfNormal
            or HalfCauchy priors, depending on the parameter.
            - The noise is given a HalfNormal prior with a scale of 1/10 of the
            smallest uncertainty on the y-data, if uncertainties are given, or
            1/10 of the standard deviation of the y data.
            - The mean of the GP is given a Gaussian prior with a mean of the
            mean of the y-data and a standard deviation of 1/10 of the standard
            deviation of the y-data.
            - For spectral-mixture models the mixture means, scales and weights
            receive LogNormal(0, 1) priors.
            - If *prior_set* is provided, :meth:`set_period_prior` is called
            with that prior set to register a physically motivated prior on
            the period/frequency parameter of the model.

        Parameters
        ----------
        prior_set : str or None, optional
            Name of a predefined prior set to apply to the period/frequency
            parameter (e.g. ``"LPV"``).  If ``None`` (default), no period
            prior is added.  See :meth:`set_period_prior` and
            :data:`~pgmuvi.priors.PRIOR_SETS` for available options.
        **kwargs : dict, optional
            Any keyword arguments to be passed to the Prior constructors.
        """

        # Gpytorch currently crashes if you try to do MCMC while learning additional
        # diagonal noise with the FixedNoiseGaussianLikelihood. So we only need to
        # set priors for the noise if we don't have uncertainties on the data.
        if not hasattr(self, "_yerr_transformed"):
            try:
                noise_scale = np.minimum(1e-4, self._yerr_transformed.min() / 10)
            except AttributeError:
                noise_scale = 1e-4 * self._ydata_transformed.std()
            # noise_prior = gpytorch.priors.HalfCauchyPrior(noise_scale)
            noise_prior = gpytorch.priors.LogNormalPrior(
                torch.log(noise_scale), noise_scale
            )
            self._model_pars["noise"]["module"].register_prior(
                "noise_prior", noise_prior, "noise"
            )
        with contextlib.suppress(RuntimeError):
            mean_prior = gpytorch.priors.NormalPrior(
                self._ydata_transformed.mean(), self._ydata_transformed.std() / 10
            )
            for key in self._model_pars:
                if "mean_module.constant" in key:
                    self._model_pars[key]["module"].register_prior(
                        "mean_prior", mean_prior, "constant"
                    )
        # we use a lognormal prior for the means, because we want to make sure
        # that the means are positive, but we don't want to restrict them to
        # be close to zero. In fact, we want to penalise both very high and very low
        # frequencies, so we use a lognormal prior with mu = 0 and sigma = 1
        if "mixture_means" in self._model_pars:
            mixture_means_prior = gpytorch.priors.LogNormalPrior(
                0, 1
            )  # /self._xdata_transformed.max())
            self._model_pars["mixture_means"]["module"].register_prior(
                "mixture_means_prior", mixture_means_prior, "mixture_means"
            )

        # now we need a prior for the mixture scales
        # we want to penalise very large scales, so we use a half-cauchy prior
        # with a scale of 1/10 of the maximum frequency
        # mixture_scales_prior = gpytorch.priors.HalfCauchyPrior(1/self._xdata_transformed.max())  # noqa: E501
        if "mixture_scales" in self._model_pars:
            mixture_scales_prior = gpytorch.priors.LogNormalPrior(
                0, 1
            )  # 1/self._xdata_transformed.max())
            self._model_pars["mixture_scales"]["module"].register_prior(
                "mixture_scales_prior", mixture_scales_prior, "mixture_scales"
            )
        # we use a LogNormal prior for the mixture weights, because we want to
        # make sure that they are positive (but never zero) and we don't want
        # to restrict them to be close to zero. In fact, we want to penalise
        # both very high and very low weights, so we use a LogNormal prior
        # with a scale of 1/10 of the maximum frequency
        if "mixture_weights" in self._model_pars:
            mixture_weights_prior = gpytorch.priors.LogNormalPrior(
                0, 1
            )  # 1/self._xdata_transformed.max())
            self._model_pars["mixture_weights"]["module"].register_prior(
                "mixture_weights_prior", mixture_weights_prior, "mixture_weights"
            )

        # Apply a period/frequency prior if a prior_set is requested
        if prior_set is not None:
            self.set_period_prior(prior_set=prior_set)

        # need a more general way to assign default priors to everything, but for now
        # let's see if this works!
        self.__PRIORS_SET = True

    def get_priors(self):
        """Return the priors currently registered on the model.

        Iterates over all priors registered on the model (via GPyTorch's
        ``named_priors``) and returns a dictionary mapping each prior name to
        the corresponding prior object.  A formatted summary is also printed to
        standard output.

        Returns
        -------
        dict
            A dictionary mapping prior names (str) to their
            :class:`gpytorch.priors.Prior` objects. Returns an empty dict if
            no priors have been registered.

        Raises
        ------
        RuntimeError
            If the model has not been set yet (call :meth:`set_model` first).

        See Also
        --------
        set_default_priors
        get_period_prior

        Examples
        --------
        ::

            lc.set_model("1D", num_mixtures=4)
            lc.set_default_priors()
            priors = lc.get_priors()
        """
        if not hasattr(self, "_model_pars"):
            raise RuntimeError(
                "Model has not been set yet. Call set_model() before "
                "get_priors()."
            )
        priors = {}
        for name, _module, prior, _closure, _setting_closure in (
            self.model.named_priors()
        ):
            priors[name] = prior
        print("Registered priors:")
        if priors:
            for name, prior in priors.items():
                print(f"  {name}: {prior}")
        else:
            print("  (none)")
        return priors

    def set_period_prior(
        self,
        prior_set=None,
        prior_type="lognormal",
        mu=5.0,
        sigma=1.0,
        mean=300.0,
        std=75.0,
        lower_period=None,
        upper_period=None,
        period=True,
    ):
        """Set a prior on the period or frequency parameter of the model.

        This method detects whether the model represents periodicity as a
        period (e.g. ``period_length`` in
        :class:`~pgmuvi.gps.QuasiPeriodicGPModel`) or as a frequency (e.g.
        ``mixture_means`` in
        :class:`~pgmuvi.gps.SpectralMixtureGPModel`) and registers an
        appropriate prior on the relevant parameter.

        For frequency-based models the period-space prior is transformed to
        frequency space with the correct change-of-variables Jacobian (see
        :class:`~pgmuvi.priors.LogNormalFrequencyPrior` and
        :class:`~pgmuvi.priors.NormalFrequencyPrior`).

        Models with no periodicity parameter (e.g.
        :class:`~pgmuvi.gps.MaternGPModel`) are silently skipped with a
        warning.

        Parameters
        ----------
        prior_set : str or None, optional
            Name of a predefined prior set (e.g. ``"LPV"``).  When given,
            the ``prior_type``, ``mu``, ``sigma``, ``mean``, ``std`` and
            ``lower_period`` / ``upper_period`` defaults are taken from
            :data:`~pgmuvi.priors.PRIOR_SETS`.  Any explicitly supplied
            keyword arguments override the prior-set values.
        prior_type : str, optional
            Which prior family to use.  Either ``"lognormal"`` (the default,
            LogNormal in period space with ``mu`` and ``sigma``) or
            ``"normal"`` (Normal in period space with ``mean`` and ``std``).
            Case-insensitive.
        mu : float, optional
            Mean of the underlying normal distribution for the Log-Normal
            period prior (i.e. the log-mean).  Default ``5.0``
            (median period ≈ 148 days).  Dimensionless (logarithmic units).
        sigma : float, optional
            Standard deviation of the underlying normal distribution for the
            Log-Normal period prior (i.e. the log-standard-deviation).
            Default ``1.0``.  Dimensionless (logarithmic units).
        mean : float, optional
            Mean for the Normal period prior (days).  Default ``300.0``.
        std : float, optional
            Standard deviation for the Normal period prior (days).
            Default ``75.0``.
        lower_period : float or None, optional
            Lower bound on period.  When ``period=True`` (default), this is
            in days (the assumed time unit of the data).  When ``period=False``
            this is a lower bound in frequency units (1/days).
            Values outside this bound receive ``-inf`` log-prob.
            If ``None`` and a *prior_set* is provided, the bound is taken
            from the prior set.
        upper_period : float or None, optional
            Upper bound on period (days when ``period=True``, 1/days
            when ``period=False``).
        period : bool, optional
            Controls the interpretation of ``lower_period`` and
            ``upper_period`` for *frequency-parameterised* models (i.e.
            spectral-mixture models whose periodicity is encoded as
            ``mixture_means``).  If ``True`` (default), bounds are in period
            units (days).  If ``False``, bounds are in frequency units
            (1/days).  Has no effect for period-parameterised models
            (e.g. ``QuasiPeriodicGPModel``), which always use period units.

        Raises
        ------
        ValueError
            If *prior_set* is not a recognised name or if *prior_type* is not
            ``"lognormal"`` or ``"normal"``.

        Notes
        -----
        The model must have been set (via :meth:`set_model` or :meth:`fit`)
        before calling this method.

        For spectral-mixture models the prior is registered on
        ``mixture_means`` and applies element-wise to all mixture
        components.

        For quasi-periodic models the prior is registered on each
        ``period_length`` parameter found in the model (there is typically
        only one).

        See Also
        --------
        pgmuvi.priors.LogNormalPeriodPrior
        pgmuvi.priors.LogNormalFrequencyPrior
        pgmuvi.priors.NormalPeriodPrior
        pgmuvi.priors.NormalFrequencyPrior
        pgmuvi.priors.PRIOR_SETS

        Examples
        --------
        Set the LPV default prior on a spectral-mixture model::

            lc.set_model("1D", num_mixtures=4)
            lc.set_period_prior(prior_set="LPV")

        Set a Normal period prior explicitly::

            lc.set_model("1DQuasiPeriodic")
            lc.set_period_prior(prior_type="normal", mean=300.0, std=75.0,
                                lower_period=100.0)
        """
        if not hasattr(self, "_model_pars"):
            raise RuntimeError(
                "Model has not been set yet. Call set_model() before "
                "set_period_prior()."
            )

        # Normalise prior_type to lower case so callers can use any case
        prior_type = prior_type.lower()

        # If bounds are in frequency units, convert to period units now so the
        # rest of the logic always works in period space.  The prior_set always
        # stores bounds in period space, so we only convert user-provided bounds.
        if not period:
            # lower frequency ↔ upper period, and vice versa
            if lower_period is not None and lower_period <= 0:
                raise ValueError(
                    f"lower_period as a frequency bound must be positive, "
                    f"got {lower_period}."
                )
            if upper_period is not None and upper_period <= 0:
                raise ValueError(
                    f"upper_period as a frequency bound must be positive, "
                    f"got {upper_period}."
                )
            lower_period, upper_period = (
                (1.0 / upper_period if upper_period is not None else None),
                (1.0 / lower_period if lower_period is not None else None),
            )

        # --- Resolve prior-set defaults ---
        if prior_set is not None:
            ps = get_prior_set(prior_set)
            # Use prior-set values as defaults; explicit kwargs override them
            if prior_type == "lognormal":
                mu = ps["lognormal"].get("mu", mu)
                sigma = ps["lognormal"].get("sigma", sigma)
            elif prior_type == "normal":
                mean = ps["normal"].get("mean", mean)
                std = ps["normal"].get("std", std)
            pb = ps["period_bounds"]
            if lower_period is None:
                lower_val, lower_active = pb["lower"]
                lower_period = lower_val if lower_active else None
            if upper_period is None:
                upper_val, upper_active = pb["upper"]
                upper_period = upper_val if upper_active else None

        if prior_type not in ("lognormal", "normal"):
            raise ValueError(
                f"prior_type must be 'lognormal' or 'normal', got {prior_type!r}"
            )

        # --- Compute scale factor to convert raw period bounds to model space ---
        # In model (transformed) space the period is related to the raw period by
        #   period_model = period_raw * (x_trans_span / x_orig_span)
        # For linear transforms this equals period_raw when no transform is used.
        if self.ndim > 1:
            x_orig_span = float(
                self._xdata_raw[:, 0].max() - self._xdata_raw[:, 0].min()
            )
            x_trans_span = float(
                self._xdata_transformed[:, 0].max()
                - self._xdata_transformed[:, 0].min()
            )
        else:
            if hasattr(self, "_xdata_raw"):
                x_orig_span = float(
                    self._xdata_raw.max() - self._xdata_raw.min()
                )
            else:
                x_orig_span = float(
                    self._xdata_transformed.max() - self._xdata_transformed.min()
                )
            x_trans_span = float(
                self._xdata_transformed.max() - self._xdata_transformed.min()
            )
        period_scale = (
            x_trans_span / x_orig_span if x_orig_span > 0 else 1.0
        )

        # Convert raw period bounds to model-space period bounds
        lower_model = (
            float(lower_period) * period_scale if lower_period is not None else None
        )
        upper_model = (
            float(upper_period) * period_scale if upper_period is not None else None
        )

        # --- Detect model type and register the prior ---
        # Case 1: frequency-based model (spectral mixture) → mixture_means
        if "mixture_means" in self._model_pars:
            if prior_type == "lognormal":
                prior = LogNormalFrequencyPrior(
                    mu=mu, sigma=sigma,
                    lower_period=lower_model,
                    upper_period=upper_model,
                    period=True,
                )
            else:
                prior = NormalFrequencyPrior(
                    mean=mean, std=std,
                    lower_period=lower_model,
                    upper_period=upper_model,
                    period=True,
                )
            self._model_pars["mixture_means"]["module"].register_prior(
                "mixture_means_prior", prior, "mixture_means"
            )
            return

        # Case 2: period-based model (quasi-periodic) → period_length parameters
        period_keys = [
            k for k in self._model_pars
            if "period_length" in k and "raw_" not in k
        ]
        if period_keys:
            if prior_type == "lognormal":
                prior = LogNormalPeriodPrior(
                    mu=mu, sigma=sigma,
                    lower_bound=lower_model,
                    upper_bound=upper_model,
                )
            else:
                prior = NormalPeriodPrior(
                    mean=mean, std=std,
                    lower_bound=lower_model,
                    upper_bound=upper_model,
                )
            for key in period_keys:
                module = self._model_pars[key]["module"]
                module.register_prior("period_length_prior", prior, "period_length")
            return

        # Case 3: no periodicity parameter
        warnings.warn(
            "No period or frequency parameter found in the model. "
            "set_period_prior() has no effect for this model type.",
            stacklevel=2,
        )

    def get_period_prior(self):
        """Return the period or frequency prior currently registered on the model.

        Searches for a prior on the period or frequency parameter of the model
        (``mixture_means_prior`` for spectral-mixture models,
        ``period_length_prior`` for quasi-periodic models) and returns a
        dictionary of the priors found, keyed by the full parameter path.  A
        formatted summary is also printed to standard output.

        Returns
        -------
        dict
            A dictionary mapping prior names (str) to their prior objects.
            Returns an empty dict if no period/frequency prior has been
            registered or the model has no periodicity parameter.

        Raises
        ------
        RuntimeError
            If the model has not been set yet (call :meth:`set_model` first).

        See Also
        --------
        set_period_prior
        get_priors

        Examples
        --------
        ::

            lc.set_model("1D", num_mixtures=4)
            lc.set_period_prior(prior_set="LPV")
            prior_info = lc.get_period_prior()
        """
        if not hasattr(self, "_model_pars"):
            raise RuntimeError(
                "Model has not been set yet. Call set_model() before "
                "get_period_prior()."
            )
        period_priors = {}
        for name, _module, prior, _closure, _setting_closure in (
            self.model.named_priors()
        ):
            if "mixture_means_prior" in name or "period_length_prior" in name:
                period_priors[name] = prior

        print("Registered period/frequency priors:")
        if period_priors:
            for name, prior in period_priors.items():
                prior_type = type(prior).__name__
                line = f"  {name}: {prior_type}"
                params = []
                if hasattr(prior, "loc"):
                    params.append(f"loc={float(prior.loc):.4g}")
                if hasattr(prior, "scale"):
                    params.append(f"scale={float(prior.scale):.4g}")
                if hasattr(prior, "lower_period") and prior.lower_period is not None:
                    params.append(f"lower_period={float(prior.lower_period):.4g}")
                if hasattr(prior, "upper_period") and prior.upper_period is not None:
                    params.append(f"upper_period={float(prior.upper_period):.4g}")
                if hasattr(prior, "lower_bound") and prior.lower_bound is not None:
                    params.append(f"lower_bound={float(prior.lower_bound):.4g}")
                if hasattr(prior, "upper_bound") and prior.upper_bound is not None:
                    params.append(f"upper_bound={float(prior.upper_bound):.4g}")
                if params:
                    line += f"({', '.join(params)})"
                print(line)
        else:
            print("  (none)")
        return period_priors

    def _validate_2d_setup(self):
        """Validate that the 2D setup is correct

        This method checks that:
        - Data shapes are correct for 2D (xdata has shape [n_samples, 2])
        - Model has appropriate ard_num_dims parameter
        - Transforms can handle 2D data

        Raises
        ------
        ValueError
            If the 2D setup is invalid

        Warnings
        --------
        If there are potential issues with the setup
        """
        if self.ndim <= 1:
            return  # Only validate for 2D data

        # Check xdata shape
        if self._xdata_transformed.dim() != 2:
            raise ValueError(
                f"For 2D data, xdata must be 2-dimensional, "
                f"got {self._xdata_transformed.dim()}D"
            )

        if self._xdata_transformed.shape[1] != 2:
            raise ValueError(
                f"For 2D data, xdata must have 2 columns (time, wavelength), "
                f"got {self._xdata_transformed.shape[1]} columns"
            )

        # Check if model is set
        if not hasattr(self, "model"):
            warnings.warn(
                "Model not set yet. Cannot validate ard_num_dims. "
                "Ensure your model has ard_num_dims=2 for 2D data.",
                stacklevel=2,
            )
            return

        # Check if the model's kernel has ard_num_dims set correctly.
        # Separable models using ProductKernel + active_dims do not set
        # ard_num_dims (it stays None) — they are valid 2D models.
        # Only raise if ard_num_dims is explicitly set to a non-2 value.
        if hasattr(self.model, "covar_module"):
            covar = self.model.covar_module
            # For KISS-GP models, check the base_kernel
            if hasattr(covar, "base_kernel"):
                covar = covar.base_kernel

            if hasattr(covar, "ard_num_dims") and covar.ard_num_dims is not None:
                if covar.ard_num_dims != 2:
                    raise ValueError(
                        f"Model's ard_num_dims is {covar.ard_num_dims}, "
                        f"but data has {self.ndim} dimensions. "
                        "Use a 2D model (e.g., '2D', '2DLinear', '2DSKI', "
                        "'2DLinearSKI', '2DSeparable', '2DAchromatic', "
                        "'2DWavelengthDependent', '2DPowerLaw', '2DPowerLawSKI', "
                        "'2DDust', '2DDustSKI', '2DDustMean', '2DPowerLawMean')."
                    )

        # Check transform compatibility
        if self.xtransform is not None:
            if not hasattr(self.xtransform, "transform"):
                raise ValueError("xtransform must have a 'transform' method")

    def set_default_constraints(self, constraint_set=None, **kwargs):
        """Set the default constraints for the model and likelihood parameters

        The default constraints are as follows:
            - All parameters are constrained to be positive, except the mean of
            the GP, which is constrained to be in the range of the y-data (a correction
            will be needed if the data are censored!)
            - The noise is constrained to be less than the standard deviation of
            the y-data, and greater than either 1e-4 or 1/10 of the smallest
            uncertainty on the y-data, if uncertainties are given, or 1e-4
            times the standard deviation of the y data.
            - The mixture means greater than the frequency corresponding to
            the separation between the earliest and latest points in the data
            and less than the frequency corresponding to the separation between
            the two closest data points (should be updated to account for the
            window function and whatever we're really sensitive to)
            - The mixture scales and weights are left with their default
            constraints as defined in GPyTorch.

        Parameters
        ----------
        constraint_set : str or None, optional
            Name of a pre-defined source-type constraint set to apply on top
            of the default constraints.  When provided, the constraints
            defined for the named set (see
            :data:`pgmuvi.constraints.CONSTRAINT_SETS`) are merged into the
            default mixture-means constraint.  Currently supported values:

            ``"LPV"``
                Long-Period Variable stars.  Enforces a lower period limit of
                20 in the same time units as the input ``xdata`` (typically
                interpreted as 20 days for LPV light curves) so that the fit is
                not pulled toward unphysically short periods.  If ``xdata``
                is provided in different time units, this numerical limit
                applies in those units.

            Pass ``None`` (the default) to use only the data-driven defaults.
        **kwargs : dict, optional
            Any keyword arguments to be passed to the Constraint constructors.
        """
        if "noise" in self._model_pars:
            # only apply the noise constraint if we're using a learnable noise
            try:
                noise_min = np.minimum(1e-4, self._yerr_transformed.min() / 10)
            except AttributeError:
                noise_min = 1e-4 * self._ydata_transformed.std()
            noise_max = self._ydata_transformed.std()  # for a non-periodic source,
            # the noise should be less than
            # the standard deviation
            noise_constraint = Interval(noise_min, noise_max)
            self._model_pars["noise"]["module"].register_constraint(
                "raw_noise", noise_constraint
            )
        with contextlib.suppress(RuntimeError):
            mean_const_constraint = Interval(
                self._ydata_transformed.min(), self._ydata_transformed.max()
            )
            for key in self._model_pars:
                if "mean_module.constant" in key:
                    self._model_pars[key]["module"].register_constraint(
                        "raw_constant", mean_const_constraint
                    )
        # Apply frequency constraints only for spectral-mixture models that
        # have a mixture_means parameter.  Models using alternative kernels
        # (e.g. Matérn, quasi-periodic, separable) do not have this parameter
        # and must not be constrained here; they would raise KeyError otherwise.
        if "mixture_means" in self._model_pars:
            # this should correspond to the longest frequency entirely
            # contained in the dataset:
            if self.ndim > 1:
                # For 2D spectral-mixture models the mixture_means parameter
                # has shape (num_mixtures, 1, ard_num_dims), where ard_num_dims
                # equals the number of input dimensions (typically 2: time and
                # wavelength).  GPyTorch applies a *single scalar* constraint
                # element-wise to every entry in that tensor — it is not
                # possible to set different lower bounds for the time dimension
                # and the wavelength dimension simultaneously via the standard
                # register_constraint API.
                #
                # We therefore base the lower bound exclusively on the *time*
                # dimension (column 0 of xdata_transformed):
                #
                #   lower_bound = 1 / time_span
                #
                # This guarantees that the time-axis frequencies are always
                # >= 1/time_span, i.e., that the inferred periods are not
                # longer than the observational baseline — a physically
                # meaningful and stable lower bound.
                #
                # Note that this same lower bound is also applied to the
                # wavelength-axis frequency elements of mixture_means.  In
                # practice, wavelength frequencies represent the spatial
                # frequency of the SED variation across bands; constraining
                # them to be >= 1/time_span is conservative (frequencies
                # corresponding to structures much narrower in wavelength than
                # the observation baseline are still allowed), and is
                # preferable to using min(time_bound, wavelength_bound) which
                # would make the time lower bound arbitrarily permissive
                # whenever the wavelength span is large or the wavelength
                # range is zero.
                #
                # Users who need achromatic behaviour (wavelength-frequency
                # near zero) should use the separable model classes
                # (AchromaticGPModel, WavelengthDependentGPModel) which apply
                # kernels to each dimension independently, avoiding this
                # limitation entirely.
                time_span = (
                    self._xdata_transformed[:, 0].max()
                    - self._xdata_transformed[:, 0].min()
                )
                if float(time_span) <= 0.0:
                    raise ValueError(
                        "set_default_constraints requires a dataset whose "
                        "timestamps span a positive time range, but all "
                        "timestamps in the 2D input are identical "
                        "(time_span = 0). Ensure the training data covers "
                        "more than one distinct observation time."
                    )
                lower_frequency = 1.0 / time_span

                # Compute the Nyquist upper bound from the minimum positive
                # gap between consecutive sorted timestamps (O(N log N), O(N)
                # memory — avoids the O(N²) pairwise-difference matrix).
                t_sorted = self._xdata_transformed[:, 0].sort().values
                consecutive_diffs = (t_sorted[1:] - t_sorted[:-1])
                positive_diffs = consecutive_diffs[consecutive_diffs > 0]
                if positive_diffs.numel() > 0:
                    min_diff = positive_diffs.min()
                    max_freq = 1 / (2 * min_diff)  # Nyquist based on time sampling
                    mixture_means_constraint = Interval(lower_frequency, max_freq)
                else:
                    # time_span > 0 guarantees at least two distinct timestamps,
                    # so positive_diffs is always non-empty here.  This branch
                    # is unreachable in practice.
                    raise ValueError(  # pragma: no cover
                        "Unexpected degenerate timestamps: time_span > 0 but "
                        "no consecutive positive differences found."
                    )
            else:
                # 1D case: base the lower-frequency bound on the time span
                # (max - min) rather than the absolute maximum. This prevents
                # allowing periods longer than the observational baseline and
                # is consistent with the 2D logic above.
                time_span = (
                    self._xdata_transformed.max()
                    - self._xdata_transformed.min()
                )
                if float(time_span) <= 0.0:
                    raise ValueError(
                        "set_default_constraints requires a dataset whose "
                        "timestamps span a positive time range, but all "
                        "timestamps in the 1D input are identical "
                        "(time_span = 0). Ensure the training data covers "
                        "more than one distinct observation time."
                    )
                mixture_means_constraint = GreaterThan(1 / time_span)

            # Apply any constraint_set period bounds to the mixture_means
            # constraint
            if constraint_set is not None:
                cs = get_constraint_set(constraint_set)
                if "period" in cs:
                    period_bounds = cs["period"]
                    lower_val, lower_active = period_bounds["lower"]
                    upper_val, upper_active = period_bounds["upper"]

                    # Compute the scale factor to convert a period in original
                    # (untransformed) units to a frequency in transformed space.
                    # For any linear rescaling transform:
                    #   freq_transformed = freq_original * (x_orig_span /
                    #                                       x_trans_span)
                    if self.ndim > 1:
                        x_orig_span = float(
                            self._xdata_raw[:, 0].max()
                            - self._xdata_raw[:, 0].min()
                        )
                        x_trans_span = float(
                            self._xdata_transformed[:, 0].max()
                            - self._xdata_transformed[:, 0].min()
                        )
                    else:
                        x_orig_span = float(
                            self._xdata_raw.max() - self._xdata_raw.min()
                        )
                        x_trans_span = float(
                            self._xdata_transformed.max()
                            - self._xdata_transformed.min()
                        )
                    freq_scale = (
                        x_orig_span / x_trans_span if x_trans_span > 0 else 1.0
                    )

                    # Period lower limit → frequency upper limit
                    if lower_active and lower_val is not None:
                        max_freq_from_period = freq_scale / lower_val
                        cur_lower = float(mixture_means_constraint.lower_bound)
                        if max_freq_from_period > cur_lower:
                            if isinstance(mixture_means_constraint, GreaterThan):
                                mixture_means_constraint = Interval(
                                    cur_lower, max_freq_from_period
                                )
                            else:
                                # Already an Interval: tighten the upper bound
                                cur_upper = float(
                                    mixture_means_constraint.upper_bound
                                )
                                mixture_means_constraint = Interval(
                                    cur_lower,
                                    min(cur_upper, max_freq_from_period),
                                )

                    # Period upper limit → frequency lower limit
                    if upper_active and upper_val is not None:
                        min_freq_from_period = freq_scale / upper_val
                        cur_lower = float(mixture_means_constraint.lower_bound)
                        cur_upper = (
                            float(mixture_means_constraint.upper_bound)
                            if isinstance(mixture_means_constraint, Interval)
                            else float("inf")
                        )
                        new_lower = max(cur_lower, min_freq_from_period)
                        if new_lower < cur_upper:
                            if isinstance(mixture_means_constraint, GreaterThan):
                                mixture_means_constraint = GreaterThan(new_lower)
                            else:
                                mixture_means_constraint = Interval(
                                    new_lower, cur_upper
                                )

            self._model_pars["mixture_means"]["module"].register_constraint(
                "raw_mixture_means", mixture_means_constraint
            )

        # to-do - check if constraints on mixture scales are useful!
        self.__CONTRAINTS_SET = True

    def get_constraints(self):
        """Return the constraints currently registered on the model.

        Iterates over all constraints registered on the model (via GPyTorch's
        ``named_constraints``) and returns a dictionary mapping each constraint
        name to the corresponding constraint object.  A formatted summary is
        also printed to standard output.

        Returns
        -------
        dict
            A dictionary mapping constraint names (str) to their
            :class:`gpytorch.constraints.Constraint` objects. Returns an empty
            dict if no constraints have been registered.

        Raises
        ------
        RuntimeError
            If the model has not been set yet (call :meth:`set_model` first).

        See Also
        --------
        set_default_constraints

        Examples
        --------
        ::

            lc.set_model("1D", num_mixtures=4)
            lc.set_default_constraints()
            constraints = lc.get_constraints()
        """
        if not hasattr(self, "_model_pars"):
            raise RuntimeError(
                "Model has not been set yet. Call set_model() before "
                "get_constraints()."
            )
        constraints = {}
        for name, constraint in self.model.named_constraints():
            constraints[name] = constraint
        print("Registered constraints:")
        if constraints:
            for name, constraint in constraints.items():
                print(f"  {name}: {constraint}")
        else:
            print("  (none)")
        return constraints

    def set_hypers(self, hypers=None, debug=False, **kwargs):
        """Set the hyperparameters for the model and likelihood. This is a
        convenience function that calls the model.initialize() to set the
        hyperparameters. However, first it applies any transforms to the
        hyperparameters, so that the user can pass the hyperparameters in
        the original data space if they wish.

        Parameters
        ----------
        hypers : dict, optional
            A dictionary of the hyperparameters to use for the model and
            likelihood. The keys should be the names of the parameters, and the
            values should be Tensors containing the values of the parameters.
            If None, no hyperparameters will be set. If a hyperparameter is
            passed for a parameter that is not a model or likelihood
            parameter, it will be ignored.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the initialize.
        """

        if hypers is None:
            return
        pars_to_transform = {
            "x": ["mixture_means", "mixture_scales"],
            "y": ["noise", "mean_module"],
        }
        if debug:
            print("hypers before transform:")
            print(hypers)
        for key in hypers:
            # first, check if the parameter needs to be transformed:
            if any(p in key for p in pars_to_transform["x"]):
                # now apply the x transform
                # remember that the means and scales are in fourier space
                # so we need to transform them back to real space
                # before applying the transform
                # and then transform them back to fourier space
                # luckily, when the shift is removed from the transform,
                # the factors of 2pi cancel out for the scales
                # so we can just do 1/ for both means and scales
                if self.xtransform is not None:
                    if debug:
                        print(f"Applying x-transform to {key}")
                    # Check if the parameter is 2D (for multi-dimensional data)
                    if hypers[key].dim() == 2:
                        # For 2D hyperparameters (num_mixtures, ard_num_dims),
                        # the transform should be applied considering each
                        # dimension's range. Since transform was fit on
                        # (n_samples, 2) data, we need to handle this
                        # carefully
                        _num_mixtures, ard_num_dims = hypers[key].shape
                        transformed = torch.zeros_like(hypers[key])

                        # For each dimension of the 2D parameter
                        for dim in range(ard_num_dims):
                            # Get the range for this dimension from the
                            # fitted transformer
                            if (
                                hasattr(self.xtransform, "range")
                                and self.xtransform.range.shape[0] > dim
                            ):
                                # Apply dimension-specific scaling to the
                                # Fourier space parameters
                                # Formula: f_transformed = 1 / ((1 / f_raw)
                                # / range)
                                # This accounts for the data transformation
                                # applied to each dimension
                                # The 1/x transformations handle the Fourier
                                # space representation
                                dim_values = hypers[key][:, dim]
                                # Transform back to real space, apply
                                # scaling, then back to Fourier
                                transformed[:, dim] = 1 / (
                                    (1 / dim_values) / self.xtransform.range[0, dim]
                                )
                            else:
                                # Fallback: just copy the values
                                transformed[:, dim] = hypers[key][:, dim]
                        hypers[key] = transformed
                    else:
                        # 1D case - original behavior
                        hypers[key] = 1 / self.xtransform.transform(
                            1 / hypers[key], shift=False
                        )
            elif any(p in key for p in pars_to_transform["y"]):
                # now apply the y transform
                # the mean function and noise are not defined in fourier
                # space, so we can just apply the transform directly
                if self.ytransform is not None:
                    if debug:
                        print(f"Applying y-transform to {key}")
                    hypers[key] = self.ytransform.transform(hypers[key])
        if debug:
            print("hypers after transform:")
            print(hypers)
        self.model.initialize(**hypers, **kwargs)

    def init_hypers_from_LombScargle(self, **kwargs):
        pass

    def _set_hypers_raw(self, hypers=None, **kwargs):
        pass

    def cpu(self):
        self.device = torch.device("cpu")
        super().cpu()
        for _ in dict_walk_generator(self._model_pars):
            with contextlib.suppress(AttributeError):
                _.cpu()

    def cuda(self, device=0):
        # First we should check that CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot call cuda() if CUDA is not available")
        # next we should log that we're using CUDA
        self.device = torch.device(f"cuda:{device}")

        # now we need to make sure that the usual nn.Module.cuda() method
        # is called, so that all of the modules and buffers are moved to the GPU
        super().cuda(device=device)

        # but we've created a few extra things that need to be tracked.
        # We have to make sure any tensors in those are also moved to the same device
        for _ in dict_walk_generator(self._model_pars):
            with contextlib.suppress(AttributeError):
                _.cuda(device=device)

        # for key in self._model_pars:
        #     with contextlib.suppress(AttributeError):
        #         self._model_pars[key]['param'] = self._model_pars[key]['param'].cuda(device=device) # noqa: E501
        # try:
        #     self.model.cuda()
        #     self.likelihood.cuda()
        # except AttributeError as e:
        #     errmsg = "You must first set the model and likelihood"
        #     _reraise_with_note(e, errmsg)

    def _train(self):
        try:
            self.model.train()
            self.likelihood.train()
        except AttributeError as e:
            errmsg = "You must first set the model and likelihood"
            _reraise_with_note(e, errmsg)

    def _eval(self):
        try:
            self.model.eval()
            self.likelihood.eval()
        except AttributeError as e:
            errmsg = "You must first set the model and likelihood"
            _reraise_with_note(e, errmsg)

    def fit_LS(
        self,
        freq_only: bool = False,
        num_peaks: int = 1,
        single_threshold: float = 0.05,
        Nyquist_factor: int = 5,
        fap_method: str | None = None,
        use_best_band_init: bool = False,
        return_full: bool = False,
        **kwargs,
    ) -> tuple:
        """
        Compute the (multiband) Lomb-Scargle periodogram.
        Periods returned for the num_peaks highest peaks in the periodogram.
        For a 1D lightcurve, the false-alarm probability is used
            to estimate the significance of the periods, which are also
            returned. These can be used to filter out insignificant periods.
        For multi-band lightcurves (2D data), LombScargleMultiband is used
            to compute periods across all bands simultaneously.

        The method can also be used to return the entire grid of frequencies,
        which can be used by other methods such as compute_psd and plot_psd.

        Parameters:
        ----------------
        - freq_only: bool, optional, default=False
            If True, only the frequency grid will be returned.
            This can be useful for methods such as compute_psd and plot_psd.
        - num_peaks: int, optional, default=1
            The number of peaks to extract from the Lomb-Scargle periodogram.
            If fewer peaks are found, only the available peaks will be returned.
        - single_threshold: float, optional, default=0.05
            The false alarm probability threshold for a single peak to be
            considered significant.
        - Nyquist_factor: int, optional, default=5
            The factor by which to multiply the Nyquist frequency to
            determine the maximum frequency to search for in the
            Lomb-Scargle periodogram.
            This will be approximately the number of points sampling
            the maximum in the resulting periodogram.
        - fap_method: str or None, optional, default=None
            Method used to compute the false-alarm probability (FAP) of the
            *maximum* periodogram peak (the global significance test).
            For 1D lightcurves the default is ``'davies'`` (fast analytical
            upper bound; equivalent to ``'baluev'`` for practical purposes
            but significantly faster). Other valid astropy options are
            ``'baluev'`` and ``'bootstrap'``. Note: ``'single'`` is a
            valid astropy option that computes the FAP for a single
            pre-specified frequency and is not appropriate for ``fap_max``
            (a warning is issued and ``'baluev'`` is used instead); it is
            however used internally as the per-frequency p-value when
            applying the Benjamini-Hochberg correction.
            For multi-band lightcurves the default is ``'analytical'`` (fast
            Baluev-style approximation).  Slower but more accurate options
            are ``'bootstrap'``, ``'phase_scramble'``, and ``'calibrated'``
            (see
            :class:`~pgmuvi.multiband_ls_significance.MultibandLSWithSignificance`).
        - use_best_band_init: bool, optional, default=False
            If True and the lightcurve is multiband (ndim > 1), the
            Lomb-Scargle frequency grid is derived from the band with the
            most observations rather than from the full multiband dataset.
            This yields a finer frequency resolution focused on the most
            informative band, which can speed up and improve the
            periodogram search when sampling is highly heterogeneous
            across bands.  Has no effect for 1D lightcurves.
        - return_full: bool, optional, default=False
            If True and ``freq_only=False``, also return the complete
            frequency grid and power spectrum alongside the peak frequencies
            and significance mask (see return values below).  Ignored when
            ``freq_only=True``.  The periodogram itself is not recomputed,
            but returning the full grid may still allocate and/or copy the
            frequency and power tensors before returning them.
        - kwargs: dict, optional
            Additional keyword arguments to be passed to the
            LombScargle(Multiband) constructor.

        Returns:
        ----------------
        The return value depends on the combination of ``freq_only`` and
        ``return_full``:

        * ``freq_only=True`` (``return_full`` is ignored):
          ``(freq_grid, power_grid)``

          - freq_grid: torch.Tensor of floats — the full frequency grid.
          - power_grid: torch.Tensor of floats — periodogram power at each
            frequency.

        * ``freq_only=False, return_full=False`` (default):
          ``(peak_freqs, significance_mask)``

          - peak_freqs: torch.Tensor of floats — frequencies of the
            ``num_peaks`` highest periodogram peaks.
          - significance_mask: torch.Tensor of bool — True for peaks that
            are statistically significant after Benjamini-Hochberg FDR
            correction.

        * ``freq_only=False, return_full=True``:
          ``(peak_freqs, significance_mask, freq_grid, power_grid)``

          - peak_freqs: torch.Tensor of floats — as above.
          - significance_mask: torch.Tensor of bool — as above.
          - freq_grid: torch.Tensor of floats — the full frequency grid
            (already computed internally; returned at no extra cost).
          - power_grid: torch.Tensor of floats — periodogram power at each
            frequency (already computed internally).
        """
        from astropy.timeseries import LombScargle
        from scipy.signal import find_peaks
        from .multiband_ls_significance import MultibandLSWithSignificance

        def fdr_bh(fap_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
            """
            Benjamini-Hochberg procedure to control the False Discovery Rate.

            The Benjamini-Hochberg (BH) procedure is a method for controlling the
            False Discovery Rate (FDR) when performing multiple hypothesis tests.
            It works by:
            1. Sorting the p-values (FAPs) in ascending order
            2. Finding the largest i such that p(i) <= (i/N) * alpha
            3. Rejecting all hypotheses with p-values <= p(i)

            This is less conservative than Bonferroni correction while still
            controlling the expected proportion of false discoveries.

            See https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.fdrcorrection.html
                for the statsmodels implementation

            Parameters:
            ----------------
            - fap_values: Array of false alarm probabilities (FAP) for peaks.
            - alpha: Desired FDR threshold (e.g., 0.05 for 5% FDR).

            Returns:
            ----------------
            - result: array(bool), True for statistically significant peaks.
            """
            # Sort FAP values in ascending order and get their original indices
            sorted_indices = np.argsort(fap_values)
            sorted_fap = fap_values[sorted_indices]
            N = len(fap_values)
            # Find the largest i such that fap(i) <= (i / N) * alpha
            threshold = np.arange(1, N + 1) / N * alpha
            significant = sorted_fap <= threshold
            # If there are significant results, keep the largest index
            if np.any(significant):
                max_signif_index = np.where(significant)[0].max()
                significant_indices = sorted_indices[: max_signif_index + 1]
                result = np.zeros(N, dtype=bool)
                result[significant_indices] = True
            else:
                result = np.zeros(N, dtype=bool)
            return result

        # Build working arrays from the stored (already finite) data.
        _has_yerr = (
            hasattr(self, "_yerr_transformed")
            and self._yerr_transformed is not None
        )
        _xdata = self.xdata
        _ydata = self.ydata
        _yerr = self.yerr if _has_yerr else None

        if self.ndim > 1:
            # Multi-band case: _xdata[:, 0] is time, _xdata[:, 1] is band/wavelength
            t = _xdata[:, 0]
            bands = _xdata[:, 1]
            y = _ydata

            # Default FAP method for multiband: analytical (fast)
            _fap_method = fap_method if fap_method is not None else 'analytical'

            if _yerr is not None:
                yerr = _yerr
                LS = MultibandLSWithSignificance(t, y, bands, dy=yerr, **kwargs)
            else:
                LS = MultibandLSWithSignificance(t, y, bands, **kwargs)

            if use_best_band_init:
                # Use the most-sampled band's 1D autofrequency as the grid
                # for the multiband LS.  The best-sampled band has finer
                # temporal resolution (more data points), yielding a denser
                # frequency grid that improves period recovery when sampling
                # is highly heterogeneous across bands.
                _unique_bands, _band_counts = torch.unique(
                    bands, return_counts=True
                )
                _best_val = _unique_bands[_band_counts.argmax()]
                _best_mask = bands == _best_val
                _t_best = t[_best_mask].detach().cpu().numpy()
                _y_best = y[_best_mask].detach().cpu().numpy()
                if _has_yerr:
                    _dy_best = yerr[_best_mask].detach().cpu().numpy()
                    _ls_1d_best = LombScargle(_t_best, _y_best, _dy_best)
                else:
                    _ls_1d_best = LombScargle(_t_best, _y_best)
                freq = _ls_1d_best.autofrequency(nyquist_factor=Nyquist_factor)
                power = _ls_1d_best.power(freq)
            else:
                freq = LS.autofrequency(nyquist_factor=Nyquist_factor)
                power = LS.power(freq)

            if freq_only:
                return (
                    torch.as_tensor(
                        freq, dtype=self.xdata.dtype, device=self.xdata.device
                    ),
                    torch.as_tensor(
                        power, dtype=self.xdata.dtype, device=self.xdata.device
                    ),
                )

            # Build full-grid tensors only when they are requested to avoid
            # unnecessary allocation/copy on the default path.
            if return_full:
                _freq_t = torch.as_tensor(
                    freq, dtype=self.xdata.dtype, device=self.xdata.device
                )
                _power_t = torch.as_tensor(
                    power, dtype=self.xdata.dtype, device=self.xdata.device
                )

            # Find peaks in the multiband periodogram
            peaks, _ = find_peaks(power, distance=Nyquist_factor)
            peaks = peaks[np.argsort(power[peaks])][::-1]

            # Handle case when no peaks found
            if len(peaks) == 0:
                _pf = torch.as_tensor(
                    [], dtype=self.xdata.dtype, device=self.xdata.device
                )
                _sm = torch.as_tensor(
                    [], dtype=torch.bool, device=self.xdata.device
                )
                # return_full=True exposes already-computed LS intermediates
                if return_full:
                    return (_pf, _sm, _freq_t, _power_t)
                return (_pf, _sm)

            # Compute FAP for multiband periodogram
            fap_max = LS.false_alarm_probability(power.max(),
                                                 method=_fap_method,
                                                 freq_grid=freq)
            n_return = min(num_peaks, len(peaks))

            if fap_max > single_threshold:
                # Highest peak is not significant, mark all as insignificant
                _pf = torch.as_tensor(
                    freq[peaks[:n_return]],
                    dtype=self.xdata.dtype,
                    device=self.xdata.device,
                )
                _sm = torch.as_tensor(
                    np.array([False] * n_return),
                    dtype=torch.bool,
                    device=self.xdata.device,
                )
                # return_full=True exposes already-computed LS intermediates
                if return_full:
                    return (_pf, _sm, _freq_t, _power_t)
                return (_pf, _sm)

            # Calculate FAP for each peak independently
            fap_single = LS.false_alarm_probability(power[peaks],
                                                    method=_fap_method,
                                                    freq_grid=freq)

            # Apply the FDR (Benjamini-Hochberg) correction
            significant_mask = fdr_bh(fap_single, alpha=single_threshold)
            significant_mask[0] = True  # since fap_max <= single_threshold

            _pf = torch.as_tensor(
                freq[peaks[:n_return]],
                dtype=self.xdata.dtype,
                device=self.xdata.device,
            )
            _sm = torch.as_tensor(
                significant_mask[:n_return],
                dtype=torch.bool,
                device=self.xdata.device,
            )
            # return_full=True exposes already-computed LS intermediates
            if return_full:
                return (_pf, _sm, _freq_t, _power_t)
            return (_pf, _sm)
        else:
            t, y = _xdata, _ydata

            # Default FAP method for single-band: 'davies' (fast analytical
            # upper bound; same accuracy as 'baluev' for typical use cases
            # but much faster). 'baluev' is another good analytical choice.
            _fap_method = fap_method if fap_method is not None else 'davies'

            if _yerr is not None:
                yerr = _yerr
                LS = LombScargle(t, y, yerr)
            else:
                LS = LombScargle(t, y)
            freq = LS.autofrequency(nyquist_factor=Nyquist_factor)
            # assume_regular_frequency=True: autofrequency() always produces
            # a regular grid, so skip the regularity check for a minor speedup
            power = LS.power(freq, assume_regular_frequency=True)
            if freq_only:
                return (
                    torch.as_tensor(
                        freq, dtype=self.xdata.dtype, device=self.xdata.device
                    ),
                    torch.as_tensor(
                        power, dtype=self.xdata.dtype, device=self.xdata.device
                    ),
                )

            # Build full-grid tensors only when they are requested to avoid
            # unnecessary allocation/copy on the default path.
            if return_full:
                _freq_t = torch.as_tensor(
                    freq, dtype=self.xdata.dtype, device=self.xdata.device
                )
                _power_t = torch.as_tensor(
                    power, dtype=self.xdata.dtype, device=self.xdata.device
                )

            # distance set to Nyquist_factor for LS frequency grid computation
            peaks, _ = find_peaks(power, distance=Nyquist_factor)
            # sort by decreasing power
            peaks = peaks[np.argsort(power[peaks])][::-1]

            # Handle case when no peaks or fewer peaks than requested
            if len(peaks) == 0:
                # No peaks found, return empty tensors
                _pf = torch.as_tensor(
                    [], dtype=self.xdata.dtype, device=self.xdata.device
                )
                _sm = torch.as_tensor(
                    [], dtype=torch.bool, device=self.xdata.device
                )
                # return_full=True exposes already-computed LS intermediates
                if return_full:
                    return (_pf, _sm, _freq_t, _power_t)
                return (_pf, _sm)

            # Calculate the false alarm probability for the highest peak.
            # 'single' is not appropriate for fap_max (it computes the FAP
            # for a single pre-specified frequency, not the global maximum).
            _fap_method_max = _fap_method
            if _fap_method_max == 'single':
                warnings.warn(
                    "fap_method='single' is not appropriate for the false alarm "
                    "probability of the maximum peak (it computes the FAP for a "
                    "single pre-specified frequency, not the global maximum). "
                    "Using method='baluev' for fap_max instead.",
                    UserWarning,
                    stacklevel=2,
                )
                _fap_method_max = 'baluev'
            fap_max = LS.false_alarm_probability(power.max(),
                                                 method=_fap_method_max)
            n_return = min(num_peaks, len(peaks))

            if fap_max > single_threshold:
                _pf = torch.as_tensor(
                    freq[peaks[:n_return]],
                    dtype=self.xdata.dtype,
                    device=self.xdata.device,
                )
                _sm = torch.as_tensor(
                    np.array([False] * n_return),
                    dtype=torch.bool,
                    device=self.xdata.device,
                )
                # return_full=True exposes already-computed LS intermediates
                if return_full:
                    return (_pf, _sm, _freq_t, _power_t)
                return (_pf, _sm)
            # Per-peak FAP for the Benjamini-Hochberg correction.
            # We use method='single' here: it gives the single-frequency FAP
            # (probability that one pre-specified frequency shows at least
            # this power by chance), which is the correct uncorrected p-value
            # to supply to BH.  The 'davies'/'baluev' methods already account
            # for multiple-frequency comparisons and would be too conservative.
            fap_single = LS.false_alarm_probability(power[peaks],
                                                    method='single')
            # Apply the FDR (Benjamini-Hochberg) correction
            significant_mask = fdr_bh(fap_single, alpha=single_threshold)
            significant_mask[0] = True  # since fap_max <= single_threshold
            _pf = torch.as_tensor(
                freq[peaks[:n_return]],
                dtype=self.xdata.dtype,
                device=self.xdata.device,
            )
            _sm = torch.as_tensor(
                significant_mask[:n_return],
                dtype=torch.bool,
                device=self.xdata.device,
            )
            # return_full=True exposes already-computed LS intermediates
            if return_full:
                return (_pf, _sm, _freq_t, _power_t)
            return (_pf, _sm)

    def compute_sampling_metrics(self) -> dict:
        """
        Compute temporal sampling quality metrics.

        Returns
        -------
        dict
            Comprehensive sampling metrics (see
            preprocess.quality.compute_sampling_metrics)

        Examples
        --------
        >>> lc = Lightcurve(t, y, yerr)
        >>> metrics = lc.compute_sampling_metrics()
        >>> print(f"Nyquist period: {metrics['nyquist_period']:.2f}")
        """
        from pgmuvi.preprocess.quality import compute_sampling_metrics

        t = self._xdata_raw.detach().cpu().numpy()
        if t.ndim > 1:
            t = t[:, 0]
        y = (
            self._ydata_raw.detach().cpu().numpy()
            if hasattr(self, "_ydata_raw")
            else None
        )
        yerr = (
            self._yerr_raw.detach().cpu().numpy()
            if hasattr(self, "_yerr_raw")
            else None
        )
        return compute_sampling_metrics(t, y, yerr)

    def assess_sampling_quality(self, verbose: bool = True, **kwargs) -> tuple:
        """
        Assess whether lightcurve sampling is adequate for GP fitting.

        Parameters
        ----------
        verbose : bool, default=True
            Print detailed assessment report
        **kwargs : dict
            Quality gate thresholds (see
            preprocess.quality.assess_sampling_quality):

            - min_points: int (default 6)
            - max_gap_fraction: float (default 0.3)
            - min_baseline_factor: float (default 3.0)
            - min_snr: float (default 3.0)
            - min_fraction_good_snr: float (default 0.5)

        Returns
        -------
        passes : bool
            True if all quality gates pass
        diagnostics : dict
            Diagnostic information including gates, metrics, warnings,
            and recommendation

        Examples
        --------
        >>> lc = Lightcurve(t, y, yerr)
        >>> passes, diag = lc.assess_sampling_quality(verbose=True)
        >>> if diag['recommendation'] == 'PROCEED':
        ...     lc.fit(...)
        """
        from pgmuvi.preprocess.quality import assess_sampling_quality

        t = self._xdata_raw.detach().cpu().numpy()
        if t.ndim > 1:
            t = t[:, 0]
        y = (
            self._ydata_raw.detach().cpu().numpy()
            if hasattr(self, "_ydata_raw")
            else None
        )
        yerr = (
            self._yerr_raw.detach().cpu().numpy()
            if hasattr(self, "_yerr_raw")
            else None
        )
        passes, diagnostics = assess_sampling_quality(
            t, y, yerr, verbose=verbose, **kwargs
        )
        return passes, diagnostics

    def compute_sampling_metrics_per_band(self) -> dict:
        """
        Compute sampling metrics independently for each wavelength band.

        Only applicable for 2D (multiband) lightcurves.

        Returns
        -------
        dict
        {
                wavelength1: metrics_dict,
                wavelength2: metrics_dict,
                ...
                'summary': {
                    'n_bands': int,
                    'min_points_across_bands': int,
                    'max_gap_fraction_worst_band': float,
                    'median_nyquist_period': float
                }
            }

        Raises
        ------
        ValueError
            If lightcurve is not 2D (multiband).
        """
        from pgmuvi.preprocess.quality import compute_sampling_metrics

        if self.ndim <= 1:
            raise ValueError(
                "compute_sampling_metrics_per_band() requires 2D (multiband) data. "
                "Use compute_sampling_metrics() for 1D data."
            )

        xdata = self._xdata_raw.detach().cpu().numpy()
        ydata = self._ydata_raw.detach().cpu().numpy()
        yerr = (
            self._yerr_raw.detach().cpu().numpy()
            if hasattr(self, "_yerr_raw")
            else None
        )

        wavelengths = np.unique(xdata[:, 1])
        results = {}

        min_points_list = []
        max_gaps_list = []
        nyquist_list = []

        for wl in wavelengths:
            mask = xdata[:, 1] == wl
            t = xdata[mask, 0]
            y = ydata[mask]
            ye = yerr[mask] if yerr is not None else None

            metrics = compute_sampling_metrics(t, y, ye)
            results[float(wl)] = metrics

            if "n_points" in metrics:
                min_points_list.append(metrics["n_points"])
            if "max_gap_fraction" in metrics:
                max_gaps_list.append(metrics["max_gap_fraction"])
            if "nyquist_period" in metrics:
                nyquist_list.append(metrics["nyquist_period"])

        results["summary"] = {
            "n_bands": len(wavelengths),
            "min_points_across_bands": min(min_points_list) if min_points_list else 0,
            "max_gap_fraction_worst_band": (
                max(max_gaps_list) if max_gaps_list else np.inf
            ),
            "median_nyquist_period": (
                float(np.median(nyquist_list)) if nyquist_list else np.inf
            ),
        }

        return results

    def assess_sampling_quality_per_band(
        self, verbose: bool = True, **kwargs
    ) -> dict:
        """
        Assess sampling quality independently for each wavelength band.

        Only applicable for 2D (multiband) lightcurves.

        Parameters
        ----------
        verbose : bool, default=True
            Print assessment for each band
        **kwargs : dict
            Quality gate thresholds

        Returns
        -------
        dict
            {
                wavelength1: diagnostics_dict,
                wavelength2: diagnostics_dict,
                ...
                'summary': {
                    'n_bands': int,
                    'n_passing': int,
                    'passing_wavelengths': list[float],
                    'failing_wavelengths': list[float]
                    }
            }

        Raises
        ------
        ValueError
            If lightcurve is not 2D (multiband).
        """
        from pgmuvi.preprocess.quality import assess_sampling_quality

        if self.ndim <= 1:
            raise ValueError(
                "assess_sampling_quality_per_band() requires 2D (multiband) data. "
                "Use assess_sampling_quality() for 1D data."
            )

        xdata = self._xdata_raw.detach().cpu().numpy()
        ydata = self._ydata_raw.detach().cpu().numpy()
        yerr = (
            self._yerr_raw.detach().cpu().numpy()
            if hasattr(self, "_yerr_raw")
            else None
        )

        wavelengths = np.unique(xdata[:, 1])
        results = {}
        passing_bands = []
        failing_bands = []

        for wl in wavelengths:
            mask = xdata[:, 1] == wl
            t = xdata[mask, 0]
            y = ydata[mask]
            ye = yerr[mask] if yerr is not None else None

            if verbose:
                print(f"\n{'=' * 70}")
                print(f"BAND: \u03bb = {wl}")
                print(f"{'=' * 70}")

            passes, diag = assess_sampling_quality(t, y, ye, verbose=verbose, **kwargs)
            results[float(wl)] = diag

            if passes:
                passing_bands.append(float(wl))
            else:
                failing_bands.append(float(wl))

        results["summary"] = {
            "n_bands": len(wavelengths),
            "n_passing": len(passing_bands),
            "passing_wavelengths": passing_bands,
            "failing_wavelengths": failing_bands,
        }

        return results

    def filter_well_sampled_bands(self, **kwargs):
        """
        Create new Lightcurve with only well-sampled bands retained.

        Only applicable for 2D (multiband) lightcurves.

        Parameters
        ----------
        **kwargs : dict
            Quality gate thresholds

        Returns
        -------
        Lightcurve
            New instance containing only wavelengths that pass sampling checks

        Raises
        ------
        ValueError
            If lightcurve is not 2D (multiband) or no bands pass sampling
            checks.
        """
        if self.ndim <= 1:
            raise ValueError(
                "filter_well_sampled_bands() requires 2D (multiband) data."
            )

        results = self.assess_sampling_quality_per_band(verbose=False, **kwargs)

        if results["summary"]["n_passing"] == 0:
            raise ValueError(
                "No bands passed sampling quality checks. "
                "Consider relaxing criteria or acquiring more data."
            )

        keep_wl = results["summary"]["passing_wavelengths"]
        xdata = self._xdata_raw
        keep_mask = torch.isin(
            xdata[:, 1],
            torch.tensor(keep_wl, dtype=xdata.dtype, device=xdata.device),
        )

        return Lightcurve(
            xdata[keep_mask].clone(),
            self._ydata_raw[keep_mask].clone(),
            self._yerr_raw[keep_mask].clone() if hasattr(self, "_yerr_raw") else None,
        )

    def _get_best_sampled_band_lc(self) -> "Lightcurve":
        """Return a 1D Lightcurve for the band with the most observations.

        For 1D lightcurves, returns ``self`` unchanged.

        Returns
        -------
        Lightcurve
            1D Lightcurve (xdata is the time column only) built from the
            raw (untransformed) data of the most-sampled wavelength band.
            If multiple bands share the same (maximum) number of observations,
            the band with the smallest band value (as returned by
            ``torch.unique``) is returned.
        """
        if self.ndim <= 1:
            return self

        bands = self._xdata_raw[:, 1]
        unique_bands, band_counts = torch.unique(bands, return_counts=True)
        best_band_val = unique_bands[band_counts.argmax()]
        mask = bands == best_band_val

        t = self._xdata_raw[mask, 0]
        y = self._ydata_raw[mask]
        yerr = (
            self._yerr_raw[mask]
            if hasattr(self, "_yerr_raw") and self._yerr_raw is not None
            else None
        )
        return Lightcurve(t, y, yerr=yerr)

    def _get_variability_arrays(self):
        """Return (y, yerr) as float64 NumPy arrays, safe for CPU and GPU tensors."""
        y = self._ydata_raw.detach().cpu().numpy()
        if hasattr(self, "_yerr_raw") and self._yerr_raw is not None:
            yerr = self._yerr_raw.detach().cpu().numpy()
        else:
            yerr = np.ones_like(y)
        return y, yerr

    def check_variability(self, **kwargs) -> dict:
        """
        Check if lightcurve shows significant variability.

        Only applicable for 1-D lightcurves. For multiband data use
        :meth:`check_variability_per_band`.

        Parameters
        ----------
        **kwargs : dict
            Arguments passed to is_variable():
            - alpha: float (default 0.01)
            - fvar_min: float (default 0.05)
            - stetson_k_min: float (default 0.95)
            - verbose: bool (default False)

        Returns
        -------
          Variability diagnostics from is_variable()
          If the lightcurve is multiband (ndim > 1). Use
          check_variability_per_band() instead.

        Examples
        --------
        >>> lc = Lightcurve(t, y, yerr)
        >>> diag = lc.check_variability(verbose=True)
        >>> print(f"Variable: {diag['decision']}")
        """
        if self.ndim > 1:
            raise ValueError(
                "check_variability() is for 1-D lightcurves. "
                "For multiband data use check_variability_per_band()."
            )
        from pgmuvi.preprocess.variability import is_variable

        y, yerr = self._get_variability_arrays()
        _is_var, diagnostics = is_variable(y, yerr, **kwargs)
        return diagnostics

    def check_variability_per_band(self, **kwargs) -> dict:
        """
        Check variability independently for each wavelength band.

        Only applicable for multiband (2D) lightcurves where
        ``xdata[:, 1]`` encodes the band/wavelength.

        Parameters
        ----------
        **kwargs : dict
            Arguments passed to is_variable()
                    'n_variable': int,
                    'variable_wavelengths': list[float]

        Returns
        -------
            If the lightcurve is not 2-D multiband data (ndim != 2 columns
            with time in column 0 and wavelength in column 1).

        Examples
        --------
        >>> lc2d = Lightcurve(xdata_2d, y, yerr)
        >>> results = lc2d.check_variability_per_band(verbose=True)
        >>> n_var = results['summary']['n_variable']
        >>> n_bands = results['summary']['n_bands']
        >>> print(f"{n_var}/{n_bands} bands variable")
        """
        if self._xdata_raw.dim() != 2 or self._xdata_raw.shape[1] < 2:
            raise ValueError(
                "check_variability_per_band() requires 2-D multiband data "
                "with shape (N, 2) where column 0 is time and column 1 is "
                "the band/wavelength. Got shape "
                f"{tuple(self._xdata_raw.shape)}."
            )
        from pgmuvi.preprocess.variability import is_variable

        # Convert to NumPy once; safe for both CPU and CUDA tensors
        xdata_band = self._xdata_raw[:, 1].detach().cpu().numpy()
        ydata = self._ydata_raw.detach().cpu().numpy()
        if hasattr(self, "_yerr_raw") and self._yerr_raw is not None:
            yerr_data = self._yerr_raw.detach().cpu().numpy()
        else:
            yerr_data = None

        wavelengths = np.unique(xdata_band)
        results = {}
        variable_bands = []

        for wl in wavelengths:
            mask = xdata_band == wl
            y = ydata[mask]
            yerr = yerr_data[mask] if yerr_data is not None else np.ones_like(y)

            is_var, diag = is_variable(y, yerr, **kwargs)
            results[float(wl)] = diag

            if is_var:
                variable_bands.append(float(wl))

        results["summary"] = {
            "n_bands": len(wavelengths),
            "n_variable": len(variable_bands),
            "variable_wavelengths": variable_bands,
        }
        return results



    def filter_variable_bands(self, **kwargs):
        """
        Create new Lightcurve with only variable bands retained.

        Only applicable for multiband (2D) lightcurves where
        ``xdata[:, 1]`` encodes the band/wavelength.


         Parameters
         ----------
            Arguments passed to is_variable()

        Returns
        -------
        lightcurve : Lightcurve
            New instance containing only wavelengths that pass variability tests
        None
            If no bands pass variability tests

        Examples
        --------
        >>> lc2d = Lightcurve(xdata_2d, y, yerr)
        >>> lc_var = lc2d.filter_variable_bands()
        >>> # Check how many bands were retained via the per-band summary
        >>> results = lc2d.check_variability_per_band()
        >>> print(f"Retained {results['summary']['n_variable']} variable bands")
        """
        results = self.check_variability_per_band(**kwargs)

        if results["summary"]["n_variable"] == 0:
            raise ValueError(
                "No bands passed variability tests. "
                "Consider relaxing criteria (alpha, fvar_min, stetson_k_min)"
            )

        keep_wl = results["summary"]["variable_wavelengths"]
        wl_array = self._xdata_raw[:, 1].detach().cpu().numpy()
        keep_mask = np.isin(wl_array, keep_wl)
        keep_tensor = torch.as_tensor(
            keep_mask,
            dtype=torch.bool,
            device=self._xdata_raw.device,
        )

        new_x = self._xdata_raw[keep_tensor].clone()
        new_y = self._ydata_raw[keep_tensor].clone()

        if hasattr(self, "_yerr_raw") and self._yerr_raw is not None:
            new_yerr = self._yerr_raw[keep_tensor].clone()
        else:
            new_yerr = None

        return Lightcurve(new_x, new_y, yerr=new_yerr)

    def auto_select_model(self, verbose=True):
        """Automatically select the best model type based on data characteristics.

        Analyses the data to recommend an appropriate GP model. For 1D data,
        the Lomb-Scargle periodogram is computed and the peak power used to
        decide between a quasi-periodic, periodic+stochastic, or Matérn model.
        For 2D multiwavelength data, per-band periodograms are compared to
        determine whether the variability is achromatic or wavelength-dependent.

        Parameters
        ----------
        verbose : bool, optional
            If True, print a summary of the recommendation, by default True.

        Returns
        -------
        model_str : str
            Recommended model string suitable for passing directly to
            ``fit()`` or ``set_model()``.
        diagnostics : dict
            Dictionary containing:
            - ``'model'`` — same as ``model_str``.
            - ``'reason'`` — human-readable explanation.
            - Additional data-dependent keys (e.g. ``'max_ls_power'``).

        Examples
        --------
        >>> from pgmuvi.lightcurve import Lightcurve
        >>> import torch
        >>> import numpy as np
        >>> t = torch.linspace(0, 20, 100)
        >>> y = torch.sin(2 * np.pi * t / 5)
        >>> lc = Lightcurve(t, y)
        >>> model_str, diag = lc.auto_select_model()
        """
        from .initialization import initialize_separable_from_data

        diagnostics = {}

        if self.ndim == 1:
            # 1D data — use Lomb-Scargle to assess periodicity strength
            _freq, power = self.fit_LS(freq_only=True)
            max_power = float(power.max()) if len(power) > 0 else 0.0
            diagnostics["max_ls_power"] = max_power

            if max_power > 0.5:
                model_str = "1DQuasiPeriodic"
                diagnostics["reason"] = (
                    f"Strong periodic signal detected (LS power={max_power:.2f}); "
                    "quasi-periodic kernel recommended."
                )
            elif max_power > 0.2:
                model_str = "1DPeriodicStochastic"
                diagnostics["reason"] = (
                    f"Moderate periodicity with stochastic component "
                    f"(LS power={max_power:.2f}); "
                    "periodic+stochastic kernel recommended."
                )
            else:
                model_str = "1DMatern"
                diagnostics["reason"] = (
                    f"No strong periodicity detected (LS power={max_power:.2f}); "
                    "Matérn kernel recommended for stochastic variability."
                )
        else:
            # 2D data — check whether periods are consistent across wavelengths
            init_params = initialize_separable_from_data(
                self._xdata_raw,
                self._ydata_raw,
            )
            diagnostics["init_params"] = init_params

            if init_params.get("is_achromatic", True):
                model_str = "2DAchromatic"
                diagnostics["reason"] = (
                    "Periods consistent across wavelengths (achromatic variability); "
                    "achromatic separable kernel recommended."
                )
            else:
                model_str = "2DWavelengthDependent"
                diagnostics["reason"] = (
                    "Periods vary with wavelength (chromatic variability); "
                    "wavelength-dependent separable kernel recommended."
                )

        diagnostics["model"] = model_str

        if verbose:
            sep = "=" * 70
            print(sep)
            print("AUTO MODEL SELECTION")
            print(sep)
            print(f"Recommended model: {model_str}")
            print(f"Reason: {diagnostics['reason']}")
            print(sep)

        return model_str, diagnostics

    def fit(
        self,
        model=None,
        likelihood=None,
        num_mixtures=None,
        guess=None,
        periods=None,
        use_mls_init=True,
        use_best_band_init: bool = False,
        constraint_set=None,
        grid_size=2000,
        cuda=False,
        training_iter=300,
        max_cg_iterations=None,
        optim="AdamW",
        miniter=None,
        stop=1e-5,
        lr=0.1,
        stopavg=30,
        variance=False,
        **kwargs,
    ):
        """Fit the lightcurve

        Parameters
        ----------
        model : string or instance of gpytorch.models.GP, optional
            The model to use for the GP, by default None. If None, an
            error will be raised. If a string, it must be one of the
            following:

            Spectral mixture models:
                '1D': SpectralMixtureGPModel
                '2D': TwoDSpectralMixtureGPModel
                '1DLinear': SpectralMixtureLinearMeanGPModel
                '2DLinear': TwoDSpectralMixtureLinearMeanGPModel
                '1DSKI': SpectralMixtureKISSGPModel
                '2DSKI': TwoDSpectralMixtureKISSGPModel
                '1DLinearSKI': SpectralMixtureLinearMeanKISSGPModel
                '2DLinearSKI': TwoDSpectralMixtureLinearMeanKISSGPModel
                '2DPowerLaw': TwoDSpectralMixturePowerLawMeanGPModel
                '2DPowerLawSKI': TwoDSpectralMixturePowerLawMeanKISSGPModel
                '2DDust': TwoDSpectralMixtureDustMeanGPModel
                '2DDustSKI': TwoDSpectralMixtureDustMeanKISSGPModel

            Alternative 1D models:
                '1DQuasiPeriodic': QuasiPeriodicGPModel
                '1DMatern': MaternGPModel
                '1DPeriodicStochastic': PeriodicPlusStochasticGPModel
                '1DLinearQuasiPeriodic': LinearMeanQuasiPeriodicGPModel

            Separable 2D models:
                '2DSeparable': SeparableGPModel
                '2DAchromatic': AchromaticGPModel
                '2DWavelengthDependent': WavelengthDependentGPModel
                '2DDustMean': DustMeanGPModel
                '2DPowerLawMean': PowerLawMeanGPModel


            If an instance of a GP class, that object will be used.
        likelihood : string, None or instance of
                        gpytorch.likelihoods.likelihood.Likelihood or Constraint,
                        optional
            If likelihood is passed, it will be passed along to `set_likelihood()`
            and used to set the likelihood function for the model. For details, see
            the documentation for `set_likelihood()`.
        num_mixtures : int or None, optional
            The number of mixtures to use in the spectral mixture kernel.  By
            default ``None``, which lets the MLS initialisation (see
            ``use_mls_init``) choose the value automatically.  When
            ``use_mls_init=True`` and ``periods`` is ``None``, setting
            ``num_mixtures`` to an integer *N* overrides the automatic count:
            the first *N* significant MLS periods are used; if *N* exceeds the
            number of significant periods, non-significant peaks are added to
            make up the difference.  When ``use_mls_init=False`` and
            ``num_mixtures`` is ``None`` a fallback of 4 is used.
        guess : dict, optional
            A dictionary of the hyperparameters to use for the model and
            likelihood. The keys should be the names of the parameters, and the
            values should be Tensors containing the values of the parameters.
            If None, no hyperparameters will be set. If a hyperparameter is
            passed for a parameter that is not a model or likelihood
            parameter, it will be ignored.
        periods : array-like or None, optional
            Initial guesses for the periods (in the same units as the
            lightcurve time axis).  When provided for 1D spectral-mixture
            kernels (i.e. when ``ard_num_dims == 1``), the MLS
            initialisation is skipped entirely: ``num_mixtures`` is set to
            the number of supplied periods and the spectral-mixture kernel
            frequencies are initialised from these values.  If both
            ``periods`` and ``guess`` are supplied, entries in ``guess`` take
            priority over the period-derived frequencies.  For multi-
            dimensional spectral-mixture models (e.g. 2D kernels), the
            current implementation does not use ``periods`` to seed mixture
            means; in those cases, only explicit initial values provided via
            ``guess`` (or the model's own defaults) will be used.
        use_mls_init : bool, optional
            If ``True`` (default) and ``periods`` is ``None`` and a 1D
            spectral-mixture model string is given (``ard_num_dims == 1``),
            the Multiband Lomb-Scargle (MLS) periodogram is run first to
            estimate the number of significant periods and their frequencies,
            which are used as initial guesses for the spectral-mixture kernel
            frequencies.  Set to ``False`` to disable this behaviour and, for
            models that call it, fall back to GPyTorch's
            ``initialize_from_data``.  Note that several 2D spectral-mixture
            models do not currently call ``initialize_from_data`` at all, so
            for those models MLS-based or period-based frequency seeding is
            not applied and the underlying GPyTorch defaults are used
            instead.
        use_best_band_init : bool, optional
            If ``True`` and the lightcurve is multiband (``ndim > 1``) and
            ``use_mls_init=True`` and ``periods`` is ``None``, a 1D
            Lomb-Scargle fit on the most-sampled band is used to seed the
            spectral-mixture frequency initialisation instead of the
            standard multiband LS.  For 2D spectral-mixture models
            (``ard_num_dims == 2``, non-SKI), the fitted temporal
            frequencies are also used to initialise the temporal dimension
            of the kernel mixture means, with the minimum wavelength
            frequency (1/wavelength_span) as the default for the
            wavelength dimension, corresponding to approximately achromatic
            variability.  This can improve convergence for sources with a
            large dynamic range in the number of observations across bands.
            Has no effect for 1D lightcurves or when ``use_mls_init=False``.
        constraint_set : str or None, optional
            Name of a pre-defined source-type constraint set to apply via
            :meth:`set_default_constraints`.  When provided, the period bounds
            defined in the constraint set are also used to filter MLS peaks
            *before* the model is constructed: peaks whose frequencies fall
            outside the constraint-set allowed range are excluded from the
            initialisation (with a ``RuntimeWarning``).  Currently supported
            values are ``"LPV"`` (Long-Period Variables, minimum period 100 in
            the native time units).  Pass ``None`` (the default) to use only
            the data-driven bounds.
        grid_size : int, optional
            The number of points to use in the grid for the KISS-GP models,
            by default 2000.
        cuda : bool, optional
            Whether to use CUDA, by default False.
        training_iter : int, optional
            The number of iterations to use for training, by default 300.
        max_cg_iterations : int, optional
            The maximum number of conjugate gradient iterations to use, by
            default None. If None, gpytorch.settings.max_cg_iterations will
            be used.
        optim : str or torch.optim.Optimizer, optional
            The optimizer to use for training, by default "AdamW". If a string,
            it must be one of the following:
                'AdamW': torch.optim.AdamW
                'Adam': torch.optim.Adam
                'SGD': torch.optim.SGD
            Otherwise, it must be an instance of torch.optim.Optimizer.
        miniter : int, optional
            The minimum number of iterations to use for training, by default
            None. If None, training_iter will be used.
        stop : float, optional
            The stopping criterion for the training, by default 1e-5.
        lr : float, optional
            The learning rate to use for the optimizer, by default 0.1.
        stopavg : int, optional
            The number of iterations to use for the stopping criterion, by
            default 30.
        variance : bool, optional
            If False (default), stored uncertainties are treated as errors
            (standard deviations) and are squared before being used as noise
            variances in the likelihood.  Set to True if the stored
            uncertainties already represent variances.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the model constructor,
            likelihood constructor, or the optimizer.

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            If no model is provided.

        Notes
        -----
        Data validation, quality checks, and subsampling are performed at
        object construction time (see :meth:`__init__`).  Use the
        ``check_sampling``, ``check_variability``, and ``max_samples``
        parameters of :meth:`__init__` to control pre-processing.
        """
        # Capture the caller's original num_mixtures argument before any
        # mutation (MLS init / fallback default).  Used later to decide
        # whether to substitute the stored _model_num_mixtures.
        _num_mixtures_arg = num_mixtures

        if not hasattr(self, "likelihood"):
            self.set_likelihood(likelihood, variance=variance, **kwargs)
        elif not self.__SET_LIKELIHOOD_CALLED and likelihood is None:
            # if no likelihood is passed, we only want to set the likelihood
            # if it hasn't already been set
            self.set_likelihood(likelihood, variance=variance, **kwargs)
        elif likelihood is not None:
            self.set_likelihood(likelihood, variance=variance, **kwargs)
        # if likelihood is None and not hasattr(self, 'likelihood'):
        #     raise ValueError("""You must provide a likelihood function""")
        # elif likelihood is not None:
        #     self.set_likelihood(likelihood, **kwargs)

        # Validate explicitly-provided num_mixtures early.
        if num_mixtures is not None:
            # Must be a (non-bool) integer and strictly positive.
            if isinstance(num_mixtures, bool) or not isinstance(
                num_mixtures, int
            ):
                raise TypeError(
                    "`num_mixtures` must be a positive integer or None, "
                    f"got {num_mixtures!r} of type {type(num_mixtures)!r}."
                )
            if num_mixtures < 1:
                raise ValueError(
                    "`num_mixtures` must be a positive integer or None, "
                    f"got {num_mixtures}."
                )

        # --- MLS-based initialisation ---
        _init_freqs = None  # frequencies (raw units) to seed the SM kernel

        # Minimum frequency in raw data units: the period cannot exceed the
        # total span of the data.  Used to filter obviously unphysical MLS
        # peaks and to generate padding frequencies when not enough peaks are
        # available.
        _t_raw = (
            self._xdata_raw[:, 0] if self.ndim > 1 else self._xdata_raw
        )
        _t_span = float(_t_raw.max() - _t_raw.min())
        _freq_lower = 1.0 / _t_span if _t_span > 0 else 0.0
        _t_sorted = _t_raw.sort().values
        _t_diffs = _t_sorted[1:] - _t_sorted[:-1]
        _pos_diffs = _t_diffs[_t_diffs > 0]
        _freq_upper = (
            1.0 / (2.0 * float(_pos_diffs.min()))
            if len(_pos_diffs) > 0
            else float("inf")
        )

        if periods is not None:
            # User supplied explicit period guesses — skip MLS entirely.
            _periods_tensor = torch.as_tensor(
                periods, dtype=self._xdata_raw.dtype
            ).flatten()

            # Validate user-supplied periods: must be non-empty, finite, and > 0
            if _periods_tensor.numel() == 0:
                raise ValueError(
                    "When providing explicit `periods`, the sequence must be "
                    "non-empty."
                )
            if not torch.isfinite(_periods_tensor).all():
                raise ValueError(
                    "All values in `periods` must be finite (no NaN or inf)."
                )
            if not (_periods_tensor > 0).all():
                raise ValueError(
                    "All values in `periods` must be strictly positive."
                )
            _init_freqs = 1.0 / _periods_tensor
            num_mixtures = len(_init_freqs)
        elif use_mls_init and isinstance(model, str) and model in _SM_MODELS:
            # Compute constraint-set frequency bounds in raw data units.
            # These are used in addition to the data-span bounds to exclude
            # MLS peaks that would lie outside user-requested period limits.
            # Note: fit_LS uses Nyquist_factor > 1, so its frequencies can
            # exceed the standard Nyquist.  We therefore only apply an upper
            # frequency limit when the constraint_set explicitly demands one
            # (via a minimum-period specification); otherwise the upper bound
            # is left unrestricted (inf).
            _cs_freq_lower = _freq_lower  # default: data-span lower bound
            _cs_freq_upper = float("inf")  # no upper cap unless constraint_set
            if constraint_set is not None:
                try:
                    cs = get_constraint_set(constraint_set)
                    if "period" in cs:
                        _pb = cs["period"]
                        _p_lower_val, _p_lower_active = _pb["lower"]
                        _p_upper_val, _p_upper_active = _pb["upper"]
                        # Period lower limit → max allowed frequency
                        if _p_lower_active and _p_lower_val is not None:
                            _cs_freq_upper = min(
                                _cs_freq_upper, 1.0 / _p_lower_val
                            )
                        # Period upper limit → min allowed frequency
                        if _p_upper_active and _p_upper_val is not None:
                            _cs_freq_lower = max(
                                _cs_freq_lower, 1.0 / _p_upper_val
                            )
                except (ValueError, KeyError):
                    warnings.warn(
                        f"constraint_set={constraint_set!r} is not recognised "
                        "and will be ignored for MLS peak filtering. "
                        "Only the data-span frequency bounds will be applied.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    # Normalise invalid constraint_set so that later code does not
                    # attempt to apply or validate an unknown set again.
                    constraint_set = None

            # Run the MLS periodogram to choose num_mixtures and seed frequencies.
            try:
                _max_peaks = max(num_mixtures or 1, 10)
                if use_best_band_init and self.ndim > 1:
                    # Use a 1D LS on the most-sampled band to get reliable
                    # temporal frequency estimates instead of the multiband LS.
                    # This is beneficial when sampling is highly heterogeneous
                    # across bands: the best-sampled band provides the most
                    # accurate period constraints.
                    _best_band_lc = self._get_best_sampled_band_lc()
                    ls_freqs, ls_sig = _best_band_lc.fit_LS(
                        num_peaks=_max_peaks
                    )
                    # Compute the best-band's own Nyquist as the upper
                    # frequency bound.  The best-band 1D LS may find alias
                    # peaks above this Nyquist (when Nyquist_factor > 1);
                    # cap at the Nyquist to avoid out-of-range initialisation.
                    _bb_t = _best_band_lc._xdata_raw.sort().values
                    _bb_diffs = _bb_t[1:] - _bb_t[:-1]
                    _bb_pos = _bb_diffs[_bb_diffs > 0]
                    _bb_nyquist = (
                        float(1.0 / (2.0 * _bb_pos.min()))
                        if len(_bb_pos) > 0
                        else float("inf")
                    )
                else:
                    ls_freqs, ls_sig = self.fit_LS(num_peaks=_max_peaks)
                    _bb_nyquist = float("inf")

                # Filter peaks whose period exceeds the data span or falls
                # outside user-specified constraint-set period bounds.
                # When use_best_band_init=True also cap at the best-band
                # Nyquist to remove alias peaks that exceed the true sampling
                # limit of the best-sampled band.
                _eff_upper = min(_cs_freq_upper, _bb_nyquist)
                # Ensure that any subsequent use of the frequency upper bound
                # (e.g. for padding when num_mixtures exceeds the number of
                # LS peaks) also respects the best-band Nyquist cap.
                if use_best_band_init and self.ndim > 1:
                    _cs_freq_upper = _eff_upper
                    _freq_upper = _eff_upper
                if len(ls_freqs) > 0 and _cs_freq_lower > 0:
                    _valid = (ls_freqs >= _cs_freq_lower) & (
                        ls_freqs <= _eff_upper
                    )
                    if not _valid.all():
                        _n_filtered = int((~_valid).sum().item())
                        warnings.warn(
                            f"{_n_filtered} MLS peak(s) fell outside the "
                            f"allowed frequency range "
                            f"[{_cs_freq_lower:.4g}, {_eff_upper:.4g}] "
                            "(derived from data span"
                            + (
                                f" and constraint_set={constraint_set!r}"
                                if constraint_set is not None
                                else ""
                            )
                            + ") and were excluded from the initialisation.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        ls_freqs = ls_freqs[_valid]
                        ls_sig = ls_sig[_valid]

                if len(ls_freqs) > 0:
                    ls_sig_freqs = ls_freqs[ls_sig]
                    ls_insig_freqs = ls_freqs[~ls_sig]

                    if num_mixtures is None:
                        # Default: use only the statistically significant peaks.
                        if len(ls_sig_freqs) > 0:
                            num_mixtures = len(ls_sig_freqs)
                            _init_freqs = ls_sig_freqs
                        else:
                            # No significant peaks; fall back to the strongest one.
                            num_mixtures = 1
                            _init_freqs = ls_freqs[:1]
                    else:
                        # User specified num_mixtures: fill with significant peaks
                        # first, then non-significant ones, then pad with
                        # evenly-spaced frequencies if still not enough.
                        n_sig = len(ls_sig_freqs)
                        if num_mixtures <= n_sig:
                            _init_freqs = ls_sig_freqs[:num_mixtures]
                        else:
                            _extra = num_mixtures - n_sig
                            _available_insig = ls_insig_freqs[:_extra]
                            _init_freqs = torch.cat(
                                [ls_sig_freqs, _available_insig]
                            )
                            # Pad with additional frequencies if still short.
                            _n_pad = num_mixtures - len(_init_freqs)
                            if _n_pad > 0:
                                # Determine padding interval as the intersection of
                                # the data-based frequency range and any
                                # constraint-set bounds.
                                _pad_lower = _freq_lower
                                _pad_upper = _freq_upper
                                if _cs_freq_lower > 0:
                                    _pad_lower = max(_pad_lower, _cs_freq_lower)
                                    _pad_upper = min(_pad_upper, _cs_freq_upper)
                                if _pad_upper > _pad_lower:
                                    _msg = (
                                        f"Only {len(_init_freqs)} MLS peak(s)"
                                        f" found but {num_mixtures} were"
                                        f" requested. Padding with {_n_pad}"
                                        " evenly-spaced frequencies in"
                                        f" [{_pad_lower:.4g},"
                                        f" {_pad_upper:.4g}]."
                                    )
                                    warnings.warn(
                                        _msg,
                                        RuntimeWarning,
                                        stacklevel=2,
                                    )
                                    _pad = torch.linspace(
                                        _pad_lower,
                                        _pad_upper,
                                        _n_pad + 2,
                                        dtype=_init_freqs.dtype,
                                    )[1:-1]
                                else:
                                    _msg = (
                                        "Could not construct a valid"
                                        " frequency range for padding MLS"
                                        " initialisation; repeating the last"
                                        " available MLS frequency to reach"
                                        f" num_mixtures={num_mixtures}."
                                    )
                                    warnings.warn(
                                        _msg,
                                        RuntimeWarning,
                                        stacklevel=2,
                                    )
                                    _last_freq = _init_freqs[-1]
                                    _pad = _init_freqs.new_full(
                                        (_n_pad,), _last_freq
                                    )
                                _init_freqs = torch.cat([_init_freqs, _pad])
                else:
                    # MLS found no peaks at all; warn and fall back.

                    if num_mixtures is None:
                        num_mixtures = 4
                    # This Warning has to be raised after the if, so that the
                    # user-defined number of mixtures is used and they still see
                    # the warning if they set a value.
                    warnings.warn(
                        "MLS periodogram returned no peaks; falling back to "
                        f"num_mixtures={num_mixtures} with default initialisation.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            except Exception as exc:
                # MLS failed for any reason; fall back gracefully but warn the
                # user.  Ensure num_mixtures is set before issuing the warning.
                if num_mixtures is None:
                    inferred_count = None
                    if hasattr(self, "model") and self.model is not None:
                        inferred_count = self._infer_num_mixtures_from_model()
                    num_mixtures = (
                        inferred_count if inferred_count is not None else 4
                    )
                # Store the authoritative mixture counts now that we know the
                # fallback value.
                self._fit_num_mixtures_requested = _num_mixtures_arg
                self._fit_num_mixtures_effective = num_mixtures
                warnings.warn(
                    "MLS-based initialisation failed; falling back to "
                    f"num_mixtures={num_mixtures}. Original error was: "
                    f"{exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Final fallback when MLS init is disabled or not applicable.
        if num_mixtures is None:
            num_mixtures = 4

        if model is None and not hasattr(self, "model"):
            raise ValueError("""You must provide a model""")
        elif model is None and self.model is None:
            # The model was discarded (e.g. after band filtering). Re-create
            # it with the updated training data.
            _stored_instance = getattr(self, "_model_instance", None)
            _stored_str = getattr(self, "_model_str", None)
            # Preserve the originally configured num_mixtures when the
            # caller did not explicitly provide a value (i.e. passed None).
            # If the caller explicitly supplied num_mixtures, honour that
            # value even if it happens to equal the stored one.
            if _num_mixtures_arg is None and hasattr(self, "_model_num_mixtures"):
                stored_nm = self._model_num_mixtures
                _effective_num_mixtures = (
                    stored_nm if stored_nm is not None else num_mixtures
                )
            else:
                _effective_num_mixtures = num_mixtures
            if _stored_instance is not None:
                # User originally provided a GP instance. Re-bind it to the
                # new (filtered) training data via set_train_data() if the
                # model supports it (ExactGP), otherwise recreate via
                # set_model() which will use the same underlying class.
                self.set_likelihood(likelihood, variance=variance, **kwargs)
                if hasattr(_stored_instance, "set_train_data"):
                    _stored_instance.set_train_data(
                        inputs=self._xdata_transformed,
                        targets=self._ydata_transformed,
                        strict=False,
                    )
                    self.model = _stored_instance
                    self._make_parameter_dict()
                else:
                    # Approximate GP (e.g. SparseSpectralMixtureGPModel):
                    # cannot cheaply rebind, so fall back to raising an
                    # informative error.
                    raise ValueError(
                        "The model instance does not support set_train_data(). "
                        "Please pass model= explicitly to fit() after band "
                        "filtering, or use a string model identifier."
                    )
            elif _stored_str is not None:
                self.set_model(
                    _stored_str,
                    self.likelihood,
                    num_mixtures=_effective_num_mixtures,
                    variance=variance,
                    **kwargs,
                )
            else:
                raise ValueError("""You must provide a model""")
        elif model is not None:
            self.set_model(
                model,
                self.likelihood,
                num_mixtures=num_mixtures,
                variance=variance,
                **kwargs,
            )

        # Validate 2D setup if we have 2D data
        if self.ndim > 1:
            self._validate_2d_setup()
        if not self.__CONTRAINTS_SET:
            self.set_default_constraints(constraint_set=constraint_set)

        if not self.__CONTRAINTS_SET:
            self.set_default_constraints()

        if cuda:
            self.cuda()
        # Build the combined hyperparameter initialisation dict.
        # MLS-derived (or user-supplied) period frequencies act as the base;
        # any explicit `guess` entries take priority on top.
        _hypers_to_set = {}
        if (
            _init_freqs is not None
            and hasattr(self, "model")
            and hasattr(self.model, "covar_module")
            and hasattr(self.model.covar_module, "mixture_means")
            and getattr(self.model.covar_module, "ard_num_dims", 1) == 1
        ):
            _hypers_to_set["covar_module.mixture_means"] = _init_freqs
        elif (
            use_best_band_init
            and _init_freqs is not None
            and self.ndim > 1
            and hasattr(self, "model")
            and hasattr(self.model, "covar_module")
            and hasattr(self.model.covar_module, "mixture_means")
            and getattr(self.model.covar_module, "ard_num_dims", 1) == 2
        ):
            # For 2D SM models: initialise the temporal dimension (dim 0)
            # from the best-band 1D LS frequencies and use the minimum
            # wavelength frequency (1/wavelength_span) as a placeholder for
            # the wavelength dimension (dim 1), which encodes approximately
            # achromatic variability.  This avoids leaving all mixture means
            # at GPyTorch defaults while still seeding the most informative
            # (temporal) dimension from the best-sampled band.
            _bands_raw = self._xdata_raw[:, 1]
            _wl_span = float(_bands_raw.max() - _bands_raw.min())
            _default_wl_freq = 1.0 / _wl_span if _wl_span > 0 else 1e-6
            _n_mix = len(_init_freqs)
            # Build a [num_mixtures, 2] tensor: col 0 = temporal frequencies
            # from the best-band LS, col 1 = default wavelength frequency.
            # Using new_full preserves device and dtype of _init_freqs.
            _init_freqs_2d = torch.stack(
                [
                    _init_freqs,
                    _init_freqs.new_full((_n_mix,), _default_wl_freq),
                ],
                dim=1,  # shape: [num_mixtures, 2]
            )
            # The mixture_means constraint is derived from the temporal
            # dimension and is applied element-wise to all entries,
            # including the wavelength dimension.  The wavelength
            # frequency (1/wavelength_span) may fall below the
            # temporal-based lower bound, causing a RuntimeError.
            # Clamp to the constraint bounds only when xtransform is None
            # (i.e. raw and transformed spaces are identical).  When an
            # xtransform is active, _init_freqs_2d is still in raw units
            # but the constraint bounds are in transformed space; clamping
            # in the wrong space could create new out-of-bounds values
            # after set_hypers() applies the transform, so we skip
            # clamping and let set_hypers() handle the transform instead.
            if self.xtransform is None:
                _mixture_means_constraint = getattr(
                    self.model.covar_module,
                    "raw_mixture_means_constraint",
                    None,
                )
                if _mixture_means_constraint is not None and hasattr(
                    _mixture_means_constraint, "lower_bound"
                ):
                    _clamp_lower = float(
                        _mixture_means_constraint.lower_bound
                    )
                    _clamp_upper = (
                        float(_mixture_means_constraint.upper_bound)
                        if hasattr(_mixture_means_constraint, "upper_bound")
                        else float("inf")
                    )
                    _init_freqs_2d = _init_freqs_2d.clamp(
                        min=_clamp_lower, max=_clamp_upper
                    )
            _hypers_to_set["covar_module.mixture_means"] = _init_freqs_2d
        if guess is not None:
            _hypers_to_set.update(guess)
        if _hypers_to_set:
            self.set_hypers(_hypers_to_set)

#             if guess is not None:
#                 # self.model.initialize(**guess)
#                 self.set_hypers(guess)

        if miniter is None:
            miniter = training_iter

        if max_cg_iterations is None:
            max_cg_iterations = 10000

        # Next we probably want to report some setup info
        # later...

        # Train the model
        # self.model.train()
        # self.likelihood.train()

        # set training mode:
        self._train()

        # for param_name, param in self.model.named_parameters():
        #    print(f'Parameter name: {param_name:42} value = {param.data}')
        self.print_parameters()

        # Now actually call the trainer!
        with gpytorch.settings.max_cg_iterations(max_cg_iterations):
            self.results = train(
                self,
                maxiter=training_iter,
                miniter=miniter,
                stop=stop,
                lr=lr,
                optim=optim,
                stopavg=stopavg,
            )
        self.__FITTED_MAP = True

        return self.results

    def mcmc(
        self,
        sampler=None,
        num_samples=500,
        warmup_steps=100,
        num_chains=1,
        disable_progbar=False,
        max_cg_iterations=None,
        cuda=False,
        **kwargs,
    ):
        """Run an MCMC sampler on the model

        This function runs an MCMC sampler on the model, using the sampler
        specified in the `sampler` attribute. The results are stored in the
        `mcmc_results` attribute.

        Parameters
        ----------
        sampler : str or MCMC, optional
            The name of the sampler to use. If None, pyro.infer.mcmc.NUTS will
            be used. If a string, it must be one of the following:
                'NUTS': pyro.infer.mcmc.NUTS
                'HMC': pyro.infer.mcmc.HMC
            Otherwise, it must be an instance of pyro.infer.mcmc.MCMC.
        num_samples : int, optional
            The number of samples to draw from the posterior, by default 500.
        warmup_steps : int, optional
            The number of warmup steps to use, by default 100.
        disable_progbar : bool, optional
            Whether to disable the progress bar, by default False.
        **kwargs : dict, optional

        Returns
        -------
        mcmc_results : dict
            A dictionary containing the results of the MCMC sampling. The
            keys are the names of the parameters, and the values are the
            samples of the parameters.
        """
        if sampler is None:
            sampler = NUTS
        elif isinstance(sampler, str):
            if sampler == "NUTS":
                sampler = NUTS
            elif sampler == "HMC":
                sampler = HMC
            else:
                raise ValueError("sampler must be one of 'NUTS' or 'HMC'")
        elif not isinstance(sampler, MCMC):
            raise TypeError(
                "sampler must be either None, a string, or an instance of "
                "pyro.infer.mcmc.MCMC"
            )

        # we need to make sure that the model is in train mode
        # self._train()
        # self._eval()

        if cuda:
            self.cuda()

        if not self.__PRIORS_SET:
            self.set_default_priors()

        if max_cg_iterations is None:
            max_cg_iterations = 10000

        model = self.model

        # mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)

        # def pyro_model(x, y):
        #     model.pyro_sample_from_prior()
        #     output = model(x)
        #     loss = mll.pyro_factor(output, y)
        #     return y

        def pyro_model(x, y):
            with (
                gpytorch.settings.fast_computations(False, False, False),
                gpytorch.settings.max_cg_iterations(max_cg_iterations),
            ):
                for key in self.state_dict().keys():
                    print(key)
                    with contextlib.suppress(AttributeError):
                        print(self.state_dict()[key].device)
                        # self.state_dict()[key] = self.state_dict()[key].cuda()
                for param_name, param in self.model.named_parameters():
                    print(
                        f"Parameter name: {param_name:42} value = {param.data}, "
                        f"device = {param.data.device}"
                    )
                print(self.model.covar_module.mixture_means)
                print(self.model.covar_module.mixture_scales)
                print(self.model.covar_module.mixture_weights)
                print(self.model.covar_module.mixture_means_prior)
                print("----")
                print("Lookup dict:")
                for param_name, param in self._model_pars.items():
                    print(param)
                    print("----")
                    # print(f'Parameter name: {param_name:42} value = {param["module"].value}, device = {param["module"].value.device}')  # noqa: E501
                sampled_model = model.pyro_sample_from_prior()  # .detatch()
                output = sampled_model.likelihood(sampled_model(x))  # .detatch()
                pyro.sample("obs", output, obs=y)
            return y

        self.num_samples = num_samples

        nuts_kernel = sampler(pyro_model)
        self.mcmc_run = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            num_chains=num_chains,
            disable_progbar=disable_progbar,
        )
        import linear_operator.utils.errors as linear

        for key in self.state_dict().keys():
            print(key)
            with contextlib.suppress(AttributeError):
                print(self.state_dict()[key].device)
                # self.state_dict()[key] = self.state_dict()[key].cuda()
        for param_name, param in self.model.named_parameters():
            print(
                f"Parameter name: {param_name:42} value = {param.data}, "
                f"device = {param.data.device}"
            )

        try:
            if cuda:
                self.mcmc_run.run(
                    self._xdata_transformed.cuda(), self._ydata_transformed.cuda()
                )
            else:
                self.mcmc_run.run(self._xdata_transformed, self._ydata_transformed)
        except linear.NanError as e:
            print("NaNError encountered, returning None")
            print(list(model.named_parameters()))
            self.print_parameters()
            raise e

        self.__FITTED_MCMC = True

        # self.mcmc_run.summary(prob=0.683)
        self.inference_data = az.from_pyro(self.mcmc_run)
        samples = self.mcmc_run.get_samples()
        self.model.pyro_load_from_samples(samples)

        self.post = self.inference_data.posterior
        transformed_mcmc_periods = 1 / self.post["covar_module.mixture_means_prior"]
        raw_mcmc_periods = self.xtransform.inverse(
            torch.as_tensor(transformed_mcmc_periods.to_numpy()), shift=False
        )
        raw_mcmc_frequencies = 1 / raw_mcmc_periods
        transformed_mcmc_period_scales = 1 / (
            2 * torch.pi * self.post["covar_module.mixture_scales_prior"]
        )
        raw_mcmc_period_scales = self.xtransform.inverse(
            torch.as_tensor(transformed_mcmc_period_scales.to_numpy()), shift=False
        )
        raw_mcmc_frequency_scales = 1 / (2 * torch.pi * raw_mcmc_period_scales)

        self.inference_data.posterior["transformed_periods"] = transformed_mcmc_periods
        self.inference_data.posterior["raw_periods"] = xr.DataArray(
            raw_mcmc_periods.reshape(
                self.post["covar_module.mixture_means_prior"].shape
            ),
            coords=self.inference_data.posterior[
                "covar_module.mixture_means_prior"
            ].indexes,
        )
        self.inference_data.posterior["raw_frequencies"] = xr.DataArray(
            raw_mcmc_frequencies.reshape(
                self.post["covar_module.mixture_means_prior"].shape
            ),
            coords=self.inference_data.posterior[
                "covar_module.mixture_means_prior"
            ].indexes,
        )
        self.inference_data.posterior["transformed_period_scales"] = (
            transformed_mcmc_period_scales
        )
        self.inference_data.posterior["raw_period_scales"] = xr.DataArray(
            raw_mcmc_period_scales.reshape(
                self.post["covar_module.mixture_scales_prior"].shape
            ),
            coords=self.inference_data.posterior[
                "covar_module.mixture_scales_prior"
            ].indexes,
        )
        self.inference_data.posterior["raw_frequency_scales"] = xr.DataArray(
            raw_mcmc_frequency_scales.reshape(
                self.post["covar_module.mixture_scales_prior"].shape
            ),
            coords=self.inference_data.posterior[
                "covar_module.mixture_scales_prior"
            ].indexes,
        )

        # self.mcmc_results = mcmc(self, sampler, **kwargs)

    def summary(
        self,
        prob=0.683,
        use_arviz=True,
        var_names=None,
        filter_vars="like",
        stat_focus="median",
        **kwargs,
    ):
        """Print a summary of the results of the MCMC sampling

        Parameters
        ----------
        prob : float, optional
            The probability to use for the credible intervals, by default
            0.683.
        use_arviz : bool, optional
            Whether to use arviz to print the summary, by default True.
        var_names : list, optional
            A list of the names of the variables to include in the summary. If
            None, the variables for the mean function and the covariance
            function will be included, by default None.
        filter_vars : str, optional
            A string specifying how to filter the variables, based on
            `arviz.summary`. If None, the default behaviour of `arviz.summary`
            will be used, by default 'like'.
        stat_focus : str, optional
            A string specifying which statistic to focus on, based on
            `arviz.summary`. If None, the default behaviour of `arviz.summary`
            will be used ('mean'), by default 'median'.
        """
        if not self.__FITTED_MCMC:
            raise RuntimeError("You must first run the MCMC sampler")
        if var_names is None:
            var_names = ["mean_module", "covar_module.mixture_weights", "raw"]
        elif var_names == "all":
            var_names = None
        if stat_focus is None:
            stat_focus = "mean"
        if use_arviz:
            self.summary = az.summary(
                self.inference_data,
                round_to=2,
                hdi_prob=prob,
                var_names=var_names,
                filter_vars=filter_vars,
                stat_focus=stat_focus,
                **kwargs,
            )
        # self.mcmc_run.summary(prob=prob)
        self.diagnostics = self.mcmc_run.diagnostics()
        # figure out how to filter these before printing!
        # print(self.diagnostics)
        return self.summary

    def plot_corner(
        self,
        kind="scatter",
        var_names=None,
        filter_vars="like",
        marginals=True,
        point_estimate="median",
        **kwargs,
    ):
        """Plot a corner plot of the results of the MCMC sampling

        Parameters
        ----------
        kind : str, optional
            The kind of plot to use, based on `arviz.plot_pair`. If None, the
            default behaviour of `arviz.plot_pair` will be used, by default
            'scatter'. Other options are 'kde' and 'hexbin'.
        var_names : list, optional
            A list of the names of the variables to include in the corner plot.
            If None, the variables for the mean function and the covariance
            function will be included, by default None.
        filter_vars : str, optional
            A string specifying how to filter the variables, based on
            `arviz.plot_pair`. If None, the default behaviour of
            `arviz.plot_pair` will be used, by default 'like'.
        marginals : bool, optional
            Whether to include the marginal distributions, by default True.
        point_estimate : str, optional
            The point estimate to plot, based on `arviz.plot_pair`. If None,
            no point estimate will be plotted, by default 'median'.
        """
        if not self.__FITTED_MCMC:
            raise RuntimeError("You must first run the MCMC sampler")
        if var_names is None:
            var_names = ["mean_module", "covar_module.mixture_weights", "raw"]
        if point_estimate is None:
            point_estimate = "median"
        az.plot_pair(
            self.inference_data,
            kind=kind,
            var_names=var_names,
            filter_vars=filter_vars,
            marginals=marginals,
            point_estimate=point_estimate,
            **kwargs,
        )

    def plot_trace(self, var_names=None, filter_vars="like", figsize=None, **kwargs):
        """Plot a trace plot of the results of the MCMC sampling

        Parameters
        ----------
        var_names : list, optional
            A list of the names of the variables to include in the trace plot.
            If None, the variables for the mean function and the covariance
            function will be included, by default None.
        """
        if not self.__FITTED_MCMC:
            raise RuntimeError("You must first run the MCMC sampler")
        if var_names is None:
            # we carefully choose the default variables to plot
            # we want to plot all parameters relating to the mean function
            # but we don't want to plot all the covariance parameters
            # because those are in the transformed space. Instead, we want
            # to plot the extra parameters we have created, which are in the
            # raw space, as well as the periods and the mixture weights
            var_names = [
                "mean_module",
                "covar_module.mixture_weights",
                "raw",
            ]  # ['mean_module', 'covar_module']
        az.plot_trace(
            self.inference_data,
            var_names=var_names,
            filter_vars=filter_vars,
            figsize=figsize,
            **kwargs,
        )

    def print_periods(self):
        if self.ndim == 1:
            for i in range(len(self.model.covar_module.mixture_means)):
                if self.xtransform is None:
                    p = 1 / self.model.covar_module.mixture_means[i]
                else:
                    p = (
                        self.xtransform.inverse(
                            1 / self.model.covar_module.mixture_means[i],
                            shift=False,
                        )
                        .cpu()
                        .detach()
                        .numpy()[0]
                    )
                print(
                    f"Period {i}: "
                    f"{p}"
                    f" weight: {self.model.covar_module.mixture_weights[i]}"
                )
        elif self.ndim == 2:
            for i in range(len(self.model.covar_module.mixture_means[:, 0])):
                if self.xtransform is None:
                    p = 1 / self.model.covar_module.mixture_means[i, 0]
                else:
                    p = (
                        self.xtransform.inverse(
                            1 / self.model.covar_module.mixture_means[i, 0],
                            shift=False,
                        )
                        .cpu()
                        .detach()
                        .numpy()[0, 0]
                    )
                print(
                    f"Period {i}: "
                    f"{p}"
                    f" weight: {self.model.covar_module.mixture_weights[i]}"
                )

    def get_periods(self):
        """
        Returns a list of the periods, scales and weights of the model. This
        is useful for getting the periods after training, for example.
        """
        periods = []
        scales = []
        weights = []
        if self.ndim == 1:
            for i in range(len(self.model.sci_kernel.mixture_means)):
                if self.xtransform is None:
                    p = 1 / self.model.sci_kernel.mixture_means[i]
                    scales.append(
                        1 / (2 * torch.pi * self.model.sci_kernel.mixture_scales[i])
                    )
                else:
                    p = (
                        self.xtransform.inverse(
                            1 / self.model.sci_kernel.mixture_means[i],
                            shift=False,
                        )
                        .cpu()
                        .detach()
                        .numpy()[0]
                    )
                    scales.append(
                        self.xtransform.inverse(
                            1
                            / (2 * torch.pi * self.model.sci_kernel.mixture_scales[i]),
                            shift=False,
                        )
                        .cpu()
                        .detach()
                        .numpy()[0]
                    )
                periods.append(p)
                weights.append(
                    self.model.sci_kernel.mixture_weights[i].detach().numpy()
                )
        elif self.ndim == 2:
            for i in range(len(self.model.sci_kernel.mixture_means[:, 0])):
                if self.xtransform is None:
                    p = 1 / self.model.sci_kernel.mixture_means[i, 0]
                    scales.append(
                        1 / (2 * torch.pi * self.model.sci_kernel.mixture_scales[i, 0])
                    )
                else:
                    p = (
                        self.xtransform.inverse(
                            1 / self.model.sci_kernel.mixture_means[i, 0],
                            shift=False,
                        )
                        .cpu()
                        .detach()
                        .numpy()[0, 0]
                    )
                    scales.append(
                        self.xtransform.inverse(
                            1
                            / (
                                2
                                * torch.pi
                                * self.model.sci_kernel.mixture_scales[i, 0]
                            ),
                            shift=False,
                        )
                        .cpu()
                        .detach()
                        .numpy()[0, 0]
                    )
                periods.append(p)
                weights.append(
                    self.model.sci_kernel.mixture_weights[i].detach().numpy()
                )

        weights = np.array(weights)
        periods = np.array(periods)
        scales = np.array(scales)

        return (
            torch.as_tensor(periods),
            torch.as_tensor(weights),
            torch.as_tensor(scales),
        )

    def _infer_num_mixtures_from_model(self):
        """Infer the number of spectral-mixture components from the current model.

        Walks the kernel tree of ``self.model.sci_kernel`` (including SKI
        and separable-2D wrappers) looking for a ``mixture_means`` attribute
        and returns ``len(mixture_means)``.  Falls back to inspecting
        ``mixture_scales`` or ``mixture_weights`` if ``mixture_means`` is
        unavailable.

        Returns ``None`` if the model does not expose mixture parameters or if
        ``self.model`` has not been initialised.

        Returns
        -------
        n_mix : int or None
        """
        if not hasattr(self, "model") or self.model is None:
            return None
        if not hasattr(self.model, "sci_kernel"):
            return None
        sk = self.model.sci_kernel
        # Unwrap GridInterpolationKernel (SKI)
        actual_sk = getattr(sk, "base_kernel", sk)
        # For separable 2D, look for the time sub-kernel
        if not hasattr(actual_sk, "mixture_means"):
            from gpytorch.kernels import ProductKernel

            if isinstance(actual_sk, ProductKernel):
                for k in actual_sk.kernels:
                    inner = getattr(k, "base_kernel", k)
                    if hasattr(inner, "mixture_means"):
                        actual_sk = inner
                        break
        # Try mixture_means first, then fallbacks
        for attr in ("mixture_means", "mixture_scales", "mixture_weights"):
            if hasattr(actual_sk, attr):
                try:
                    return len(getattr(actual_sk, attr))
                except (TypeError, RuntimeError):
                    pass
        return None

    def _extract_sm_params(self):
        """Extract raw spectral-mixture parameters in physical (data) units.

        Helper for :meth:`get_period_summary`.  Extracts per-component
        means, scales, and weights from ``self.model.sci_kernel`` (or its
        ``base_kernel`` if the sci_kernel is a
        :class:`~gpytorch.kernels.GridInterpolationKernel`) and converts
        them from the transformed (normalised) frequency space back to the
        original data units, following the same convention as
        :meth:`get_periods`.

        The conversions performed here are scientifically important:

        * If ``self.xtransform is None`` the model operates directly in the
          raw time units, so ``mixture_mean`` is already the raw frequency
          and ``mixture_scale`` is already the raw frequency scale.
        * If ``self.xtransform is not None`` the model was trained in a
          normalised time coordinate.  Frequencies and scales must be
          inverse-transformed (with ``shift=False``, i.e. scaling only)
          to recover quantities in the original time units.

        For both 1-D and 2-D spectral-mixture models the *time* dimension
        (index 0 of the last axis of ``mixture_means``) is used, consistent
        with :meth:`get_periods`.  The indexing ``[i, 0, 0]`` selects
        mixture component ``i``, collapses the redundant size-1 middle
        dimension, and picks time-dimension index 0 from the last axis.
        For a 1-D kernel the shape is ``(n_mix, 1, 1)``; for a 2-D kernel
        the shape is ``(n_mix, 1, 2)``, and index 0 of the last axis is
        always the time dimension.

        Returns
        -------
        params : dict
            Keys and values (all 1-D :class:`numpy.ndarray` of length
            ``num_mixtures``):

            * ``component_frequencies``    - raw centre frequencies
            * ``component_periods``        - raw centre periods
            * ``component_frequency_scales`` - Gaussian sigma in frequency
            * ``component_period_scales``  - Gaussian sigma in period units
            * ``component_weights``        - kernel component weights

        Raises
        ------
        RuntimeError
            If the model has not been initialised.
        ValueError
            If neither the ``sci_kernel`` nor its ``base_kernel`` expose
            ``mixture_means`` (i.e. the model is not spectral-mixture).
        """
        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError(
                "Model not initialised.  Call set_model() first."
            )
        # Some SKI variants set sci_kernel to the GridInterpolationKernel
        # wrapper rather than the SpectralMixtureKernel itself.  Unwrap it.
        sk = self.model.sci_kernel
        if not hasattr(sk, "mixture_means") and hasattr(
            sk, "base_kernel"
        ):
            sk = sk.base_kernel

        if not hasattr(sk, "mixture_means"):
            raise ValueError(
                "_extract_sm_params() requires a spectral-mixture kernel.  "
                "The current sci_kernel does not expose mixture_means."
            )

        n_mix = len(sk.mixture_means)

        freqs = []
        periods = []
        freq_scales = []
        period_scales = []
        wts = []

        for i in range(n_mix):
            # -- extract the time-dimension mean and scale ------------------
            # mixture_means shape is [n_mix, 1, ard_num_dims].
            # Index [i, 0, 0] selects mixture i, the redundant size-1
            # dimension, and dimension 0 (time axis).  This is consistent
            # with the [i, 0] indexing used in get_periods() for 2-D models.
            mu_t = sk.mixture_means[i, 0, 0]
            sig_t = sk.mixture_scales[i, 0, 0]

            if self.xtransform is None:
                # No coordinate transform: mixture_mean IS the raw frequency,
                # and mixture_scale IS the raw frequency-domain half-width.
                raw_freq = float(mu_t.detach().cpu())
                raw_period = 1.0 / raw_freq
                # Convert frequency-domain scale to period-domain scale.
                raw_freq_scale = float(sig_t.detach().cpu())
                raw_period_scale = (
                    1.0 / (2.0 * np.pi * raw_freq_scale)
                )
            else:
                # The model was trained in normalised time units.
                # Inverse-transform (shift=False = scale only) to recover
                # physical (raw) period, then compute frequency from it.
                raw_period = float(
                    self.xtransform.inverse(
                        1.0 / mu_t, shift=False
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .ravel()[0]
                )
                raw_freq = 1.0 / raw_period
                # Same inverse transform for the scale parameter,
                # converting normalised frequency scale to raw period scale.
                raw_period_scale = float(
                    self.xtransform.inverse(
                        1.0 / (2.0 * torch.pi * sig_t),
                        shift=False,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .ravel()[0]
                )
                raw_freq_scale = (
                    1.0 / (2.0 * np.pi * raw_period_scale)
                )

            freqs.append(raw_freq)
            periods.append(raw_period)
            freq_scales.append(raw_freq_scale)
            period_scales.append(raw_period_scale)
            wts.append(float(sk.mixture_weights[i].detach().cpu()))

        return {
            "component_frequencies": np.array(freqs),
            "component_periods": np.array(periods),
            "component_frequency_scales": np.array(freq_scales),
            "component_period_scales": np.array(period_scales),
            "component_weights": np.array(wts),
        }

    @staticmethod
    def _sm_psd_on_grid(freq_grid, params):
        """Evaluate the total spectral-mixture PSD on a frequency grid.

        The PSD is the (non-normalised) sum of weighted Gaussians in
        frequency space::

            PSD(f) = sum_k  w_k * exp(-0.5 * ((f - mu_k) / sigma_k)^2)

        where ``mu_k``, ``sigma_k`` and ``w_k`` are the raw (physical-unit)
        component frequencies, frequency scales, and weights returned by
        :meth:`_extract_sm_params`.  Overall normalisation is not enforced
        because only the peak *location* matters for period identification.

        Parameters
        ----------
        freq_grid : numpy.ndarray
            1-D positive-frequency evaluation grid in physical units.
        params : dict
            Output of :meth:`_extract_sm_params`.

        Returns
        -------
        psd : numpy.ndarray
            PSD values on ``freq_grid``, same shape as ``freq_grid``.
        """
        psd = np.zeros_like(freq_grid, dtype=float)
        mus = params["component_frequencies"]
        sigs = params["component_frequency_scales"]
        wts = params["component_weights"]
        if not (len(mus) == len(sigs) == len(wts)):
            raise ValueError(
                f"Spectral-mixture parameter arrays have inconsistent "
                f"lengths: component_frequencies={len(mus)}, "
                f"component_frequency_scales={len(sigs)}, "
                f"component_weights={len(wts)}.  This indicates an "
                f"internal error in _extract_sm_params()."
            )
        for mu_k, sig_k, w_k in zip(mus, sigs, wts, strict=True):
            psd += w_k * np.exp(
                -0.5 * ((freq_grid - mu_k) / sig_k) ** 2
            )
        return psd

    def _detect_period_summary_backend(self):
        """Classify the fitted model into a period-summary backend family.

        Inspects the actual kernel objects attached to the model and returns
        a string label that :meth:`get_period_summary` uses to dispatch to
        the appropriate extraction routine.

        Returns
        -------
        backend : str
            One of:

            * ``"spectral_mixture"`` - SpectralMixture kernel (or SKI
              wrapper around one) - use PSD-peak extraction.
            * ``"explicit_period"`` - kernel tree contains a
              :class:`~gpytorch.kernels.PeriodicKernel` with a fitted
              ``period_length`` parameter (e.g. quasi-periodic models).
            * ``"periodic_plus_stochastic"`` - AdditiveKernel combining a
              quasi-periodic term with a stochastic (RBF) term.
            * ``"separable_2d"`` - ProductKernel with per-dimension
              ``active_dims`` (separable 2D models); the time sub-kernel
              is inspected independently.
            * ``"non_periodic"`` - no periodic structure found (e.g.
              Matérn-only model).
        """
        from gpytorch.kernels import AdditiveKernel, ProductKernel

        sk = self.model.sci_kernel
        # Unwrap GridInterpolationKernel if present
        actual_sk = getattr(sk, "base_kernel", sk)

        # 1. Spectral-mixture family (includes SKI wrappers)
        if hasattr(actual_sk, "mixture_means"):
            return "spectral_mixture"

        # 2. Additive kernel - periodic + stochastic decomposition
        if isinstance(sk, AdditiveKernel):
            return "periodic_plus_stochastic"

        # 3. Product kernel with active_dims on sub-kernels - separable 2D
        if isinstance(sk, ProductKernel):
            has_active_dims = any(
                hasattr(k, "active_dims") and k.active_dims is not None
                for k in sk.kernels
            )
            if has_active_dims:
                return "separable_2d"

        # 4. Any kernel that contains a PeriodicKernel with period_length
        if self._find_period_length_in_kernel(sk) is not None:
            return "explicit_period"

        # 5. Non-periodic fallback
        return "non_periodic"

    @staticmethod
    def _find_period_length_in_kernel(kernel):
        """Recursively search a kernel tree for a PeriodicKernel.

        Walks the kernel tree depth-first via ``base_kernel`` and
        ``kernels`` attributes and returns the first kernel instance that
        has a ``period_length`` attribute (i.e. a
        :class:`~gpytorch.kernels.PeriodicKernel`).

        Parameters
        ----------
        kernel : gpytorch.kernels.Kernel
            Root kernel to search.

        Returns
        -------
        periodic_kernel : gpytorch.kernels.Kernel or None
            The first kernel with ``period_length``, or ``None`` if none
            is found.
        """
        if hasattr(kernel, "period_length"):
            return kernel
        # Unwrap ScaleKernel / GridInterpolationKernel wrappers
        if hasattr(kernel, "base_kernel"):
            result = Lightcurve._find_period_length_in_kernel(
                kernel.base_kernel
            )
            if result is not None:
                return result
        # Recurse into ProductKernel / AdditiveKernel sub-kernels
        if hasattr(kernel, "kernels"):
            for k in kernel.kernels:
                result = Lightcurve._find_period_length_in_kernel(k)
                if result is not None:
                    return result
        return None

    def _extract_explicit_period_params(self, kernel):
        """Extract the dominant period from a kernel containing a PeriodicKernel.

        Finds the first :class:`~gpytorch.kernels.PeriodicKernel` in the
        kernel tree (via :meth:`_find_period_length_in_kernel`), reads its
        ``period_length``, and inverse-transforms it back to raw data units
        using ``self.xtransform`` (with ``shift=False`` - scaling only,
        since a period is a duration, not an absolute coordinate).

        If an RBF sub-kernel is found alongside the PeriodicKernel (as in
        the quasi-periodic product), its lengthscale is used to derive a
        practical coherence-based period interval and Q-factor.

        The scientifically important transforms are:

        * If ``self.xtransform is None``: period is stored in raw units
          already.
        * If ``self.xtransform is not None``: ``period_length`` is in the
          normalised time coordinate; ``xtransform.inverse(..., shift=False)``
          recovers the raw-unit period (``shift=False`` because a period is a
          *duration*, not an absolute time, so only the scale factor matters).

        Parameters
        ----------
        kernel : gpytorch.kernels.Kernel
            Kernel tree to search (typically ``self.model.sci_kernel`` or a
            sub-kernel thereof).

        Returns
        -------
        params : dict or None
            Dictionary with:

            * ``raw_period`` - dominant period in raw data units
            * ``raw_freq`` - ``1 / raw_period``
            * ``raw_rbf_lengthscale`` - coherence timescale in raw units, or
              ``None`` if no RBF kernel was found alongside the periodic one
            * ``period_lo``, ``period_hi`` - coherence-based interval (or
              equal to ``raw_period`` when no RBF lengthscale is available)
            * ``q_factor`` - coherence Q (RBF-based), or ``None``

            Returns ``None`` if no PeriodicKernel is found.
        """
        pk = self._find_period_length_in_kernel(kernel)
        if pk is None:
            return None

        period_norm = float(
            pk.period_length.detach().cpu().numpy().ravel()[0]
        )

        if self.xtransform is None:
            # period_length is in raw data units already
            raw_period = period_norm
        else:
            # Inverse-transform: shift=False because a period is a duration
            # (only the scale factor matters, not the origin shift).
            raw_period = float(
                self.xtransform.inverse(
                    torch.as_tensor([period_norm]), shift=False
                )
                .detach()
                .cpu()
                .numpy()
                .ravel()[0]
            )

        raw_period = abs(raw_period)
        raw_freq = 1.0 / raw_period if raw_period > 0 else np.nan

        # -- RBF lengthscale for coherence estimate (optional) --------------
        # In a quasi-periodic kernel the RBF lengthscale sets the coherence
        # time.  We search the same kernel tree for an RBF lengthscale that
        # lives alongside the PeriodicKernel.
        raw_rbf_ls = None
        if hasattr(kernel, "kernels"):
            # ProductKernel or AdditiveKernel at top level
            _kernels_to_search = list(kernel.kernels)
        elif hasattr(kernel, "base_kernel") and hasattr(
            kernel.base_kernel, "kernels"
        ):
            # ScaleKernel wrapping a ProductKernel
            _kernels_to_search = list(kernel.base_kernel.kernels)
        else:
            _kernels_to_search = []

        for k in _kernels_to_search:
            # Unwrap ScaleKernel wrappers
            inner = getattr(k, "base_kernel", k)
            if hasattr(inner, "lengthscale") and not hasattr(
                inner, "period_length"
            ):
                ls_norm = float(
                    inner.lengthscale.detach().cpu().numpy().ravel()[0]
                )
                if self.xtransform is None:
                    raw_rbf_ls = ls_norm
                else:
                    raw_rbf_ls = float(
                        self.xtransform.inverse(
                            torch.as_tensor([ls_norm]), shift=False
                        )
                        .detach()
                        .cpu()
                        .numpy()
                        .ravel()[0]
                    )
                break

        # -- period interval and Q from RBF coherence time -----------------
        if raw_rbf_ls is not None and raw_rbf_ls > 0:
            # Bandwidth from Gaussian (RBF) envelope: delta_f ~ 1/(2pi*L)
            # Linearised period uncertainty: delta_p ~ P^2 * delta_f
            delta_p = raw_period**2 / (2.0 * np.pi * raw_rbf_ls)
            period_lo = max(raw_period - delta_p / 2.0, 1e-12)
            period_hi = raw_period + delta_p / 2.0
            # Q = f_peak / FWHM_f ~ (2*pi * L) / P
            q_factor = 2.0 * np.pi * raw_rbf_ls / raw_period
        else:
            period_lo = raw_period
            period_hi = raw_period
            q_factor = None

        return {
            "raw_period": raw_period,
            "raw_freq": raw_freq,
            "raw_rbf_lengthscale": raw_rbf_ls,
            "period_lo": period_lo,
            "period_hi": period_hi,
            "q_factor": q_factor,
        }

    @staticmethod
    def _kernel_family_name(kernel):
        """Return the class name of *kernel*, or ``""`` if not available.

        Used to populate ``kernel_family`` and ``time_kernel_family`` on
        :class:`PeriodSummaryResult` objects.  If *kernel* is ``None`` an
        empty string is returned so that callers can fall back gracefully.

        Parameters
        ----------
        kernel : gpytorch.kernels.Kernel or None
            The kernel whose class name should be returned.

        Returns
        -------
        str
            ``type(kernel).__name__`` or ``""``.
        """
        if kernel is None:
            return ""
        return type(kernel).__name__

    def _get_non_periodic_summary(self, kernel=None):
        """Return a graceful period summary for a non-periodic kernel.

        Used when the model has no periodic structure (e.g.
        :class:`~pgmuvi.gps.MaternGPModel`).  All period-related fields
        are set to ``None`` or empty arrays, and the ``backend`` field is
        set to ``"non_periodic"`` so that downstream code can
        distinguish this from a genuine period summary.

        Parameters
        ----------
        kernel : gpytorch.kernels.Kernel or None, optional
            Kernel to report as the family name.  Defaults to
            ``self.model.sci_kernel`` when available.

        Returns
        -------
        summary : PeriodSummaryResult
            Consistent structured result with no period information.
        """
        _k = (
            kernel if kernel is not None
            else getattr(getattr(self, "model", None), "sci_kernel", None)
        )
        kf = self._kernel_family_name(_k)
        return PeriodSummaryResult(
            method="non_periodic_kernel",
            backend="non_periodic",
            kernel_family=kf,
            time_kernel_family=kf,
            has_stochastic_background=False,
            dominant_period=None,
            dominant_frequency=None,
            peaks=[],
            freq_grid=None,
            psd=None,
            interval_definition="none",
            notes=(
                "This kernel family does not encode a periodic timescale, "
                "so no dominant period is defined. "
                f"Kernel: {kf}."
            ),
        )

    def _get_explicit_period_summary(self, kernel=None):
        """Return a period summary for an explicit-period kernel (e.g. QP).

        Extracts the dominant period directly from a
        :class:`~gpytorch.kernels.PeriodicKernel` embedded in the model's
        ``sci_kernel``.  This is appropriate for quasi-periodic models
        (:class:`~pgmuvi.gps.QuasiPeriodicGPModel`,
        :class:`~pgmuvi.gps.LinearMeanQuasiPeriodicGPModel`) where the
        period is a directly fitted parameter.

        The uncertainty is a coherence-based proxy derived from the RBF
        lengthscale that accompanies the PeriodicKernel in the product
        ``k_periodic * k_rbf``.  It is **not** a posterior credible
        interval; MCMC-based intervals are not yet implemented.

        Parameters
        ----------
        kernel : gpytorch.kernels.Kernel or None, optional
            Kernel to search.  Defaults to ``self.model.sci_kernel``.

        Returns
        -------
        summary : PeriodSummaryResult
            Structured result.  ``freq_grid`` and ``psd`` are ``None``
            (no PSD is computed for this backend).  The dominant period
            comes from the kernel's fitted period parameter, not a PSD
            peak search.  The uncertainty interval, when present, is a
            coherence proxy from the RBF lengthscale, labelled
            ``"coherence_proxy_from_rbf_lengthscale"``.
        """
        if kernel is None:
            kernel = self.model.sci_kernel

        kf = self._kernel_family_name(kernel)
        ep = self._extract_explicit_period_params(kernel)
        if ep is None:
            return self._get_non_periodic_summary(kernel=kernel)

        raw_period = ep["raw_period"]
        raw_freq = ep["raw_freq"]
        period_lo = ep["period_lo"]
        period_hi = ep["period_hi"]
        raw_rbf_ls = ep["raw_rbf_lengthscale"]
        q_factor = ep["q_factor"]

        if raw_rbf_ls is not None:
            interval_def = "coherence_proxy_from_rbf_lengthscale"
            notes = (
                "Dominant period extracted from the fitted period_length "
                "parameter of the PeriodicKernel (explicit_period backend). "
                "The uncertainty interval is a coherence proxy derived from "
                "the RBF lengthscale; it is NOT a PSD-derived peak interval "
                "and NOT a posterior credible interval."
            )
            # Frequency interval from the period interval
            f_lo = 1.0 / period_hi if period_hi > 0 else float("nan")
            f_hi = 1.0 / period_lo if period_lo > 0 else float("nan")
        else:
            interval_def = "none"
            notes = (
                "Dominant period extracted from the fitted period_length "
                "parameter of the PeriodicKernel (explicit_period backend). "
                "No coherence timescale found; no defensible interval is "
                "reported."
            )
            period_lo = float("nan")
            period_hi = float("nan")
            f_lo = float("nan")
            f_hi = float("nan")

        # Represent the single explicit period as a PeriodPeakResult so
        # that all dict-access keys (period_interval_fwhm_like, n_peaks, …)
        # remain backward-compatible.  area_fraction=1.0 marks this as the
        # sole significant period.
        _peak = PeriodPeakResult(
            rank=1,
            frequency=raw_freq,
            period=raw_period,
            height=float("nan"),
            prominence=float("nan"),
            area_fraction=1.0,
            interval_frequency=(f_lo, f_hi),
            interval_period=(period_lo, period_hi),
            period_ratio_to_primary=1.0,
            is_candidate_lsp=False,
            notes=(
                "Coherence-proxy interval from RBF lengthscale"
                if raw_rbf_ls is not None
                else "No interval available"
            ),
        )

        return PeriodSummaryResult(
            method="explicit_period_parameter",
            backend="explicit_period",
            kernel_family=kf,
            time_kernel_family=kf,
            has_stochastic_background=False,
            dominant_period=raw_period,
            dominant_frequency=raw_freq,
            peaks=[_peak],
            freq_grid=None,
            psd=None,
            notes=notes,
            interval_definition=interval_def,
            q_factor=q_factor,
        )

    def _get_periodic_plus_stochastic_summary(self):
        """Return a period summary for a periodic-plus-stochastic kernel.

        Used for :class:`~pgmuvi.gps.PeriodicPlusStochasticGPModel`, whose
        ``sci_kernel`` is an :class:`~gpytorch.kernels.AdditiveKernel`
        combining a quasi-periodic part (``k_periodic * k_rbf``) with a
        purely stochastic RBF part.

        The dominant period is extracted from the quasi-periodic sub-kernel
        using the same logic as :meth:`_get_explicit_period_summary`.
        The stochastic component is treated as non-periodic background
        support and is **not** interpreted as an independent period.

        Returns
        -------
        summary : PeriodSummaryResult
            Structured result.  ``backend == "periodic_plus_stochastic"``,
            ``has_stochastic_background is True``.
        """
        overall_kf = self._kernel_family_name(
            getattr(self.model, "sci_kernel", None)
        )
        # The first sub-kernel of the AdditiveKernel is the QP part.
        qp_kernel = self.model.sci_kernel.kernels[0]
        qp_kf = self._kernel_family_name(qp_kernel)
        ep_summary = self._get_explicit_period_summary(kernel=qp_kernel)

        # Build updated notes that are explicit about periodic vs stochastic
        _pps_note = (
            "Periodic-plus-stochastic model (periodic_plus_stochastic "
            "backend).  The reported period comes from the periodic "
            "sub-kernel only.  The stochastic (RBF) component is treated "
            "as non-periodic background support and is NOT interpreted as "
            "an independent period.  "
        )
        return PeriodSummaryResult(
            method="periodic_plus_stochastic",
            backend="periodic_plus_stochastic",
            kernel_family=overall_kf,
            time_kernel_family=qp_kf,
            has_stochastic_background=True,
            dominant_period=ep_summary.dominant_period,
            dominant_frequency=ep_summary.dominant_frequency,
            peaks=list(ep_summary.peaks),
            freq_grid=ep_summary.freq_grid,
            psd=ep_summary.psd,
            notes=_pps_note + ep_summary.notes,
            interval_definition=ep_summary.interval_definition,
        )

    def _get_separable_2d_period_summary(self, **kwargs):
        """Return a period summary for a separable-product 2D kernel.

        For separable 2D models (e.g. :class:`~pgmuvi.gps.SeparableGPModel`,
        :class:`~pgmuvi.gps.AchromaticGPModel`,
        :class:`~pgmuvi.gps.WavelengthDependentGPModel`,
        :class:`~pgmuvi.gps.DustMeanGPModel`,
        :class:`~pgmuvi.gps.PowerLawMeanGPModel`) the ``sci_kernel`` is a
        :class:`~gpytorch.kernels.ProductKernel` whose sub-kernels each
        carry an ``active_dims`` attribute that identifies which input
        dimension (time = 0, wavelength = 1) they act on.

        This method:

        1. Identifies the time sub-kernel (``active_dims`` contains 0).
        2. Classifies the time kernel into a period-summary backend.
        3. Delegates to the appropriate backend method.
        4. Wraps the result in a ``separable_2d``-annotated
           :class:`PeriodSummaryResult` that explicitly records the
           time-kernel family used for the summary.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`_get_sm_period_summary` when the time
            kernel is spectral-mixture.

        Returns
        -------
        summary : PeriodSummaryResult
            Period summary based on the time kernel only.
            ``backend == "separable_2d"``, ``time_kernel_family`` names
            the time kernel.  The wavelength kernel does not contribute
            to the reported period.
        """

        sk = self.model.sci_kernel
        overall_kf = self._kernel_family_name(sk)
        # Identify the time sub-kernel (active_dims contains 0)
        time_kernel = None
        for k in sk.kernels:
            ad = getattr(k, "active_dims", None)
            if ad is not None and 0 in ad.tolist():
                time_kernel = k
                break

        if time_kernel is None:
            # Cannot identify time kernel; fall back to non-periodic
            np_summary = self._get_non_periodic_summary()
            return PeriodSummaryResult(
                method=np_summary.method,
                backend="separable_2d",
                kernel_family=overall_kf,
                time_kernel_family="",
                has_stochastic_background=False,
                dominant_period=np_summary.dominant_period,
                dominant_frequency=np_summary.dominant_frequency,
                peaks=list(np_summary.peaks),
                freq_grid=np_summary.freq_grid,
                psd=np_summary.psd,
                notes=(
                    "Separable 2D model: could not identify the time "
                    "sub-kernel; no period summary is available."
                ),
                interval_definition=np_summary.interval_definition,
            )

        # Classify the time kernel
        actual_tk = getattr(time_kernel, "base_kernel", time_kernel)
        tkf = self._kernel_family_name(actual_tk)
        _2d_prefix = (
            "Separable 2D model (separable_2d backend): period summary "
            f"derived from the time kernel ({tkf}) only.  "
            "The wavelength kernel does not contribute to this period "
            "determination.  "
        )

        if hasattr(actual_tk, "mixture_means"):
            # Spectral-mixture time kernel - use PSD method
            # Temporarily swap sci_kernel to expose the time kernel
            # as a stand-alone SM kernel for _extract_sm_params.
            _orig_sk = self.model.sci_kernel
            self.model.sci_kernel = actual_tk
            try:
                inner = self._get_sm_period_summary(**kwargs)
            finally:
                self.model.sci_kernel = _orig_sk
            return PeriodSummaryResult(
                method=inner.method,
                backend="separable_2d",
                kernel_family=overall_kf,
                time_kernel_family=tkf,
                has_stochastic_background=inner.has_stochastic_background,
                dominant_period=inner.dominant_period,
                dominant_frequency=inner.dominant_frequency,
                peaks=list(inner.peaks),
                freq_grid=inner.freq_grid,
                psd=inner.psd,
                notes=_2d_prefix + inner.notes,
                component_diagnostics=inner.component_diagnostics,
                interval_definition=inner.interval_definition,
                n_peaks_detected=inner.n_peaks_detected,
                n_peaks_analyzed=inner.n_peaks_analyzed,
                n_peaks_requested=inner.n_peaks_requested,
            )

        if self._find_period_length_in_kernel(time_kernel) is not None:
            # Explicit-period time kernel (e.g. quasi-periodic)
            inner = self._get_explicit_period_summary(kernel=time_kernel)
            return PeriodSummaryResult(
                method=inner.method,
                backend="separable_2d",
                kernel_family=overall_kf,
                time_kernel_family=tkf,
                has_stochastic_background=inner.has_stochastic_background,
                dominant_period=inner.dominant_period,
                dominant_frequency=inner.dominant_frequency,
                peaks=list(inner.peaks),
                freq_grid=inner.freq_grid,
                psd=inner.psd,
                notes=_2d_prefix + inner.notes,
                interval_definition=inner.interval_definition,
            )

        # Non-periodic time kernel
        return PeriodSummaryResult(
            method="non_periodic_kernel",
            backend="separable_2d",
            kernel_family=overall_kf,
            time_kernel_family=tkf,
            has_stochastic_background=False,
            dominant_period=None,
            dominant_frequency=None,
            peaks=[],
            freq_grid=None,
            psd=None,
            notes=(
                "Separable 2D model: the time kernel "
                f"({tkf}) is non-periodic, "
                "so no dominant period is defined."
            ),
            interval_definition="none",
        )

    @staticmethod
    def _find_dominant_peak_basin(psd, dominant_idx):
        """Identify the basin of the dominant PSD peak.

        Starting from the dominant peak index, walk left and right until a
        local minimum is found or the edge of the array is reached.  The
        basin is the contiguous region associated with the dominant mode.

        Parameters
        ----------
        psd : numpy.ndarray
            1-D PSD values on a frequency grid.
        dominant_idx : int
            Index of the dominant PSD peak in ``psd``.

        Returns
        -------
        basin_left : int
            Index of the left edge of the basin (inclusive).
        basin_right : int
            Index of the right edge of the basin (inclusive).
        left_at_boundary : bool
            ``True`` if the left edge reached the array boundary without
            finding a local minimum.
        right_at_boundary : bool
            ``True`` if the right edge reached the array boundary without
            finding a local minimum.
        """
        n = len(psd)

        # Walk left: stop at local minimum (psd starts rising)
        left = dominant_idx
        while left > 0 and psd[left - 1] < psd[left]:
            left -= 1
        left_at_boundary = left == 0

        # Walk right: stop at local minimum (psd starts rising)
        right = dominant_idx
        while right < n - 1 and psd[right + 1] < psd[right]:
            right += 1
        right_at_boundary = right == n - 1

        return left, right, left_at_boundary, right_at_boundary

    @staticmethod
    def _integrate_logspace(psd, freq_grid):
        """Integrate a PSD over a log-spaced frequency grid.

        Computes the integral of ``psd * freq`` with respect to
        ``log(freq)``, which equals the linear integral of ``psd`` over
        ``freq`` when the grid is logarithmically spaced.  Using this
        formulation on a log-spaced grid avoids the strong bias towards
        high frequencies that arises from naively applying the trapezoidal
        rule in linear frequency space.

        Formally this implements::

            integral of psd(f) df  ~  integral of (f * psd(f)) d(log f)
                                   ~  trapz(psd * freq, log(freq))

        Parameters
        ----------
        psd : numpy.ndarray
            1-D PSD values on ``freq_grid``.
        freq_grid : numpy.ndarray
            Positive, log-spaced frequency values with the same length as
            ``psd``.

        Returns
        -------
        integral : float
            Estimated integral value.  Always ≥ 0.
        """
        if len(freq_grid) < 2:
            return 0.0
        log_f = np.log(freq_grid)
        weights = psd * freq_grid
        try:
            return float(np.trapezoid(weights, log_f))
        except AttributeError:
            return float(np.trapz(weights, log_f))

    @staticmethod
    def _compute_equal_tail_mass_interval(
        freq_grid, psd, basin_left, basin_right, mass_level=0.68
    ):
        """Compute an equal-tail mass interval within the dominant peak basin.

        .. deprecated::
            This method is retained as a legacy helper.  The default
            ``uncertainty="peak_mass"`` path now uses
            :meth:`_compute_peak_centered_mass_interval`, which guarantees
            that the returned interval contains the peak frequency.

        Integrates the PSD (as a proxy for a probability density) over the
        basin region and returns the frequency interval that contains a
        centred fraction ``mass_level`` of the total basin mass, using an
        equal-tail (symmetric quantile) approach.

        Integration is performed in log-frequency space (``trapz(psd *
        freq, log_freq)``) to avoid the bias towards high frequencies that
        arises from naive linear-space integration on a log-spaced grid.

        Parameters
        ----------
        freq_grid : numpy.ndarray
            Full frequency evaluation grid (positive, log-spaced).
        psd : numpy.ndarray
            PSD values on ``freq_grid``.
        basin_left : int
            Left edge index of the dominant-peak basin (inclusive).
        basin_right : int
            Right edge index of the dominant-peak basin (inclusive).
        mass_level : float, optional
            Fraction of the basin mass to enclose (default 0.68 for
            approximately ``1 sigma`` coverage).  Must be in ``(0, 1)``.

        Returns
        -------
        f_lo : float
            Lower frequency bound of the equal-tail interval.
        f_hi : float
            Upper frequency bound of the equal-tail interval.
        success : bool
            ``True`` if the interval could be computed; ``False`` if the
            basin was too narrow (< 2 grid points) or the total mass was
            numerically zero.
        """
        f_basin = freq_grid[basin_left : basin_right + 1]
        p_basin = psd[basin_left : basin_right + 1]

        if len(f_basin) < 2:
            return float(f_basin[0]), float(f_basin[0]), False

        # Log-space integration via shared helper
        total_mass = Lightcurve._integrate_logspace(p_basin, f_basin)
        if total_mass <= 0:
            return float(f_basin[0]), float(f_basin[-1]), False

        # Build cumulative mass in log-space
        log_f = np.log(f_basin)
        weights = p_basin * f_basin
        cum = np.zeros(len(f_basin))
        for i in range(1, len(f_basin)):
            dlogf = log_f[i] - log_f[i - 1]
            cum[i] = cum[i - 1] + 0.5 * (
                weights[i - 1] + weights[i]
            ) * dlogf
        cum /= total_mass  # normalise to [0, 1]

        tail = (1.0 - mass_level) / 2.0

        # Interpolate lower quantile
        f_lo = float(np.interp(tail, cum, f_basin))
        # Interpolate upper quantile
        f_hi = float(np.interp(1.0 - tail, cum, f_basin))

        return f_lo, f_hi, True

    @staticmethod
    def _compute_peak_centered_mass_interval(
        freq_grid, psd, basin_left, basin_right, peak_idx, mass_level=0.68
    ):
        """Compute a peak-centered mass interval within a PSD basin.

        This is the preferred method for ``uncertainty="peak_mass"``.

        Unlike the equal-tail approach, this method **guarantees that the
        returned interval contains the peak frequency** by growing the
        interval outward from the peak, always expanding toward the side
        that contributes more mass per log-frequency unit.  This greedy
        "grow from the peak" strategy is equivalent to finding the shortest
        interval (in log-frequency space) that encloses the requested mass
        fraction and still contains the peak.

        Integration is performed in log-frequency space to avoid the
        high-frequency bias of naive linear-space trapezoidal integration
        on a log-spaced grid.

        Parameters
        ----------
        freq_grid : numpy.ndarray
            Full frequency evaluation grid (positive, log-spaced).
        psd : numpy.ndarray
            PSD values on ``freq_grid``.
        basin_left : int
            Left edge index of the basin (inclusive).
        basin_right : int
            Right edge index of the basin (inclusive).
        peak_idx : int
            Index of the peak in the full ``freq_grid``.  Must satisfy
            ``basin_left <= peak_idx <= basin_right``.
        mass_level : float, optional
            Target fraction of basin mass to enclose.  Default 0.68.

        Returns
        -------
        f_lo : float
            Lower frequency bound of the peak-centered interval.
        f_hi : float
            Upper frequency bound of the peak-centered interval.
        success : bool
            ``True`` if the interval was computed successfully.  ``False``
            if the basin has fewer than 2 grid points or the total mass is
            numerically zero.
        """
        f_basin = freq_grid[basin_left : basin_right + 1]
        p_basin = psd[basin_left : basin_right + 1]
        pk_rel = int(peak_idx) - int(basin_left)

        if len(f_basin) < 2:
            return float(f_basin[0]), float(f_basin[0]), False

        # Log-space integration weights
        log_f = np.log(f_basin)
        weights = p_basin * f_basin

        # Total basin mass via shared helper (avoids duplicating try/except)
        total_mass = Lightcurve._integrate_logspace(p_basin, f_basin)
        if total_mass <= 0:
            return float(f_basin[0]), float(f_basin[-1]), False

        # Per-segment mass in log-space
        # seg_mass[i] = mass of segment [i, i+1]
        n = len(f_basin)
        seg_mass = np.zeros(n - 1)
        for i in range(n - 1):
            dlogf = log_f[i + 1] - log_f[i]
            seg_mass[i] = 0.5 * (weights[i] + weights[i + 1]) * dlogf

        # Greedy grow from the peak: always expand into the denser side
        left_ptr = pk_rel
        right_ptr = pk_rel
        accumulated = 0.0

        while accumulated / total_mass < mass_level:
            can_go_left = left_ptr > 0
            can_go_right = right_ptr < n - 1

            if not can_go_left and not can_go_right:
                break

            if can_go_left and can_go_right:
                left_seg = seg_mass[left_ptr - 1]
                right_seg = seg_mass[right_ptr]
                if left_seg >= right_seg:
                    accumulated += left_seg
                    left_ptr -= 1
                else:
                    accumulated += right_seg
                    right_ptr += 1
            elif can_go_left:
                accumulated += seg_mass[left_ptr - 1]
                left_ptr -= 1
            else:
                accumulated += seg_mass[right_ptr]
                right_ptr += 1

        f_lo = float(f_basin[left_ptr])
        f_hi = float(f_basin[right_ptr])
        return f_lo, f_hi, True

    @staticmethod
    def _build_frequency_grid(min_freq, max_freq, n_grid, spacing="log"):
        """Build a 1-D positive-frequency evaluation grid.

        Parameters
        ----------
        min_freq : float
            Lowest frequency.  Must be strictly positive.
        max_freq : float
            Highest frequency.  Must be greater than ``min_freq``.
        n_grid : int
            Number of grid points.
        spacing : str, optional
            ``"log"`` (default) for logarithmically spaced points;
            ``"linear"`` for linearly spaced points.  The spectral-mixture
            summary uses ``"log"`` to ensure adequate low-frequency
            resolution across wide dynamic ranges.

        Returns
        -------
        freq_grid : numpy.ndarray
            1-D array of ``n_grid`` frequencies in ``[min_freq, max_freq]``.

        Notes
        -----
        If ``max_freq <= min_freq`` on entry, ``max_freq`` is automatically
        adjusted to ``min_freq * 2.0`` so that the grid is always valid.

        Raises
        ------
        ValueError
            If ``min_freq <= 0`` when ``spacing="log"``.
        """
        min_freq = float(min_freq)
        max_freq = float(max_freq)
        n_grid = int(n_grid)

        if max_freq <= min_freq:
            max_freq = min_freq * 2.0

        if spacing == "log":
            if min_freq <= 0:
                raise ValueError(
                    f"min_freq must be > 0 for log spacing, "
                    f"got {min_freq!r}"
                )
            return np.logspace(
                np.log10(min_freq), np.log10(max_freq), n_grid
            )
        return np.linspace(min_freq, max_freq, n_grid)

    @staticmethod
    def _refine_peak_region(
        freq_grid, psd, params, dominant_idx,
        f_left_approx, f_right_approx,
        pad_log_factor=0.2, n_refine=None,
    ):
        """Refine the half-max crossing estimate with a denser local grid.

        Builds a fine log-spaced grid over a padded window around the
        approximate half-max interval ``[f_left_approx, f_right_approx]``,
        recomputes the PSD, re-finds the dominant peak, and walks the new
        PSD to locate the bracketing indices for both crossings.

        Parameters
        ----------
        freq_grid : numpy.ndarray
            Global log-spaced frequency grid (used for fallback bounds).
        psd : numpy.ndarray
            PSD on ``freq_grid``.
        params : dict
            Output of :meth:`_extract_sm_params`.
        dominant_idx : int
            Index of the dominant peak on ``freq_grid``.
        f_left_approx, f_right_approx : float
            Approximate left and right half-max crossing frequencies from
            the global grid.
        pad_log_factor : float, optional
            Fractional padding in log space on each side.  Default 0.2
            (i.e. widen the local window by 20 % in log units on each side).
        n_refine : int or None, optional
            Number of points in the local grid.  Defaults to
            ``max(4 * len(freq_grid), 2000)``.  The factor of 4 ensures
            the local grid is at least 4x denser than the global grid;
            the minimum of 2000 avoids a coarse local grid when the
            global grid is small.

        Returns
        -------
        freq_fine : numpy.ndarray
            Dense local frequency grid.
        psd_fine : numpy.ndarray
            PSD on ``freq_fine``.
        dominant_idx_fine : int
            Index of the dominant peak on ``freq_fine``.
        """
        from scipy.signal import find_peaks

        if n_refine is None:
            n_refine = max(4 * len(freq_grid), 2000)
        n_refine = int(n_refine)

        dom_freq = float(freq_grid[dominant_idx])

        # Pad in log space on both sides
        log_lo = np.log10(f_left_approx) - pad_log_factor
        log_hi = np.log10(f_right_approx) + pad_log_factor

        # Clamp within the global grid bounds
        log_lo = max(log_lo, np.log10(float(freq_grid[0])))
        log_hi = min(log_hi, np.log10(float(freq_grid[-1])))

        # Ensure the dominant frequency is bracketed
        if np.log10(dom_freq) < log_lo:
            log_lo = np.log10(dom_freq) - pad_log_factor
        if np.log10(dom_freq) > log_hi:
            log_hi = np.log10(dom_freq) + pad_log_factor

        if log_hi <= log_lo:
            log_hi = log_lo + 0.1

        freq_fine = np.logspace(log_lo, log_hi, n_refine)
        psd_fine = Lightcurve._sm_psd_on_grid(freq_fine, params)

        peaks_fine, _ = find_peaks(psd_fine)
        if len(peaks_fine) == 0:
            dominant_idx_fine = int(np.argmax(psd_fine))
        else:
            dominant_idx_fine = int(
                peaks_fine[np.argmax(psd_fine[peaks_fine])]
            )

        return freq_fine, psd_fine, dominant_idx_fine

    @staticmethod
    def _interpolate_halfmax_crossing(freq_grid, psd, idx, direction, half_max):
        """Linearly interpolate the frequency where PSD crosses ``half_max``.

        Given that ``psd[idx]`` is the last point **above** (or at) the
        half-maximum on one side of the dominant peak, and ``psd[idx ±1]``
        is the first point **below** it, return the linearly interpolated
        crossing frequency.

        If the neighbouring index is out of range (the crossing was at the
        very boundary of the grid), the exact grid frequency at ``idx`` is
        returned as a fallback.

        Parameters
        ----------
        freq_grid : numpy.ndarray
            1-D frequency evaluation grid.
        psd : numpy.ndarray
            PSD values on ``freq_grid``.
        idx : int
            Index of the last point whose PSD is still at or above
            ``half_max`` on the side being interpolated.
        direction : str
            ``"left"`` or ``"right"``.  Determines which neighbour to use
            for interpolation (``idx - 1`` for left, ``idx + 1`` for right).
        half_max : float
            The half-maximum level, i.e. ``0.5 * peak_height``.

        Returns
        -------
        f_crossing : float
            Interpolated crossing frequency.
        interpolated : bool
            ``True`` if a proper bracketed interpolation was performed;
            ``False`` if the boundary fallback was used.
        """
        if direction == "left":
            neighbor = idx - 1
        else:
            neighbor = idx + 1

        if neighbor < 0 or neighbor >= len(freq_grid):
            # Boundary fallback: can't interpolate, return the grid point
            return float(freq_grid[idx]), False

        f_a = float(freq_grid[idx])
        f_b = float(freq_grid[neighbor])
        psd_a = float(psd[idx])
        psd_b = float(psd[neighbor])

        # Safeguard: if psd values don't bracket half_max (shouldn't happen
        # given how idx was found, but guard against numerical edge cases)
        if psd_a == psd_b:
            return f_a, False

        # Linear interpolation: half_max = psd_a + t * (psd_b - psd_a)
        t = (half_max - psd_a) / (psd_b - psd_a)
        f_crossing = f_a + t * (f_b - f_a)
        return float(f_crossing), True

    @staticmethod
    def _expand_psd_grid_until_contained(
        freq_grid, psd, params, dominant_idx, half_max,
        max_expansions=10, expansion_factor=2.0, n_grid=5000,
    ):
        """Expand the frequency grid until both half-max crossings are inside.

        Starting from the already-computed ``freq_grid`` / ``psd``, test
        whether the half-maximum crossings of the dominant PSD peak are
        contained within the grid.  If the left crossing is at the first
        grid point (``psd[0] >= half_max``) the low end of the grid is
        extended by dividing ``min_freq`` by ``expansion_factor``.  If the
        right crossing is at the last grid point the high end is extended by
        multiplying ``max_freq`` by ``expansion_factor``.  The dominant peak
        position is re-evaluated after each expansion to remain consistent.

        The grid is always rebuilt as a **log-spaced** grid via
        :meth:`_build_frequency_grid` to ensure adequate low-frequency
        resolution across wide dynamic ranges.

        Parameters
        ----------
        freq_grid : numpy.ndarray
            Initial evaluation grid.
        psd : numpy.ndarray
            PSD values on ``freq_grid``.
        params : dict
            Output of :meth:`_extract_sm_params`, passed to
            :meth:`_sm_psd_on_grid` for PSD recomputation.
        dominant_idx : int
            Index of the dominant PSD peak on the *current* grid.
        half_max : float
            ``0.5 * psd[dominant_idx]``.
        max_expansions : int, optional
            Maximum number of expansion iterations.  Default 10.
        expansion_factor : float, optional
            Multiplicative factor for grid edge expansion.  Default 2.0.
        n_grid : int, optional
            Number of grid points to use when rebuilding.  Default 5000.

        Returns
        -------
        freq_grid : numpy.ndarray
            Possibly expanded frequency grid.
        psd : numpy.ndarray
            PSD values on the returned ``freq_grid``.
        dominant_idx : int
            Index of the dominant peak on the returned grid.
        left_truncated : bool
            ``True`` if the left half-max crossing is still at the boundary
            after all expansion attempts.
        right_truncated : bool
            ``True`` if the right half-max crossing is still at the boundary.
        n_expansions : int
            Number of expansions that were performed.
        """
        from scipy.signal import find_peaks

        min_freq = float(freq_grid[0])
        max_freq = float(freq_grid[-1])
        n_grid = int(n_grid)
        n_expansions = 0

        for _ in range(max_expansions):
            left_truncated = psd[0] >= half_max
            right_truncated = psd[-1] >= half_max

            if not left_truncated and not right_truncated:
                break  # Both crossings are inside the grid

            if left_truncated:
                min_freq = max(min_freq / expansion_factor, 1e-12)
            if right_truncated:
                max_freq = max_freq * expansion_factor

            freq_grid = Lightcurve._build_frequency_grid(
                min_freq, max_freq, n_grid, spacing="log"
            )
            psd = Lightcurve._sm_psd_on_grid(freq_grid, params)

            # Re-find dominant peak on the new grid
            peaks, _ = find_peaks(psd)
            if len(peaks) == 0:
                dominant_idx = int(np.argmax(psd))
            else:
                dominant_idx = int(peaks[np.argmax(psd[peaks])])
            half_max = 0.5 * float(psd[dominant_idx])

            n_expansions += 1

        left_truncated = bool(psd[0] >= half_max)
        right_truncated = bool(psd[-1] >= half_max)

        return (
            freq_grid, psd, dominant_idx,
            left_truncated, right_truncated, n_expansions,
        )

    @staticmethod
    def _find_psd_peaks(freq_grid, psd):
        """Detect all local maxima in a PSD array, sorted by height.

        Returns ``(peak_indices, prominences)`` where ``peak_indices`` is a
        1-D numpy int array and ``prominences`` is a 1-D float array of the
        same length.  If no peaks are detected the global maximum is returned
        as a single peak with prominence equal to its height.

        Parameters
        ----------
        freq_grid : numpy.ndarray
            Frequency evaluation grid (unused directly; kept for API symmetry).
        psd : numpy.ndarray
            1-D PSD values.

        Returns
        -------
        peak_indices : numpy.ndarray
            Indices into ``psd`` of the detected peaks, sorted by descending
            height.
        prominences : numpy.ndarray
            Corresponding peak prominences.
        """
        from scipy.signal import find_peaks as _scipy_find_peaks

        peaks_idx, props = _scipy_find_peaks(psd, prominence=0)
        if len(peaks_idx) == 0:
            dom = int(np.argmax(psd))
            return np.array([dom]), np.array([float(psd[dom])])
        proms = props["prominences"]
        order = np.argsort(psd[peaks_idx])[::-1]
        return peaks_idx[order], proms[order]

    @staticmethod
    def _characterize_peak_basin(
        freq_grid, psd, peak_idx, mass_level=0.68
    ):
        """Characterize a single PSD peak basin.

        Finds the basin boundaries by walking left/right from the peak
        until the PSD stops decreasing, computes the peak-centered mass
        interval (which always contains the peak), and returns a summary
        dict.

        Parameters
        ----------
        freq_grid : numpy.ndarray
            Frequency evaluation grid.
        psd : numpy.ndarray
            1-D PSD values.
        peak_idx : int
            Index of the peak in ``psd``.
        mass_level : float, optional
            Fraction of basin mass to enclose.  Default 0.68.

        Returns
        -------
        info : dict
            Keys: ``height``, ``basin_left``, ``basin_right``,
            ``f_lo``, ``f_hi``, ``area_fraction``, ``mass_ok``.
        """
        peak_idx = int(peak_idx)
        height = float(psd[peak_idx])

        n = len(psd)
        left = peak_idx
        while left > 0 and psd[left - 1] < psd[left]:
            left -= 1
        right = peak_idx
        while right < n - 1 and psd[right + 1] < psd[right]:
            right += 1

        f_lo, f_hi, mass_ok = Lightcurve._compute_peak_centered_mass_interval(
            freq_grid, psd, left, right, peak_idx, mass_level=mass_level
        )

        f_basin = freq_grid[left : right + 1]
        p_basin = psd[left : right + 1]
        basin_mass = Lightcurve._integrate_logspace(p_basin, f_basin)
        total_mass = Lightcurve._integrate_logspace(psd, freq_grid)
        area_fraction = (
            basin_mass / total_mass if total_mass > 0 else float("nan")
        )

        return {
            "height": height,
            "basin_left": left,
            "basin_right": right,
            "f_lo": f_lo,
            "f_hi": f_hi,
            "area_fraction": area_fraction,
            "mass_ok": mass_ok,
        }

    @staticmethod
    def _identify_lsp_candidates(
        peaks_list,
        ratio_range=(5.0, 15.0),
        min_area_fraction=0.05,
    ):
        """Flag peaks that are candidate Long Secondary Periods (LSPs).

        A peak is flagged as a candidate LSP if its
        ``period_ratio_to_primary`` lies within ``ratio_range`` and its
        ``area_fraction`` is at least ``min_area_fraction``.

        Parameters
        ----------
        peaks_list : list[PeriodPeakResult]
            Peaks with ``period_ratio_to_primary`` already set.
        ratio_range : tuple of float, optional
            ``(min_ratio, max_ratio)`` for LSP detection.  Default
            ``(5.0, 15.0)``.
        min_area_fraction : float, optional
            Minimum basin area fraction.  Default 0.05.

        Returns
        -------
        list[PeriodPeakResult]
            Same list with ``is_candidate_lsp`` updated via
            ``dataclasses.replace()``.
        """
        updated = []
        for p in peaks_list:
            r = p.period_ratio_to_primary
            is_lsp = (
                r > 1.0
                and ratio_range[0] <= r <= ratio_range[1]
                and p.area_fraction >= min_area_fraction
            )
            updated.append(dataclasses.replace(p, is_candidate_lsp=is_lsp))
        return updated

    def _get_sm_period_summary(
        self,
        n_grid=5000,
        min_freq=None,
        max_freq=None,
        peak_threshold_rel=0.2,
        uncertainty="peak_mass",
        n_peaks=None,
        mass_level=0.68,
        classify_lsp=False,
    ):
        """Return the PSD-based period summary for a spectral-mixture model.

        Implements the core PSD-peak extraction logic used when the model
        (or the time sub-kernel of a separable 2D model) is a
        :class:`~gpytorch.kernels.SpectralMixtureKernel`.

        Parameters
        ----------
        n_grid : int, optional
            Number of points in the positive-frequency evaluation grid.
        min_freq, max_freq : float or None, optional
            Initial grid limits.  If ``None``, defaults are derived from
            the data time span and the component centres + five sigma.
            These are treated as *starting* bounds only; the grid may be
            expanded automatically.
        peak_threshold_rel : float, optional
            Relative height threshold for significant peak detection.
        uncertainty : str, optional
            Uncertainty method.  Only ``"peak_mass"`` is supported
            (``"peak_width"`` raises ``NotImplementedError``).
        n_peaks : int or None, optional
            Number of peaks to analyse.  If ``None``, defaults to the
            effective number of mixtures used at fit time.
        mass_level : float, optional
            Fraction of basin mass to enclose.  Default 0.68.
        classify_lsp : bool, optional
            If ``True``, flag candidate Long Secondary Periods.

        Returns
        -------
        summary : PeriodSummaryResult
        """
        n_grid = int(n_grid)
        params = self._extract_sm_params()

        comp_freqs = params["component_frequencies"]
        comp_scales = params["component_frequency_scales"]

        if min_freq is None:
            if self.ndim == 1:
                t_span = (
                    self._xdata_raw.max() - self._xdata_raw.min()
                ).item()
            else:
                t_col = self._xdata_raw[:, 0]
                t_span = (t_col.max() - t_col.min()).item()
            t_span = max(float(t_span), 1e-10)
            min_freq = 1.0 / t_span

        if max_freq is None:
            max_freq = float(
                np.max(comp_freqs + 5.0 * comp_scales)
            )

        min_freq = max(float(min_freq), 1e-12)
        max_freq = max(float(max_freq), min_freq * 2.0)

        freq_grid = self._build_frequency_grid(
            min_freq, max_freq, n_grid, spacing="log"
        )
        psd = self._sm_psd_on_grid(freq_grid, params)

        from scipy.signal import find_peaks as _sp_find_peaks

        _peaks, _ = _sp_find_peaks(psd)
        if len(_peaks) == 0:
            dominant_idx = int(np.argmax(psd))
        else:
            dominant_idx = int(_peaks[np.argmax(psd[_peaks])])

        peak_height = float(psd[dominant_idx])
        half_max = 0.5 * peak_height

        # -- adaptive grid expansion to contain both half-max crossings ----
        (
            freq_grid, psd, dominant_idx,
            left_truncated, right_truncated, n_expansions,
        ) = self._expand_psd_grid_until_contained(
            freq_grid, psd, params, dominant_idx, half_max,
            max_expansions=10, expansion_factor=2.0, n_grid=n_grid,
        )
        peak_height = float(psd[dominant_idx])

        # -- detect all peaks, sorted by height (descending) ---------------
        all_peak_indices, all_prominences = self._find_psd_peaks(
            freq_grid, psd
        )

        # -- determine how many peaks to analyse ---------------------------
        n_peaks_requested = n_peaks
        if n_peaks is not None:
            n_peaks_to_analyze = int(n_peaks)
        else:
            n_eff = getattr(self, "_fit_num_mixtures_effective", None)
            n_peaks_to_analyze = (
                int(n_eff) if n_eff is not None else len(all_peak_indices)
            )
        n_peaks_available = len(all_peak_indices)
        n_peaks_to_analyze = min(n_peaks_to_analyze, n_peaks_available)

        selected_indices = all_peak_indices[:n_peaks_to_analyze]
        selected_proms = all_prominences[:n_peaks_to_analyze]

        # -- characterize each selected peak --------------------------------
        dominant_freq = float(freq_grid[selected_indices[0]])
        dominant_period = 1.0 / dominant_freq

        peak_objects = []
        for rank_idx, (pidx, prom) in enumerate(
            zip(selected_indices, selected_proms, strict=True)
        ):
            info = self._characterize_peak_basin(
                freq_grid, psd, pidx, mass_level=mass_level
            )
            f_pk = float(freq_grid[pidx])
            p_pk = 1.0 / f_pk
            f_lo = info["f_lo"]
            f_hi = info["f_hi"]
            p_lo = 1.0 / f_hi if f_hi > 0 else float("nan")
            p_hi = 1.0 / f_lo if f_lo > 0 else float("nan")
            ratio = p_pk / dominant_period if dominant_period > 0 else 1.0
            peak_objects.append(
                PeriodPeakResult(
                    rank=rank_idx + 1,
                    frequency=f_pk,
                    period=p_pk,
                    height=info["height"],
                    prominence=float(prom),
                    area_fraction=info["area_fraction"],
                    interval_frequency=(f_lo, f_hi),
                    interval_period=(p_lo, p_hi),
                    period_ratio_to_primary=ratio,
                    is_candidate_lsp=False,
                    notes="",
                )
            )

        if classify_lsp:
            peak_objects = self._identify_lsp_candidates(peak_objects)

        # -- backward-compat: significant peaks via threshold ---------------
        threshold = peak_threshold_rel * peak_height
        sig_mask = psd[all_peak_indices] >= threshold
        n_sig_peaks = int(np.sum(sig_mask))

        # -- notes string ---------------------------------------------------
        dominant_info = self._characterize_peak_basin(
            freq_grid, psd, dominant_idx, mass_level=mass_level
        )
        _mass_ok = dominant_info["mass_ok"]
        _basin_l, _basin_r, _basin_left_at_bdy, _basin_right_at_bdy = (
            self._find_dominant_peak_basin(psd, dominant_idx)
        )
        _note_parts = [
            "Interval is based on the integrated PSD mass within the "
            "dominant peak basin (peak-centered shortest-mass interval). "
            "The interval is guaranteed to contain the peak frequency. "
            "Integration is performed in log-frequency space to avoid "
            "high-frequency bias on a log-spaced grid. "
            "PSD evaluated on a log-spaced frequency grid."
        ]
        if _basin_left_at_bdy:
            _note_parts.append(
                "  Basin reached the left grid boundary; "
                "left edge of the basin may be underestimated."
            )
        if _basin_right_at_bdy:
            _note_parts.append(
                "  Basin reached the right grid boundary; "
                "right edge of the basin may be underestimated."
            )
        if not _mass_ok:
            _note_parts.append(
                "  WARNING: peak-mass interval could not be computed "
                "(basin too narrow); falling back to basin edges."
            )
        if n_expansions > 0:
            _note_parts.append(
                f"  Grid expanded {n_expansions} time(s) to contain "
                "the half-maximum interval."
            )
        if left_truncated or right_truncated:
            _sides = []
            if left_truncated:
                _sides.append("left")
            if right_truncated:
                _sides.append("right")
            _note_parts.append(
                f"  WARNING: half-maximum crossing on the "
                f"{' and '.join(_sides)} side(s) may still be "
                "truncated; width estimate is a lower bound."
            )
        _sm_psd_note = (
            "Spectral-mixture model (spectral_mixture backend).  "
            "Periods are derived from peaks of the SUMMED PSD on a "
            "frequency grid — this is the literature-comparable output.  "
            "The component periods/frequencies listed in the kernel "
            "component diagnostics section are direct kernel hyperparameter "
            "values and are for diagnostic purposes only; they are NOT "
            "the reported period determinations.  "
        )
        notes = _sm_psd_note + "".join(_note_parts)

        if uncertainty == "peak_width":
            raise NotImplementedError(
                "uncertainty='peak_width' is not implemented for the "
                "spectral_mixture backend because the reported interval is "
                "still computed using the peak-centered mass method. "
                "Use uncertainty='peak_mass' instead."
            )
        _interval_def = "peak_centered_68pct_mass_interval"

        _kf = self._kernel_family_name(
            getattr(self.model, "sci_kernel", None)
        )
        _diag_notes = (
            "Spectral-mixture kernel components.  These are internal kernel "
            "parameters and are NOT independent physical periods.  "
            "The summed-PSD peaks (see 'peaks') are the "
            "literature-comparable period estimates."
        )
        _diag = ComponentDiagnosticsResult(
            component_periods=params["component_periods"],
            component_frequencies=params["component_frequencies"],
            component_weights=params["component_weights"],
            component_period_scales=params["component_period_scales"],
            component_frequency_scales=(
                params["component_frequency_scales"]
            ),
            n_components=len(params["component_periods"]),
            kernel_family=_kf,
            notes=_diag_notes,
        )
        return PeriodSummaryResult(
            method="spectral_mixture_psd_peak",
            backend="spectral_mixture",
            kernel_family=_kf,
            time_kernel_family=_kf,
            has_stochastic_background=False,
            model_name="",
            n_peaks_detected=n_sig_peaks,
            n_peaks_analyzed=len(peak_objects),
            n_peaks_requested=n_peaks_requested,
            dominant_period=dominant_period,
            dominant_frequency=dominant_freq,
            peaks=peak_objects,
            freq_grid=freq_grid,
            psd=psd,
            notes=notes,
            component_diagnostics=_diag,
            interval_definition=_interval_def,
        )

    def get_period_summary(
        self,
        n_grid=5000,
        min_freq=None,
        max_freq=None,
        peak_threshold_rel=0.2,
        uncertainty="peak_mass",
        n_peaks=None,
        mass_level=0.68,
        classify_lsp=False,
    ):
        """Return a literature-comparable period summary for the fitted model.

        Unlike :meth:`get_periods`, which returns the raw kernel-basis
        parameters of each spectral-mixture component (component centres,
        scales, and weights), this method aims to produce a *single dominant
        period* that can be directly compared to published values.

        The method dispatches to the appropriate backend based on the type of
        kernel used by the model:

        **Spectral-mixture models** (all ``"1D"``, ``"2D"``, ``"SKI"``,
        ``"PowerLaw"``, ``"Dust"`` variants):
            Constructs the total positive-frequency PSD as a sum of weighted
            Gaussians, identifies the highest PSD peak, and returns its
            location as the dominant period.  The half-maximum width of the
            peak provides a practical uncertainty interval.

        **Explicit-period models** (``"1DQuasiPeriodic"``,
        ``"1DLinearQuasiPeriodic"``):
            Reads the fitted ``period_length`` parameter directly from the
            :class:`~gpytorch.kernels.PeriodicKernel`.  The RBF lengthscale
            is used as a coherence proxy to derive a period interval and
            Q-factor.

        **Periodic-plus-stochastic** (``"1DPeriodicStochastic"``):
            Extracts the period from the quasi-periodic sub-kernel.  The
            summary notes flag the mixed periodic/stochastic nature of the
            model.

        **Separable 2D models** (``"2DSeparable"``, ``"2DAchromatic"``,
        ``"2DWavelengthDependent"``, ``"2DDustMean"``,
        ``"2DPowerLawMean"``):
            Identifies the time sub-kernel (``active_dims = [0]``) and
            applies the appropriate backend to that sub-kernel only.

        **Non-periodic models** (``"1DMatern"``):
            Returns a consistent summary dictionary with ``None`` values for
            all period-related fields rather than raising an exception, so
            that automated scripts can handle all model types gracefully.

        .. note::
            All uncertainty estimates are *practical proxies*, not posterior
            credible intervals.  MCMC-based credible intervals are not yet
            implemented.

        Parameters
        ----------
        n_grid : int, optional
            Number of points in the positive-frequency evaluation grid
            (spectral-mixture backend only).  Default 5000.
        min_freq : float or None, optional
            Minimum frequency for the evaluation grid (SM backend only).
            Defaults to ``1 / time_span``.
        max_freq : float or None, optional
            Maximum frequency for the evaluation grid (SM backend only).
            Defaults to the highest component centre plus five sigma.
        peak_threshold_rel : float, optional
            Relative height threshold for significant peaks (SM backend).
            Default 0.2.
        uncertainty : str, optional
            Uncertainty method.  Only ``"peak_mass"`` is supported for the
            spectral-mixture backend (``"peak_width"`` raises
            ``NotImplementedError``).  Non-SM backends always use their
            native interval method and ignore this parameter.  Default
            ``"peak_mass"``.
        n_peaks : int or None, optional
            Number of peaks to analyze and return in ``peaks``.  If ``None``
            (default), defaults to ``_fit_num_mixtures_effective`` when that
            attribute is available (i.e. after a call to :meth:`fit` or
            :meth:`set_model`), otherwise all detected peaks are returned.
            Pass an explicit integer to override.
        mass_level : float, optional
            Fraction of basin mass to enclose in the equal-tail interval
            (``"peak_mass"`` mode only).  Default 0.68 (~1 sigma).
        classify_lsp : bool, optional
            If ``True``, flag peaks whose period ratio to the dominant peak
            falls within the Long Secondary Period range (5-15) and whose
            basin area fraction exceeds 0.05.  Default ``False``.

        Returns
        -------
        summary : dict
            Dictionary with keys:

            * ``component_periods``          - raw kernel component periods
            * ``component_weights``          - raw kernel component weights
            * ``component_period_scales``    - raw kernel period widths
            * ``component_frequencies``      - raw kernel component freqs
            * ``component_frequency_scales`` - raw kernel frequency widths
            * ``freq_grid``  - evaluation grid (``None`` for non-PSD backends)
            * ``psd``        - PSD values (``None`` for non-PSD backends)
            * ``dominant_frequency`` - frequency of the dominant peak
              (``None`` for non-periodic models)
            * ``dominant_period``    - ``1 / dominant_frequency``
              (``None`` for non-periodic models)
            * ``period_interval_fwhm_like`` - ``(period_lo, period_hi)``
              uncertainty interval (``None`` for non-periodic models;
              kept for backward compatibility)
            * ``period_interval`` - same as ``period_interval_fwhm_like``
              (generic key independent of uncertainty method)
            * ``interval_definition`` - string describing the interval type
            * ``q_factor``        - coherence Q (``None`` if not defined)
            * ``peak_fraction``   - dominant peak height / total weight
            * ``n_significant_peaks`` - peaks above threshold
            * ``significant_periods`` - periods of significant peaks
            * ``method``  - string identifying the backend used
            * ``notes``   - additional diagnostic notes

        Raises
        ------
        RuntimeError
            If the model has not been initialised.
        NotImplementedError
            If an unsupported ``uncertainty`` method is requested.
        """
        _sm_uncertainties = {"peak_mass"}
        if uncertainty not in _sm_uncertainties:
            raise NotImplementedError(
                f"uncertainty='{uncertainty}' is not yet implemented. "
                f"Supported values: {sorted(_sm_uncertainties)!r}."
            )

        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError(
                "Model not initialised.  Call set_model() first."
            )

        backend = self._detect_period_summary_backend()

        if backend == "spectral_mixture":
            return self._get_sm_period_summary(
                n_grid=n_grid,
                min_freq=min_freq,
                max_freq=max_freq,
                peak_threshold_rel=peak_threshold_rel,
                uncertainty=uncertainty,
                n_peaks=n_peaks,
                mass_level=mass_level,
                classify_lsp=classify_lsp,
            )

        if backend == "explicit_period":
            return self._get_explicit_period_summary()

        if backend == "periodic_plus_stochastic":
            return self._get_periodic_plus_stochastic_summary()

        if backend == "separable_2d":
            return self._get_separable_2d_period_summary(
                n_grid=n_grid,
                min_freq=min_freq,
                max_freq=max_freq,
                peak_threshold_rel=peak_threshold_rel,
                uncertainty=uncertainty,
                n_peaks=n_peaks,
                mass_level=mass_level,
                classify_lsp=classify_lsp,
            )

        # backend == "non_periodic"
        return self._get_non_periodic_summary()

    def plot_period_summary(
        self,
        summary=None,
        show=True,
        log_freq=True,
        show_full_psd=None,
        max_peaks_to_mark=3,
        **kwargs,
    ):
        """Plot the period summary from :meth:`get_period_summary`.

        Produces a matplotlib figure appropriate for the type of period
        summary:

        * **Spectral-mixture PSD summary with a single analyzed peak**
          (``PeriodSummaryResult``, ``n_peaks_analyzed == 1``): generates a
          **single peak-centered panel** zoomed in on the dominant peak.
          Pass ``show_full_psd=True`` to add a second full-range PSD panel.
        * **Spectral-mixture PSD summary with structured peaks**
          (``PeriodSummaryResult``, ``n_peaks_analyzed > 1``): generates a
          **multi-panel figure** with the full PSD in the top panel and one
          zoomed panel per analyzed peak below.  Each peak is labeled
          P1, P2, … with a distinct color.
        * **Spectral-mixture PSD summary (plain dict)**: plots the PSD curve
          with the dominant peak and dotted lines for other significant peaks.
        * **Explicit-period summary** (e.g. quasi-periodic): plots a single
          vertical line at the dominant frequency with an annotated period,
          interval, and Q-factor.  No PSD curve is drawn because none is
          computed for this backend.
        * **Non-periodic summary**: produces a simple figure with explanatory
          text stating that no dominant period is defined for this kernel.

        The figure type is determined by ``summary["method"]`` and by whether
        ``summary["freq_grid"]`` is ``None``.

        Parameters
        ----------
        summary : dict or None, optional
            Output of :meth:`get_period_summary`.  If ``None``, it is
            computed automatically.  Extra keyword arguments (``**kwargs``)
            are forwarded to :meth:`get_period_summary`.
        show : bool, optional
            If ``True`` (default), call ``plt.show()``.  If ``False``,
            return ``(fig, ax)`` for further customisation.
        log_freq : bool, optional
            If ``True`` (default), plot the x-axis (frequency) on a log
            scale.  Ignored for non-periodic summaries.
        show_full_psd : bool or None, optional
            Controls whether a full-range PSD panel is included in the
            single-peak case.  When ``None`` (default), a full-range panel
            is *not* added in single-peak mode (the main panel is already
            peak-centered) but *is* included in multi-peak mode.  Set to
            ``True`` to force a full-range panel even in single-peak mode;
            set to ``False`` to suppress it even in multi-peak mode.
        max_peaks_to_mark : int, optional
            Maximum number of peaks to mark on the plot.  In multi-peak
            mode this also limits the number of zoom panels created.
            Default is ``3``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`get_period_summary` when ``summary`` is ``None``.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
            Returned when ``show=False``; otherwise ``None``.
            For the multi-panel case ``ax`` is the top axes.
        """
        if summary is None:
            summary = self.get_period_summary(**kwargs)

        method = summary.get("method", "")
        has_psd = summary["freq_grid"] is not None

        # -- non-periodic: informational plot only -------------------------
        if method == "non_periodic_kernel" or (
            summary["dominant_period"] is None
        ):
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.text(
                0.5,
                0.5,
                summary.get(
                    "notes",
                    "No dominant period defined for this kernel.",
                ),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                wrap=True,
            )
            ax.set_axis_off()
            ax.set_title("Period summary")
            if show:
                plt.show()
                return None
            return fig, ax

        # -- common fields -------------------------------------------------
        f_peak = summary["dominant_frequency"]
        p_dom = summary["dominant_period"]
        # Prefer the generic key; fall back to the legacy key for old summaries
        interval = summary.get(
            "period_interval", summary.get("period_interval_fwhm_like")
        )
        interval_definition = summary.get("interval_definition", "")
        q = summary["q_factor"]
        n_sig = summary["n_significant_peaks"]

        # Build a human-readable interval type label for annotations
        _interval_labels = {
            "equal_tail_68pct_peak_mass": "68% peak mass interval",
            "peak_centered_68pct_mass_interval": "68% peak-centered mass interval",
            "half_maximum_fwhm_like": "half-max interval",
            "coherence_proxy": "coherence-proxy interval",
            "coherence_proxy_from_rbf_lengthscale": (
                "coherence-proxy interval (RBF lengthscale)"
            ),
        }
        interval_label = _interval_labels.get(
            interval_definition, interval_definition or "interval"
        )

        # Decide whether we have a structured PeriodSummaryResult with peaks.
        # Plain-dict summaries (non-SM backends) do not have a .peaks attr.
        structured_peaks = getattr(summary, "peaks", None)
        has_structured_peaks = (
            structured_peaks is not None and len(structured_peaks) > 0
        )

        # -- colour palette for per-peak markers ---------------------------
        # crimson = P1 (dominant), then cycling through a friendly palette
        _peak_colors = [
            "crimson",
            "darkorange",
            "forestgreen",
            "mediumpurple",
            "saddlebrown",
            "deepskyblue",
        ]

        def _peak_color(rank):
            """Return the color for a peak by rank (1-indexed)."""
            idx = max(rank - 1, 0)
            return _peak_colors[idx % len(_peak_colors)]

        # ------------------------------------------------------------------
        # Helpers shared by both structured-peak plot paths
        # ------------------------------------------------------------------
        def _zoom_window(pk, freq_grid):
            """Return (f_win_lo, f_win_hi, f_zoom, p_zoom) for one peak.

            The window is centered on the peak and expanded symmetrically
            around it.  If the interval bounds are finite and sensible the
            interval half-width is used as the core; otherwise a ±25%
            fallback is applied.  A ±10% emergency fallback is used when
            the resulting slice is too narrow.
            """
            f_ctr = pk.frequency
            p_lo, p_hi = pk.interval_period
            if (
                np.isfinite(p_lo) and np.isfinite(p_hi)
                and p_lo > 0 and p_hi > 0
            ):
                f_int_lo = 1.0 / p_hi
                f_int_hi = 1.0 / p_lo
                # Half-width of the interval, but at least 10% of peak freq
                half = max(0.5 * (f_int_hi - f_int_lo), 0.1 * f_ctr)
                # Expand by 50% symmetrically around the peak
                f_win_lo = max(f_ctr - 1.5 * half, freq_grid[0])
                f_win_hi = min(f_ctr + 1.5 * half, freq_grid[-1])
            else:
                # Fallback: ±25% symmetric window
                half = 0.25 * f_ctr
                f_win_lo = max(f_ctr - half, freq_grid[0])
                f_win_hi = min(f_ctr + half, freq_grid[-1])
            mask = (freq_grid >= f_win_lo) & (freq_grid <= f_win_hi)
            f_zoom = freq_grid[mask]
            p_zoom = psd[mask]
            if len(f_zoom) < 2:
                # Emergency: ±10% around peak
                f_win_lo = f_ctr * 0.9
                f_win_hi = f_ctr * 1.1
                mask = (freq_grid >= f_win_lo) & (freq_grid <= f_win_hi)
                f_zoom = freq_grid[mask]
                p_zoom = psd[mask]
            return f_win_lo, f_win_hi, f_zoom, p_zoom

        def _draw_peak_zoom(panel_ax, pk, f_win_lo, f_win_hi,
                            f_zoom, p_zoom):
            """Populate a zoom panel for one peak."""
            col = _peak_color(pk.rank)
            panel_ax.plot(f_zoom, p_zoom, color="steelblue", lw=1.5)
            panel_ax.axvline(pk.frequency, color=col, lw=1.5, ls="--")
            p_lo, p_hi = pk.interval_period
            if (
                np.isfinite(p_lo) and np.isfinite(p_hi) and p_lo > 0
            ):
                f_lo_int = 1.0 / p_hi
                f_hi_int = 1.0 / p_lo
                if (
                    f_lo_int < f_hi_int
                    and f_lo_int >= f_win_lo
                    and f_hi_int <= f_win_hi
                ):
                    panel_ax.axvspan(
                        f_lo_int, f_hi_int,
                        alpha=0.25, color=col,
                        label=(
                            f"{interval_label}  "
                            f"[{p_lo:.4g}, {p_hi:.4g}]"
                        ),
                    )
            _ratio_str = (
                f"  ratio={pk.period_ratio_to_primary:.3g}"
                if pk.rank > 1
                else ""
            )
            panel_ax.set_title(
                f"P{pk.rank}  period = {pk.period:.6g}{_ratio_str}"
            )
            if log_freq:
                panel_ax.set_xscale("log")
            panel_ax.set_xlabel("Frequency")
            panel_ax.set_ylabel("PSD")
            panel_ax.legend(fontsize=7, loc="upper left")

        # ------------------------------------------------------------------
        # Structured PeriodSummaryResult with PSD available
        # ------------------------------------------------------------------
        if has_structured_peaks and has_psd:
            freq_grid = summary["freq_grid"]
            psd = summary["psd"]
            # Limit to max_peaks_to_mark peaks
            _peaks_to_plot = structured_peaks[:max_peaks_to_mark]
            _n_peaks = len(_peaks_to_plot)
            # Determine whether we are in single-peak mode.
            # show_full_psd=None means: auto (False for 1 peak, True for >1).
            _single_peak = _n_peaks == 1
            _include_full = (
                show_full_psd
                if show_full_psd is not None
                else not _single_peak
            )

            if _single_peak:
                # -------------------------------------------------------
                # Single-peak mode: one peak-centered panel (+ optional
                # full-PSD panel if show_full_psd=True was requested).
                # -------------------------------------------------------
                pk = _peaks_to_plot[0]
                col = _peak_color(pk.rank)
                f_win_lo, f_win_hi, f_zoom, p_zoom = _zoom_window(
                    pk, freq_grid
                )

                if _include_full:
                    fig, axes = plt.subplots(
                        2, 1, figsize=(9, 7), squeeze=False
                    )
                    axes = axes[:, 0]
                    ax = axes[0]  # main = peak-centered
                    ax_full = axes[1]
                else:
                    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
                    ax_full = None

                # Main panel: peak-centered zoom
                _draw_peak_zoom(ax, pk, f_win_lo, f_win_hi, f_zoom, p_zoom)
                ax.set_title(
                    f"Period summary - dominant peak  "
                    f"(P = {pk.period:.6g})"
                )

                if ax_full is not None:
                    # Optional full-range panel below
                    ax_full.plot(
                        freq_grid, psd,
                        color="steelblue", lw=1.5, label="PSD"
                    )
                    ax_full.axvline(
                        pk.frequency, color=col, lw=1.5, ls="--",
                        label=f"P1  period={pk.period:.4g}",
                    )
                    p_lo_fp, p_hi_fp = pk.interval_period
                    if (
                        np.isfinite(p_lo_fp) and np.isfinite(p_hi_fp)
                        and p_lo_fp > 0 and p_hi_fp > 0
                    ):
                        f_lo_int = 1.0 / p_hi_fp
                        f_hi_int = 1.0 / p_lo_fp
                        if f_lo_int < f_hi_int:
                            ax_full.axvspan(
                                f_lo_int, f_hi_int,
                                alpha=0.15, color=col,
                                label=(
                                    f"{interval_label}  "
                                    f"[{p_lo_fp:.4g}, {p_hi_fp:.4g}]"
                                ),
                            )
                    if log_freq:
                        ax_full.set_xscale("log")
                    ax_full.set_ylabel("PSD")
                    ax_full.set_title(
                        f"Period summary - full PSD ({method})"
                    )
                    ax_full.legend(fontsize=7, loc="upper left", ncol=2)

            else:
                # -------------------------------------------------------
                # Multi-peak mode: full PSD top + one zoom panel per peak
                # (limited to max_peaks_to_mark peaks)
                # -------------------------------------------------------
                n_panels = 1 + _n_peaks
                fig, axes = plt.subplots(
                    n_panels, 1,
                    figsize=(9, 3.5 + 2.5 * _n_peaks),
                    squeeze=False,
                )
                axes = axes[:, 0]
                ax = axes[0]  # top panel = full PSD

                # Top panel: full PSD
                ax.plot(
                    freq_grid, psd, color="steelblue", lw=1.5, label="PSD"
                )
                for pk in _peaks_to_plot:
                    col = _peak_color(pk.rank)
                    ax.axvline(
                        pk.frequency,
                        color=col,
                        lw=1.5,
                        ls="--",
                        label=f"P{pk.rank}  period={pk.period:.4g}",
                    )
                    p_lo, p_hi = pk.interval_period
                    if (
                        np.isfinite(p_lo) and np.isfinite(p_hi)
                        and p_lo > 0 and p_hi > 0
                    ):
                        f_lo_int = 1.0 / p_hi
                        f_hi_int = 1.0 / p_lo
                        if f_lo_int < f_hi_int:
                            _span_label = (
                                f"{interval_label}  "
                                f"[{p_lo:.4g}, {p_hi:.4g}]"
                                if pk.rank == 1
                                else None
                            )
                            ax.axvspan(
                                f_lo_int, f_hi_int,
                                alpha=0.15, color=col,
                                label=_span_label,
                            )
                if log_freq:
                    ax.set_xscale("log")
                ax.set_ylabel("PSD")
                ax.set_title(f"Period summary - full PSD ({method})")
                ax.legend(fontsize=7, loc="upper left", ncol=2)

                # Per-peak zoom panels (one per plotted peak)
                for i, pk in enumerate(_peaks_to_plot):
                    panel_ax = axes[i + 1]
                    f_win_lo, f_win_hi, f_zoom, p_zoom = _zoom_window(
                        pk, freq_grid
                    )
                    _draw_peak_zoom(
                        panel_ax, pk, f_win_lo, f_win_hi, f_zoom, p_zoom
                    )

            fig.tight_layout()
            if show:
                plt.show()
                return None
            return fig, ax

        # ------------------------------------------------------------------
        # Single-panel fallback (non-structured or no PSD)
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        # -- PSD curve (spectral-mixture only) -----------------------------
        if has_psd:
            freq_grid = summary["freq_grid"]
            psd = summary["psd"]
            ax.plot(
                freq_grid, psd, color="steelblue", lw=1.5, label="PSD"
            )

        # -- dominant peak marker -----------------------------------------
        ax.axvline(
            f_peak,
            color="crimson",
            lw=1.5,
            ls="--",
            label=f"Dominant peak  P = {p_dom:.4g}",
        )

        # -- period interval shaded band (if finite interval) --------------
        if interval is not None:
            period_lo, period_hi = interval
            f_left = (
                1.0 / period_hi if period_hi and period_hi > 0 else None
            )
            f_right = (
                1.0 / period_lo if period_lo and period_lo > 0 else None
            )
            if (
                f_left is not None and f_right is not None
                and np.isfinite(f_left) and np.isfinite(f_right)
                and f_left < f_right
            ):
                ax.axvspan(
                    f_left,
                    f_right,
                    alpha=0.25,
                    color="crimson",
                    label=(
                        f"{interval_label}  "
                        f"[{period_lo:.4g}, {period_hi:.4g}]"
                    ),
                )

        # -- other significant peaks from structured summary ---------------
        if has_structured_peaks:
            for pk in structured_peaks[1:max_peaks_to_mark]:
                col = _peak_color(pk.rank)
                ax.axvline(
                    pk.frequency,
                    color=col,
                    lw=1.0,
                    ls=":",
                    alpha=0.9,
                    label=f"P{pk.rank}  period={pk.period:.4g}",
                )
        else:
            sig_periods = summary.get("significant_periods", np.array([]))
            for sp in sig_periods:
                sf = 1.0 / sp
                if abs(sf - f_peak) > 1e-12 * max(f_peak, 1e-12):
                    ax.axvline(
                        sf,
                        color="darkorange",
                        lw=1.0,
                        ls=":",
                        alpha=0.8,
                    )

        # -- text annotation -----------------------------------------------
        if q is not None and np.isfinite(q):
            q_str = f"{q:.2f}"
        elif q is not None and np.isinf(q):
            q_str = "inf"
        else:
            q_str = "N/A"

        if interval is not None:
            p_lo, p_hi = interval
            int_str = f"[{p_lo:.4g}, {p_hi:.4g}]"
        else:
            int_str = "N/A"

        ann_lines = [
            f"Dominant period:   {p_dom:.6g}",
            f"Interval ({interval_label}): {int_str}",
            f"Q-factor:          {q_str}",
            f"Significant peaks: {n_sig}",
        ]
        ax.text(
            0.97,
            0.97,
            "\n".join(ann_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", alpha=0.8
            ),
        )

        if log_freq:
            ax.set_xscale("log")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("PSD" if has_psd else "")
        ax.set_title(f"Period summary ({method})")
        ax.legend(fontsize=8, loc="upper left")

        if show:
            plt.show()
            return None
        return fig, ax

    # ------------------------------------------------------------------
    # High-level output-writing convenience
    # ------------------------------------------------------------------

    def _save_period_summary_figure(
        self,
        summary,
        filename,
        plot_kwargs=None,
        close_figure=True,
        dpi=150,
    ):
        """Internal helper for write_period_summary_outputs().

        Calls :meth:`plot_period_summary` with ``show=False``, saves the
        resulting figure, and optionally closes it.

        Parameters
        ----------
        summary : dict or PeriodSummaryResult
            Pre-computed period summary (passed straight through to
            :meth:`plot_period_summary`).
        filename : str or Path-like
            Destination path for the PNG (or any format supported by
            matplotlib's ``savefig``).
        plot_kwargs : dict or None, optional
            Extra keyword arguments forwarded to :meth:`plot_period_summary`.
        close_figure : bool, optional
            If ``True`` (default), call ``plt.close(fig)`` after saving.
        dpi : int, optional
            Resolution in dots per inch, default ``150``.

        Returns
        -------
        pathlib.Path
            Path to the saved figure file (same as *filename* as a
            ``pathlib.Path``; may be relative).
        """
        from pathlib import Path

        if plot_kwargs is None:
            plot_kwargs = {}
        path = Path(filename)
        result = self.plot_period_summary(
            summary=summary, show=False, **plot_kwargs
        )
        # plot_period_summary returns None when show=True; that should not
        # happen here (we always pass show=False), but guard defensively.
        if result is None:
            return path
        fig, _ax = result
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if close_figure:
            plt.close(fig)
        return path

    def write_period_summary_outputs(
        self,
        text_file=None,
        png_file=None,
        json_file=None,
        summary=None,
        show=False,
        close_figure=True,
        include_components=True,
        include_peaks=True,
        include_psd_info=False,
        include_psd_in_json=False,
        summary_kwargs=None,
        plot_kwargs=None,
    ):
        """Write period-summary outputs (text, PNG, JSON) to disk.

        This is a high-level **convenience wrapper** around:

        * :meth:`get_period_summary` — computes the summary if not supplied
        * :meth:`PeriodSummaryResult.write_text` — human-readable text report
        * :meth:`_save_period_summary_figure` — period-summary figure (PNG)
        * :meth:`PeriodSummaryResult.write_json` — machine-readable JSON export

        The method writes only the files whose paths are provided. Pass
        ``text_file``, ``png_file``, and/or ``json_file`` in any combination.

        Parameters
        ----------
        text_file : str, Path-like, or None, optional
            If given, the human-readable period-summary text is written here.
            The output is intended for direct reading by a researcher: it
            includes the dominant period, peak table, kernel-component
            diagnostics, and (optionally) PSD grid information.
        png_file : str, Path-like, or None, optional
            If given, the period-summary figure is saved here.  The PNG is a
            visualisation of the analyzed peak structure produced by
            :meth:`plot_period_summary`.
        json_file : str, Path-like, or None, optional
            If given, a machine-readable JSON export is written here.  The
            JSON contains the same information as the text report plus the
            raw array data (unless *include_psd_in_json* is ``False``).
        summary : dict or PeriodSummaryResult or None, optional
            A pre-computed period summary returned by
            :meth:`get_period_summary`.  If ``None`` (default) the summary is
            computed by calling ``get_period_summary(**summary_kwargs)``.
            Supplying a pre-computed summary avoids redundant computation when
            multiple output files are requested.
        show : bool, optional
            Passed through to :meth:`plot_period_summary`.  Ignored when
            *png_file* is ``None``.  Default is ``False``.
        close_figure : bool, optional
            If ``True`` (default), close the matplotlib figure after saving.
            Set to ``False`` to keep the figure in memory for further
            inspection.
        include_components : bool, optional
            Forwarded to :meth:`PeriodSummaryResult.write_text`.  Controls
            whether the kernel-component diagnostics block appears in the text
            output.  Default is ``True``.
        include_peaks : bool, optional
            Forwarded to :meth:`PeriodSummaryResult.write_text`.  Controls
            whether the analyzed-peaks block appears in the text output.
            Default is ``True``.
        include_psd_info : bool, optional
            Forwarded to :meth:`PeriodSummaryResult.write_text`.  Controls
            whether PSD grid statistics appear in the text output.  Default is
            ``False``.
        include_psd_in_json : bool, optional
            Forwarded to :meth:`PeriodSummaryResult.write_json`.  When
            ``True`` the full frequency grid and PSD arrays are embedded in
            the JSON file.  Default is ``False`` (arrays are omitted to keep
            the file small).
        summary_kwargs : dict or None, optional
            Extra keyword arguments forwarded to :meth:`get_period_summary`
            when *summary* is ``None``.  Ignored if *summary* is supplied.
        plot_kwargs : dict or None, optional
            Extra keyword arguments forwarded to :meth:`plot_period_summary`
            (and thus to :meth:`_save_period_summary_figure`).  Ignored when
            *png_file* is ``None``.

        Returns
        -------
        PeriodSummaryResult or dict
            The period summary (computed or passed in).

        Examples
        --------
        Write all three output types in one call::

            lc.write_period_summary_outputs(
                text_file="results/summary.txt",
                png_file="results/summary.png",
                json_file="results/summary.json",
            )

        Reuse an existing summary to avoid recomputation::

            s = lc.get_period_summary()
            lc.write_period_summary_outputs(
                summary=s,
                text_file="results/summary.txt",
                png_file="results/summary.png",
            )
        """
        if summary_kwargs is None:
            summary_kwargs = {}
        if summary is None:
            summary = self.get_period_summary(**summary_kwargs)
        elif summary_kwargs:
            warnings.warn(
                "summary_kwargs are ignored because a pre-computed summary "
                "was supplied via the summary= argument.",
                UserWarning,
                stacklevel=2,
            )

        if text_file is not None:
            summary.write_text(
                text_file,
                include_components=include_components,
                include_peaks=include_peaks,
                include_psd_info=include_psd_info,
            )

        if json_file is not None:
            summary.write_json(json_file, include_psd=include_psd_in_json)

        if png_file is not None:
            self._save_period_summary_figure(
                summary,
                png_file,
                plot_kwargs=plot_kwargs,
                close_figure=close_figure,
            )

        return summary

    def get_parameters(self, raw=False, transform=True):
        """
        Returns a dictionary of the parameters of the model, with the keys
        being the names of the parameters and the values being the values of
        the parameters. This is useful for getting the values of the parameters
        after training, for example.

        The routine is rather hacky, since there is no built-in way to get the
        unconstrained values of the parameters from the model without knowing
        exactly what they are ahead of time. This routine therefore gets the
        names of the raw parameters, and then uses those names with string
        manipulation and `__getattr__` to get the values of the constrained
        parameters.

        Parameters
        ----------
        raw : bool, default False
            If True, returns the raw values of the parameters, otherwise
            returns the constrained values of the parameters.

        Returns
        -------
        pars : dict
            A dictionary of the parameters of the model, with the keys
            being the names of the parameters and the values being the values
            of the parameters.
        """
        pars = {}
        pars_to_transform = {
            "x": ["mixture_means", "mixture_scales"],
            "y": ["noise", "mean_module"],
        }
        for param_name, param in self.model.named_parameters():
            comps = list(param_name.split("."))
            if not raw and "raw" in param_name:
                # This is a constrained parameter, so we need to get the
                # unconstrained value
                pn = ".".join([c.lstrip("raw_") for c in comps])
                tmp = self.model.__getattr__(comps[0])
                for i in range(1, len(comps)):
                    c = comps[i] if "raw" not in comps[i] else comps[i].lstrip("raw_")
                    try:
                        tmp = tmp.__getattr__(c)
                    except AttributeError:
                        tmp = tmp.__getattribute__(c)
                if (
                    any(p in pn for p in pars_to_transform["x"])
                    and transform
                    and self.xtransform is not None
                ):
                    d = 1 / self.xtransform.inverse(1 / tmp.data, shift=False)
                elif (
                    any(p in pn for p in pars_to_transform["y"])
                    and transform
                    and self.ytransform is not None
                ):
                    d = self.ytransform.inverse(tmp.data)
                else:
                    d = tmp.data
                pars[pn] = d
            else:
                # Either we actually want the raw values, or it's not a
                # constrained parameter
                if (
                    any(p in param_name for p in pars_to_transform["x"])
                    and transform
                    and self.xtransform is not None
                ):
                    d = 1 / self.xtransform.inverse(1 / param.data, shift=False)
                elif (
                    any(p in param_name for p in pars_to_transform["y"])
                    and transform
                    and self.ytransform is not None
                ):
                    d = self.ytransform.inverse(param.data)
                else:
                    d = param.data
                pars[param_name] = d
        return pars

    def print_parameters(self, raw=False):
        """
        Prints the parameters of the model, with the keys being the names of
        the parameters and the values being the values of the parameters. This
        is useful for getting the values of the parameters after training, for
        example.

        Parameters
        ----------
        raw : bool, default False
            If True, prints the raw values of the parameters, otherwise prints
            the constrained values of the parameters.

        """
        pars = self.get_parameters(raw=raw)
        for key, value in pars.items():
            print(f"{key}: {value}")

    def print_results(self):
        for key in self.results.keys():
            results_tmp = self.results[key][-1]
            results_tmp_shape = results_tmp.shape  # e.g. (4,1,1)
            results_tmp_shape_len = len(results_tmp.shape)
            if results_tmp_shape_len == 1:
                print(f"{key}: {results_tmp}")
            else:
                sum_over_shape = sum(j > 1 for j in results_tmp_shape)
                if sum_over_shape in [0, 1]:
                    print(f"{key}: {results_tmp.flatten()}")
                elif sum_over_shape == 2:
                    for i in range(results_tmp.shape[-1]):
                        print(f"{key}: {results_tmp[...,i].flatten()}")

    def plot_psd(
        self,
        freq=None,
        means=None,
        scales=None,
        weights=None,
        show=True,
        raw=False,
        log=(True, False),
        truncate_psd=True,
        logpsd=False,
        mcmc_samples=False,
        **kwargs,
    ):
        """Plot the power spectral density of the model

        Parameters
        ----------
        freq : array_like, optional
            The frequencies at which to compute the PSD, by default None. If
            None, the frequencies will be computed automatically.
        means : array_like, optional
            The means of the gaussians in the spectral mixture kernel, by
            default None. If None, the means from the model will be used.
        scales : array_like, optional
            The scales of the gaussians in the spectral mixture kernel, by
            default None. If None, the scales from the model will be used.
        weights : array_like, optional
            The weights of the gaussians in the spectral mixture kernel, by
            default None. If None, the weights from the model will be used.
        show : bool, optional
            Whether to show the plot, by default True.
        raw : bool, optional
            If True, the PSD will be computed in the space that the model was
            trained in, by default False. If False, the PSD will be computed
            in the original space of the data.
        log : tuple, optional
            A tuple of two booleans, indicating whether to plot the x-axis and
            y-axis on a log scale, respectively, by default (True, False).
        truncate_psd : float or bool, optional
            If not False, the PSD will be truncated at this value, by default
            True. This is useful for speeding up plotting when the frequency
            range is large. If logpsd is True, this value should be given in
            (natural) log space. If truncate_psd is True, the PSD will be
            truncated at 1e-6 times the maximum PSD for logpsd=False, and 1e-15
            of the maximum PSD (i.e. max(ln(psd)) - 34.5388) for logpsd=True.
        logpsd : bool, optional
            If True, the PSD will be plotted on a log scale, by default False.
            If True, truncate_psd must be given in (natural) log space.
        mcmc_samples : bool, optional
            If True, many sample PSDs will be plotted using the MCMC samples,
            by default False. This will only work if the model has been fitted
            using MCMC.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the plotting routine.

        Returns
        -------
        fig, ax : matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
            The figure and axes objects of the plot.
        """

        if freq is None:
            if self.ndim == 1:
                if raw:
                    # our step size only needs to be small enough to resolve
                    # the width of the narrowest gaussian
                    step = self.model.sci_kernel.mixture_scales.min() / 5
                    # this isn't really the correct way to do this, but it will
                    # do for now
                    diffs = (
                        self._xdata_transformed.sort().values[1:]
                        - self._xdata_transformed.sort().values[:-1]
                    )
                    mindelta = (diffs[diffs > 0]).min().item()
                    freq = torch.arange(
                        1
                        / (
                            self._xdata_transformed.max()
                            - self._xdata_transformed.min()
                        ).item(),
                        1 / (mindelta),
                        step.item(),
                    )
                else:
                    # we have to transform the step size to the original space
                    # to get the correct frequency range
                    step = 1 / self.xtransform.inverse(
                        1 / (self.model.sci_kernel.mixture_scales.min() / 5),
                        shift=False,
                    )
                    # this isn't really the correct way to do this, but it will
                    # do for now
                    diffs = (
                        self._xdata_raw.sort().values[1:]
                        - self._xdata_raw.sort().values[:-1]
                    )
                    mindelta = (diffs[diffs > 0]).min().item()

                    # we want to sample a set of frequencies that are spaced
                    # in the range covered by the gaussian mixture, but we
                    # want to sample them densely enough to resolve the
                    # narrowest gaussian so we want a minimum frequency

                    freq = torch.arange(
                        1 / (self._xdata_raw.max() - self._xdata_raw.min()).item(),
                        1 / (mindelta / 2),
                        step.item(),
                    )

            elif self.ndim == 2:
                raise NotImplementedError(
                    """Plotting PSDs in more than 1 dimension is
                                          not currently supported. Please get in touch
                                          if you need this functionality!
                """
                )
            else:
                raise NotImplementedError(
                    """Plotting PSDs in more than 2 dimensions
                                          is not currently supported. Please get in
                                          touch if you need this functionality!
                """
                )

        if mcmc_samples:
            fig, ax = self._plot_psd_mcmc(
                freq,
                means=means,
                scales=scales,
                weights=weights,
                show=show,
                raw=raw,
                log=log,
                truncate_psd=truncate_psd,
                logpsd=logpsd,
                **kwargs,
            )
            return fig, ax
        # Computing the psd for frequencies f
        psd = self.compute_psd(
            freq,
            means=means,
            scales=scales,
            weights=weights,
            raw=raw,
            log=logpsd,
            **kwargs,
        )

        if truncate_psd is True:
            if logpsd:
                freq = freq[psd > psd.max() - 34.5388]
                psd = psd[psd > psd.max() - 34.5388]
            else:
                freq = freq[psd > 1e-6 * psd.max()]
                psd = psd[psd > 1e-6 * psd.max()]
        elif truncate_psd:
            freq = freq[psd > truncate_psd]
            psd = psd[psd > truncate_psd]

        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # plotting psd
        ax.plot(freq, psd)
        if log[0]:
            ax.set_xscale("log")
        if log[1] and not logpsd:  # we don't need to double-log the Y axis (I hope!)
            ax.set_yscale("log")
        if show:
            plt.show()
        else:
            return fig, ax

    def _plot_psd_mcmc(
        self,
        freq,
        means=None,
        scales=None,
        weights=None,
        show=True,
        raw=False,
        log=(True, True),
        truncate_psd=True,
        logpsd=False,
        n_samples_to_plot=25,
        **kwargs,
    ):
        """Plot the power spectral density of the model using MCMC samples

        Parameters
        ----------
        freq : array_like
            The frequencies at which to compute the PSD
        means : array_like, optional
            The means of the gaussians in the spectral mixture kernel, by
            default None. If None, the means from the model will be used.
        scales : array_like, optional
            The scales of the gaussians in the spectral mixture kernel, by
            default None. If None, the scales from the model will be used.
        weights : array_like, optional
            The weights of the gaussians in the spectral mixture kernel, by
            default None. If None, the weights from the model will be used.
        show : bool, optional
            Whether to show the plot, by default True.
        raw : bool, optional
            If True, the PSD will be computed in the space that the model was
            trained in, by default False. If False, the PSD will be computed
            in the original space of the data.
        log : tuple, optional
            A tuple of two booleans, indicating whether to plot the x-axis and
            y-axis on a log scale, respectively, by default (True, False).
        truncate_psd : float or bool, optional
            If not False, the PSD will be truncated at this value, by default
            True. This is useful for speeding up plotting when the frequency
            range is large. If logpsd is True, this value should be given in
            (natural) log space. If truncate_psd is True, the PSD will be
            truncated at 1e-6 times the maximum PSD for logpsd=False, and 1e-15
            of the maximum PSD (i.e. max(ln(psd)) - 34.5388) for logpsd=True.
        logpsd : bool, optional
            If True, the PSD will be plotted on a log scale, by default False.
            If True, truncate_psd must be given in (natural) log space.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the plotting routine.

        Returns
        -------
        fig, ax : matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
            The figure and axes objects of the plot.
        """

        if not self.__FITTED_MCMC:
            raise RuntimeError("You must first run the MCMC sampler")
        n_samples = min(self.num_samples, n_samples_to_plot)
        if means is None:
            # this approach is slightly bugged - if more than one chain is used,
            # it will only draw samples from the first chain
            # will change this to generate random indices instead
            # at some point!
            # right now, this will end up having shape (1, chains, samples) (I thinkk)
            means = (
                torch.as_tensor(self.inference_data.posterior["raw_frequencies"].values)
                .squeeze()[:n_samples]
                .unsqueeze(0)
            )
            # print(means.shape)
            # print(freq.shape)
        if scales is None:
            scales = (
                torch.as_tensor(
                    self.inference_data.posterior["raw_frequency_scales"].values
                )
                .squeeze()[:n_samples]
                .unsqueeze(0)
            )  # .unsqueeze(-1)
        if weights is None:
            weights = (
                torch.as_tensor(
                    self.inference_data.posterior[
                        "covar_module.mixture_weights_prior"
                    ].values
                )
                .squeeze()[:n_samples]
                .unsqueeze(0)
            )  # .unsqueeze(-1)

        # computing the psd for all samples simultaneously is very expensive,
        # so we're just going to loop over them and plot them individually
        # this means we have to do things in a differnet order to the other
        # plotting routines

        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for i in range(n_samples):
            # Computing the psd for frequencies f
            psd = self.compute_psd(
                freq,
                means=means[..., i],
                scales=scales[..., i],
                weights=weights[..., i],
                raw=raw,
                log=logpsd,
                **kwargs,
            )

            if truncate_psd is True:
                mask = psd > psd.max() - 34.5388 if logpsd else psd > 1e-6 * psd.max()
            elif truncate_psd:
                mask = psd > truncate_psd
            # now we can plot it:
            ax.plot(freq[mask], psd[mask], alpha=0.2, color="b")

        # final plot formatting
        if log[0]:
            ax.set_xscale("log")
        if log[1] and not logpsd:  # we don't need to double-log the Y axis (I hope!)
            ax.set_yscale("log")
        if show:
            plt.show()
        return fig, ax

    def compute_psd(
        self,
        freq,
        means=None,
        scales=None,
        weights=None,
        raw=False,
        log=False,
        debug=False,
        **kwargs,
    ):
        """Compute the power spectral density for the model

        Parameters
        ----------
        freq : array_like or tuple(array_likes)
            The Fourier duals at which to compute the PSD
            If array_like, assumes only one dual present.
            If tuple, duals are unpacked from it.
        means : array_like, optional
            The means of the gaussians in the spectral mixture kernel, by
            default None. If None, the means from the model will be used.
        scales : array_like, optional
            The scales of the gaussians in the spectral mixture kernel, by
            default None. If None, the scales from the model will be used.
        weights : array_like, optional
            The weights of the gaussians in the spectral mixture kernel, by
            default None. If None, the weights from the model will be used.
        raw : bool, optional
            If True, the PSD will be computed in the space that the model was
            trained in, by default False. If False, the PSD will be computed
            in the original space of the data.
        **kwargs : dict, optional
            Any other keyword arguments to be passed.

        Returns
        -------
        psd : array_like
            The power spectral density of the model at the frequencies given
            by freq.
        """
        if means is None:
            means = self.model.sci_kernel.mixture_means
            # now apply the transform too!
            if self.xtransform is not None and not raw:
                # there's probably an easier way to do this than converting to
                # a period and back, but this will do for now
                means = 1 / self.xtransform.inverse(1 / means, shift=False).detach()
        if scales is None:
            scales = self.model.sci_kernel.mixture_scales
            # now apply the transform too!
            if self.xtransform is not None and not raw:
                scales = 1 / (
                    2
                    * np.pi
                    * self.xtransform.inverse(
                        1 / (2 * torch.pi * scales), shift=False
                    ).detach()
                )
        if weights is None:
            weights = self.model.sci_kernel.mixture_weights.detach()  # .numpy()

        from torch.distributions import Normal as torchnorm

        # Computing the psd for frequencies f
        if debug:
            print(freq.shape, means.shape, scales.shape, weights.shape)
        norm = torchnorm(means, scales)
        if debug:
            print(norm)
        if self.ndim > 1:
            if not isinstance(freq, tuple):
                raise ValueError(
                    "freq must be a tuple of array_likes for "
                    "multidimensional light curves!"
                )
            if len(freq) > 2:
                raise NotImplementedError(
                    "PSD for more than two duals not implemented yet"
                )
            if len(freq) != self.ndim:
                raise ValueError(
                    "freq must have the same number of duals as the number "
                    "of light curve dimensions!"
                )
            f1, f2 = freq
            norm1 = torchnorm(means[..., -2], scales[..., -2])
            norm2 = torchnorm(means[..., -1], scales[..., -1])
            psd = norm1.log_prob(f1).unsqueeze(-1) + norm2.log_prob(f2).unsqueeze(1)
            try:
                psd_tot = torch.logsumexp(
                    torch.log(weights.unsqueeze(-1).unsqueeze(-1)) + psd, dim=-3
                )
            except RuntimeError as e:
                # chunk it
                print(f"{e}. Chunking not implemented yet in compute_psd.")
        else:
            if len(freq.shape) > 1:
                raise ValueError(
                    "array-like freq must be one-dimensional for 1D light " "curves!"
                )
            f1 = torch.as_tensor(freq)
            # marginalise over Fourier dual variables
            psd1 = norm.log_prob(f1.unsqueeze(-1)).sum(dim=-1)
            # marginalise over Fourier dual variables
            psd2 = norm.log_prob(-f1.unsqueeze(-1)).sum(dim=-1)
            psd = (
                torch.log(torch.Tensor([0.5]))
                + psd1
                + torch.log(1.0 + torch.exp(psd2 - psd1))
            )
            try:
                psd_tot = torch.logsumexp(
                    torch.log(weights.unsqueeze(-1)) + psd, dim=-2
                )
            except RuntimeError:  # logsumexp tries to allocate a large array and
                # then do the summation so let's do it in a loop instead and see
                # if that avoids the problem
                psd_tot = torch.zeros_like(f1)
                for i in range(len(freq[0])):
                    psd_tot[i] = torch.logsumexp(
                        torch.log(weights) + psd[..., i], dim=-1
                    )
        if debug:
            print(psd_tot.shape)
        if not log:
            psd_tot = psd_tot.exp().cpu().detach().numpy()
        return psd_tot

    def plot(self, ylim=None, yscale="auto", show=True, mcmc_samples=False, **kwargs):
        """Plot the model and data

        Parameters
        ----------
        ylim : list, optional
            The y-limits of the plot, by default None. If None, the y-limits
            will be set automatically. For 2-D (multiwavelength) data the
            limits are determined independently for each wavelength.
        yscale : str, optional
            The y-axis scale to use. Can be ``'auto'`` (default), ``'linear'``
            or ``'log'``. When ``'auto'``, log scale is chosen for a given
            wavelength if all its flux values are positive and the ratio of
            maximum to minimum flux exceeds 100; otherwise linear scale is
            used. For 2-D data the scale is decided independently per
            wavelength. Note that when ``mcmc_samples`` is ``True``, this
            parameter is currently ignored and the y-axis scale is set by the
            MCMC plotting routine.
        show : bool, optional
            Whether to show the plot, by default True.
        mcmc_samples : bool, optional
            Whether to plot the samples from the MCMC run, by default False.
            This will only work if the MCMC sampler has been run.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the plotting routine.

        Returns
        -------
        fig : matplotlib.pyplot.Figure or list of matplotlib.pyplot.Figure
            The figure object of the plot.  For 2-D (multiwavelength) data a
            list of figures is returned, one per wavelength.
        """
        _VALID_YSCALES = ("auto", "linear", "log")
        if yscale not in _VALID_YSCALES:
            raise ValueError(
                f"yscale must be one of {_VALID_YSCALES!r}, got {yscale!r}"
            )
        if ylim is None and self.ndim == 1:
            # ylim = [-3, 3]
            y_min = float(self.ydata.min())
            y_max = float(self.ydata.max())
            y_range = y_max - y_min
            if y_range != 0.0:
                padding = 0.1 * abs(y_range)
            else:
                # If all y values are identical, pad based on their magnitude,
                # or fall back to a small absolute padding.
                base = abs(y_max) if y_max != 0.0 else 1.0
                padding = 0.1 * base
            ylim = [y_min - padding, y_max + padding]
        if mcmc_samples:
            if self.__FITTED_MCMC:
                return self._plot_mcmc(ylim=ylim, show=show, **kwargs)
            else:
                raise RuntimeError("You must first run the MCMC sampler")
        elif not self.__FITTED_MAP:
            return self._plot_data_only(ylim=ylim, yscale=yscale, show=show)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get into evaluation (predictive posterior) mode
            # self.model.eval()
            # self.likelihood.eval()

            self._eval()

            # Importing raw x and y training data from xdata and
            # ydata functions
            if self.ndim == 1:
                x_raw = self.xdata
            elif self.ndim == 2:
                x_raw = self.xdata[:, 0]
            # y_raw = self.ydata

            # creating array of 10000 test points across the range of the data
            x_fine_raw = torch.linspace(x_raw.min(), x_raw.max(), 10000)

            if self.ndim == 1:
                fig = self._plot_1d(
                    x_fine_raw, ylim=ylim, yscale=yscale, show=show, **kwargs
                )
            elif self.ndim == 2:
                fig = self._plot_2d(
                    x_fine_raw, ylim=ylim, yscale=yscale, show=show, **kwargs
                )
            else:
                raise NotImplementedError(
                    """
                Plotting models and data in more than 2 dimensions is not
                currently supported. Please get in touch if you need this
                functionality!
                """
                )
        return fig

    def _plot_mcmc(self, ylim=None, show=False, n_samples_to_plot=25, **kwargs):
        """Plot the model and data, including samples from the MCMC run

        Parameters
        ----------
        ylim : list, optional
            The y-limits of the plot, by default None. If None, the y-limits
            will be set automatically.
        show : bool, optional
            Whether to show the plot, by default True.
        n_samples_to_plot : int, optional
            The number of samples to plot, by default 25.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the plotting routine.

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure object of the plot.
        """
        # Get into evaluation (predictive posterior) mode
        self._eval()

        if self.ndim > 1:
            raise NotImplementedError(
                """
            Plotting models and data in more than 1 dimension is not
            currently supported. Please get in touch if you need this
            functionality!
            """
            )
        # Importing raw x and y training data from xdata and
        # ydata functions
        x_raw = self.xdata
        # y_raw = self.ydata

        # creating array of 10000 test points across the range of the data
        x_fine_raw = torch.linspace(x_raw.min(), x_raw.max(), 10000).unsqueeze(-1)

        # transforming the x_fine_raw data to the space that the GP was
        # trained in (so it can predict)
        if self.xtransform is None:
            self.x_fine_transformed = x_fine_raw
        elif isinstance(self.xtransform, Transformer):
            self.x_fine_transformed = self.xtransform.transform(
                x_fine_raw.to(self.xtransform.min.device)
            )

        self.expanded_test_x = self.x_fine_transformed.unsqueeze(0).repeat(
            self.num_samples, 1, 1
        )  # .unsqueeze(0)
        output = self.model(self.expanded_test_x)
        with torch.no_grad():
            f, ax = plt.subplots(1, 1, figsize=(8, 6))
            for i in range(min(n_samples_to_plot, self.num_samples)):
                # Plot predictive samples as colored lines
                ax.plot(
                    x_fine_raw.cpu().numpy(),
                    output[i].sample().cpu().numpy(),
                    "b",
                    alpha=0.2,
                )

            # Plot training data as black stars (on top of model predictions)
            ax.plot(self.xdata.cpu().numpy(), self.ydata.cpu().numpy(), "k*")

            ax.legend(["Observed Data", "Sample means"])
            if ylim is not None:
                ax.set_ylim(ylim)
            if show:
                plt.show()
        return f

    @staticmethod
    def _yscale_and_ylim(y_vals, yscale, ylim):
        """Resolve the y-axis scale and limits for a single band.

        Parameters
        ----------
        y_vals : array-like
            Flux values for the band (must support ``min()``/``max()``).
        yscale : str
            One of ``'auto'``, ``'linear'``, or ``'log'``.  Values outside
            this set are passed through to ``ax.set_yscale()`` unchanged;
            callers should validate beforehand (``plot()`` does this).
        ylim : list or None
            Caller-supplied y-axis limits.  ``None`` means auto-compute.

        Returns
        -------
        scale : str
            Either ``'linear'`` or ``'log'``.
        lim : list or None
            Two-element list ``[y_lo, y_hi]``, or ``None`` when the limits
            should be left to matplotlib (e.g. log scale with non-positive
            data, or an explicit ``ylim`` that is incompatible with log scale).
        """
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))

        # Resolve scale
        if yscale == "auto":
            scale = (
                "log" if y_min > 0 and y_max / y_min > 100 else "linear"
            )
        else:
            scale = yscale

        # Resolve limits
        if ylim is None:
            if scale == "log" and y_min > 0:
                log_min = np.log10(y_min)
                log_max = np.log10(y_max)
                log_range = log_max - log_min
                padding = 0.1 * abs(log_range) if log_range != 0.0 else 0.1
                lim = [10 ** (log_min - padding), 10 ** (log_max + padding)]
            elif scale != "log":
                y_range = y_max - y_min
                if y_range != 0.0:
                    padding = 0.1 * abs(y_range)
                else:
                    base = abs(y_max) if y_max != 0.0 else 1.0
                    padding = 0.1 * base
                lim = [y_min - padding, y_max + padding]
            else:
                # Log scale forced/selected but data contains non-positive
                # values: let matplotlib choose an appropriate range.
                lim = None
        else:
            # Caller-supplied limits: skip setting them when they are
            # incompatible with a log axis (non-positive lower bound).
            lim = None if scale == "log" and ylim[0] <= 0 else ylim

        return scale, lim

    def _plot_data_only(self, ylim=None, yscale="auto", show=False):
        """Plot only the data, without any GP predictions.

        Used when the GP has not yet been fitted.

        For 1-D data, a single figure is returned.  For 2-D (multiband) data,
        a separate figure is created for each wavelength (matching the layout
        of :meth:`_plot_2d` used after fitting), and a list of figures is
        returned.
        """
        if self.ndim == 2:
            unique_values_axis2 = torch.unique(self.xdata[:, 1])
            figs = []
            for val in unique_values_axis2:
                mask = self.xdata[:, 1] == val
                x_plot = self.xdata[mask, 0].cpu().numpy()
                y_data_for_val = self.ydata[mask]
                y_plot = y_data_for_val.cpu().numpy()

                fig = plt.figure()
                ax = fig.add_subplot(111)

                if hasattr(self, "yerr") and self.yerr is not None:
                    ax.errorbar(
                        x_plot,
                        y_plot,
                        yerr=self.yerr[mask].cpu().numpy(),
                        fmt="k*",
                        label="Observed",
                    )
                else:
                    ax.plot(x_plot, y_plot, "k*", label="Observed")

                ax.set_ylabel("y")
                ax.set_xlabel("x")
                ax.set_title(f"y vs x for {val}")

                current_yscale, current_ylim = self._yscale_and_ylim(
                    y_plot, yscale, ylim
                )
                ax.set_yscale(current_yscale)
                if current_ylim is not None:
                    ax.set_ylim(current_ylim)
                ax.legend()

                if show:
                    plt.show()
                figs.append(fig)
            return figs

        # 1-D case
        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        x_plot = self.xdata.cpu().numpy()
        y_plot = self.ydata.cpu().numpy()
        if hasattr(self, "yerr") and self.yerr is not None:
            ax.errorbar(
                x_plot, y_plot, yerr=self.yerr.cpu().numpy(), fmt="k*", label="Observed"
            )
        else:
            ax.plot(x_plot, y_plot, "k*", label="Observed")
        current_yscale, current_ylim = self._yscale_and_ylim(y_plot, yscale, ylim)
        ax.set_yscale(current_yscale)
        if current_ylim is not None:
            ax.set_ylim(current_ylim)
        ax.legend()
        if show:
            plt.show()
        return f

    def _plot_1d(
        self, x_fine_raw, ylim=None, yscale="auto", show=False, save=True, **kwargs
    ):
        # transforming the x_fine_raw data to the space that the GP was
        # trained in (so it can predict)
        if self.xtransform is None:
            x_fine_transformed = x_fine_raw
        elif isinstance(self.xtransform, Transformer):
            x_fine_transformed = self.xtransform.transform(
                x_fine_raw.to(self.xtransform.min.device)
            )

        # Make predictions
        observed_pred = self.likelihood(self.model(x_fine_transformed))

        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()

        # Plot predictive GP mean as blue line
        ax.plot(
            x_fine_raw.cpu().numpy(),
            observed_pred.mean.cpu().numpy(),
            "b",
            label="Mean",
        )

        # Shade between the lower and upper confidence bounds
        ax.fill_between(
            x_fine_raw.cpu().numpy(),
            lower.cpu().numpy(),
            upper.cpu().numpy(),
            alpha=0.5,
            label="Confidence",
        )

        # Plot training data as black stars (on top of model predictions)
        if self.yerr is not None:
            ax.errorbar(
                self.xdata.cpu().numpy(),
                self.ydata.cpu().numpy(),
                yerr=self.yerr.cpu().numpy(),
                fmt="ko",
                label="Observed",
            )
        else:
            ax.plot(
                self.xdata.cpu().numpy(),
                self.ydata.cpu().numpy(),
                "ko",
                label="Observed",
            )

        # Determine y-axis scale and limits using the shared helper
        current_yscale, current_ylim = self._yscale_and_ylim(
            self.ydata.cpu().numpy(), yscale, ylim
        )
        ax.set_yscale(current_yscale)
        if current_ylim is not None:
            ax.set_ylim(current_ylim)
        ax.legend()
        if save:
            plt.savefig(f"{self.name}_fit.png")
        if show:
            plt.show()
        return f

    def _plot_2d(
        self, x_fine_raw, ylim=None, yscale="auto", show=False, save=True, **kwargs
    ):
        if self.xtransform is None:
            x_fine_transformed = x_fine_raw
        elif isinstance(self.xtransform, Transformer):
            x_fine_transformed = self.xtransform.transform(
                x_fine_raw.to(self.xtransform.min.device),
                apply_to=(0, 0),
            )
        unique_values_axis2 = torch.unique(self.xdata[:, 1])
        figs = []
        for val in unique_values_axis2:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            vals = torch.ones_like(x_fine_transformed) * val
            x_fine_tmp = torch.cat((x_fine_transformed[:, None], vals[:, None]), dim=1)

            observed_pred = self.likelihood(self.model(x_fine_tmp))
            ax.plot(x_fine_raw.cpu().numpy(), observed_pred.mean.cpu().numpy(),
                    "b", label = "Mean")

            lower, upper = observed_pred.confidence_region()
            ax.fill_between(
                x_fine_raw.cpu().numpy(),
                lower.cpu().numpy(),
                upper.cpu().numpy(),
                alpha=0.5,
                label = "Confidence"
            )

            # Plot training data as black stars (on top of model predictions)
            y_data_for_val = self.ydata[self.xdata[:, 1] == val]
            if self.yerr is not None:
                ax.errorbar(
                    self.xdata[self.xdata[:, 1] == val, 0],
                    y_data_for_val,
                    yerr = self.yerr,
                    fmt = "ko",
                    label = "Observed Data"
                )
            else:
                ax.plot(
                    self.xdata[self.xdata[:, 1] == val, 0],
                    y_data_for_val,
                    "ko",
                    label = "Observed Data"
                )
            ax.legend()

            ax.set_ylabel("y")
            ax.set_xlabel("x")
            ax.set_title(f"y vs x for {val}")

            # Determine y-axis scale and limits for this wavelength
            # independently, using the shared helper.
            current_yscale, current_ylim = self._yscale_and_ylim(
                y_data_for_val.cpu().numpy(), yscale, ylim
            )
            ax.set_yscale(current_yscale)
            if current_ylim is not None:
                ax.set_ylim(current_ylim)

            if save:
                plt.savefig(f"{self.name}_{val}_fit.png")

            if show:
                plt.show()
            figs.append(fig)
        return figs

    def _plot_nd(self):
        raise NotImplementedError(
            """
        Plotting models and data in more than 2 dimensions is not currently supported.
        Please get in touch if you need this functionality!
        """
        )

    def plot_results(self):
        for key, value in self.results.item():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            with contextlib.suppress(ValueError):
                ax.plot(value, "-")
            ax.set_ylabel(key)
            ax.set_xlabel("Iteration")

            if "means" in key:
                self.value_inversed = self.xtransform.inverse(value)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(torch.Tensor(self.value_reversed), "-")
                ax.set_ylabel(key)
                ax.set_xlabel("Iteration")
        plt.show()

    def to_table(self):
        """Create an astropy table with the results.

        Parameters
        ----------
        none

        Returns
        -------
        tab_results : astropy.table.Table
            Astropy table with the results.
        """
        from astropy.table import Table

        t = Table()
        t["x"] = [np.asarray(self.xdata.cpu())]
        t["y"] = [np.asarray(self.ydata.cpu())]
        if hasattr(self, "yerr"):
            t["yerr"] = [np.asarray(self.yerr.cpu())]
        if self.__FITTED_MCMC or self.__FITTED_MAP:
            # These outputs can only be produced if a fit has been run.
            periods, weights, scales = self.get_periods()
            t["period"] = [np.asarray(periods)]
            try:
                t["weights"] = [np.asarray(weights)]
            except RuntimeError:
                t["weights"] = [torch.as_tensor(weights).cpu().detach().numpy()]
            try:
                t["scales"] = [np.asarray(scales)]
            except RuntimeError:
                t["scales"] = [torch.as_tensor(scales).cpu().detach().numpy()]
            for key, value in self.results.items():
                try:
                    t[key] = [np.asarray(value)]
                except RuntimeError:
                    t[key] = [torch.as_tensor(value).cpu().detach().numpy()]
            if self.__FITTED_MAP:
                # Loss isn't relevant for MCMC, I think
                t["loss"] = [np.asarray(self.results["loss"])]
            # Now we want the model predictions for the input times:
            if self.__FITTED_MAP:
                self._eval()
                with torch.no_grad():
                    observed_pred = self.likelihood(self.model(self._xdata_transformed))
                    t["y_pred_mean_obs"] = [np.asarray(observed_pred.mean.cpu())]
                    t["y_pred_lower_obs"] = [
                        np.asarray(observed_pred.confidence_region()[0].cpu())
                    ]
                    t["y_pred_upper_obs"] = [
                        np.asarray(observed_pred.confidence_region()[1].cpu())
                    ]

                    if self.ndim == 1:
                        x_raw = self.xdata
                    elif self.ndim == 2:
                        x_raw = self.xdata[:, 0]
                    # y_raw = self.ydata

                    # creating array of 10000 test points across the range of the data
                    x_fine_raw = torch.linspace(x_raw.min(), x_raw.max(), 10000)
                    if self.xtransform is None:
                        x_fine_transformed = x_fine_raw
                    elif isinstance(self.xtransform, Transformer):
                        x_fine_transformed = self.xtransform.transform(
                            x_fine_raw.to(self.xtransform.min.device)
                        )

                    # Make predictions
                    observed_pred = self.likelihood(self.model(x_fine_transformed))
                    t["x_fine"] = [np.asarray(x_fine_raw.cpu())]
                    t["y_pred_mean"] = [np.asarray(observed_pred.mean.cpu())]
                    t["y_pred_lower"] = [
                        np.asarray(observed_pred.confidence_region()[0].cpu())
                    ]
                    t["y_pred_upper"] = [
                        np.asarray(observed_pred.confidence_region()[1].cpu())
                    ]
            elif self.__FITTED_MCMC:
                raise NotImplementedError("MCMC predictions not yet implemented")
                # with torch.no_grad():

        return t

    def write_votable(self, filename):
        """Write the results to a votable file.

        Parameters
        ----------
        filename : str
            The name of the file to write to.
        """
        t = self.to_table()
        t.write(filename, format="votable", overwrite=True)
