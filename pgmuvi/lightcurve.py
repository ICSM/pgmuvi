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
        Candidate column names used for auto-detecting the wavelength or band
        column, checked case-insensitively in order.  When such a column is
        found and contains more than one unique value, the data are loaded as
        a 2-D lightcurve whose ``xdata`` has shape ``(N, 2)`` with the time
        values in column 0 and the wavelength/band values in column 1.
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
        "band",
        "filter",
        "freq",
        "frequency",
        "channel",
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
            When the CSV contains a wavelength or band column with more than
            one unique value, the resulting ``xdata`` has shape ``(N, 2)``
            where column 0 holds the time values and column 1 holds the
            wavelength/band values.  The ``ydata`` (and optional ``yerr``)
            remain 1-D tensors of shape ``(N,)``.

            The wavelength/band column is selected in one of three ways:

            1. *Explicit ``xcol`` list*: pass ``xcol`` as a list of two
               column names, e.g. ``xcol=["time", "band"]``.  The first
               element is the time column and the second is the
               wavelength/band column.  All subsequent x-axis columns are
               stacked in the order given.
            2. *Explicit ``wavelcol``*: pass the column name as a separate
               ``wavelcol`` keyword argument.
            3. *Auto-detection*: if neither an iterable ``xcol`` nor a
               ``wavelcol`` is supplied, the method searches for a column
               whose name matches one of the entries in
               :attr:`_WAVELENGTH_COLUMN_NAMES`.  If such a column is found
               *and* it contains more than one unique value, a 2-D lightcurve
               is returned automatically.

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
        data = np.genfromtxt(
            filepath, delimiter=",", names=True, dtype=float, encoding=None
        )
        columns = list(data.dtype.names)

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
        # Resolve the x (time + optional band) columns
        # ------------------------------------------------------------------
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
            # Resolve wavelength/band column (explicit or auto-detected)
            if wavelcol is None:
                wavelcol = cls._find_column(columns, cls._WAVELENGTH_COLUMN_NAMES)
            elif wavelcol not in columns:
                raise ValueError(
                    f"Column '{wavelcol}' not found in CSV. "
                    f"Available columns: {columns}"
                )
            xcol_names = [xcol] + ([wavelcol] if wavelcol is not None else [])

        # ------------------------------------------------------------------
        # Build NaN mask from ALL relevant columns before creating tensors
        # ------------------------------------------------------------------
        relevant_cols = xcol_names + [ycol] + ([yerrcol] if yerrcol else [])
        valid_mask = np.ones(len(data), dtype=bool)
        for col in relevant_cols:
            valid_mask &= ~np.isnan(data[col])

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
        # Build tensors from clean data
        # ------------------------------------------------------------------
        if isinstance(xcol, list):
            x_tensors = [
                torch.as_tensor(clean[col], dtype=torch.float32) for col in xcol
            ]
            x = torch.stack(x_tensors, dim=1) if len(x_tensors) > 1 else x_tensors[0]
        else:
            time_tensor = torch.as_tensor(clean[xcol], dtype=torch.float32)
            if wavelcol is not None:
                wave_tensor = torch.as_tensor(clean[wavelcol], dtype=torch.float32)
                if wave_tensor.unique().numel() > 1:
                    # Multiple wavelengths/bands → 2-D lightcurve
                    x = torch.stack([time_tensor, wave_tensor], dim=1)
                else:
                    # Single wavelength → treat as 1-D
                    x = time_tensor
            else:
                x = time_tensor

        y = torch.as_tensor(clean[ycol], dtype=torch.float32)
        yerr = torch.as_tensor(clean[yerrcol], dtype=torch.float32) if yerrcol else None


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
        max_samples: int | None = 1000,
        max_samples_per_band: int | None = None,
        subsample_seed: int | None = None,
        check_sampling: bool = False,
        sampling_kwargs: dict | None = None,
        check_variability: bool = False,
        variability_kwargs: dict | None = None,
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
        max_samples_per_band : int or None, optional
            Maximum number of observations to retain per band for 2-D
            (multiband) lightcurves.  Each band is checked independently:
            only bands that exceed `max_samples_per_band` are subsampled;
            bands already at or below the limit are left untouched.  For
            1-D lightcurves, this parameter has no effect.  Set to ``None``
            (default) to fall back to `max_samples` as the per-band cap.
            A :class:`UserWarning` is issued whenever subsampling occurs
            (see :func:`~pgmuvi.preprocess.subsample_lightcurve`).
        max_samples : int or None, optional
            Maximum number of observations to retain.  For 1-D lightcurves,
            when the total number of points exceeds `max_samples`, a
            gap-preserving random subsample of `max_samples` points is
            drawn and stored permanently.  For 2-D lightcurves, `max_samples`
            is used as the per-band cap when `max_samples_per_band` is
            ``None``; additionally, a :class:`UserWarning` is issued if the
            total point count after per-band subsampling still exceeds
            `max_samples`.  Default is ``1000``.  Set to ``None`` to
            disable all automatic subsampling.
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
        # For 2D (multiband) light curves, each band is subsampled
        # independently so that bands already below the limit are untouched.
        # ------------------------------------------------------------------
        # Effective per-band cap: explicit max_samples_per_band overrides;
        # falls back to max_samples when max_samples_per_band is not set.
        _eff_per_band = (
            max_samples_per_band
            if max_samples_per_band is not None
            else max_samples
        )
        _run_2d = self.ndim > 1 and _eff_per_band is not None
        _run_1d = self.ndim == 1 and max_samples is not None
        if _run_1d or _run_2d:
            from pgmuvi.preprocess import subsample_lightcurve

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

            mgf = (sampling_kwargs or {}).get("max_gap_fraction", 0.3)
            _buffer_names = (
                "_xdata_raw",
                "_xdata_transformed",
                "_ydata_raw",
                "_ydata_transformed",
                "_yerr_raw",
                "_yerr_transformed",
            )

            if _run_2d:
                # 2D (multiband): subsample each band independently so that
                # bands with fewer than _eff_per_band points are left
                # untouched.
                xdata_np = self._xdata_raw.detach().cpu().numpy()
                # `self.band` is an optional attribute that, when present,
                # stores categorical band labels (e.g. strings) for each
                # observation.  When it is absent, the second column of
                # xdata holds the numeric wavelength/band identifier — the
                # standard encoding for 2-D lightcurves.
                if hasattr(self, "band") and self.band is not None:
                    band_ids = np.asarray(self.band)
                else:
                    band_ids = xdata_np[:, 1]
                unique_bands = np.unique(band_ids)
                global_keep = []
                subsampled_bands = []
                for bval in unique_bands:
                    band_mask = np.where(band_ids == bval)[0]
                    n_band = len(band_mask)
                    if n_band > _eff_per_band:
                        t_band = xdata_np[band_mask, 0]
                        local_idx = subsample_lightcurve(
                            t_band,
                            max_samples=_eff_per_band,
                            max_gap_fraction=mgf,
                            random_seed=subsample_seed,
                        )
                        global_keep.append(band_mask[local_idx])
                        subsampled_bands.append(bval)
                    else:
                        global_keep.append(band_mask)
                if subsampled_bands:
                    _param_name = (
                        "max_samples_per_band"
                        if max_samples_per_band is not None
                        else "max_samples"
                    )
                    _band_str = ", ".join(
                        f"\u03bb={b}" for b in subsampled_bands
                    )
                    _struct_lines = "\n".join(
                        f"    \u03bb={bval}: {len(keep)} points"
                        for bval, keep in zip(
                            unique_bands, global_keep, strict=True
                        )
                    )
                    _msg = (
                        f"The following bands exceed "
                        f"{_param_name}={_eff_per_band}"
                        f" and were randomly subsampled: {_band_str}. "
                        f"Set {_param_name}=None to disable subsampling.\n"
                        "The subsampled 2D light curve has the following "
                        f"structure:\n{_struct_lines}"
                    )
                    warnings.warn(_msg, UserWarning, stacklevel=2)
                    idx = np.concatenate(global_keep)
                    # Sort by time column to preserve temporal ordering.
                    idx = idx[np.argsort(xdata_np[idx, 0], kind="stable")]
                    idx_t = torch.as_tensor(
                        idx,
                        dtype=torch.long,
                        device=self._xdata_raw.device,
                    )
                    for bname in _buffer_names:
                        if (
                            hasattr(self, bname)
                            and getattr(self, bname) is not None
                        ):
                            self.register_buffer(
                                bname, getattr(self, bname)[idx_t]
                            )
                if (
                    max_samples is not None
                    and self._xdata_raw.shape[0] > max_samples
                ):
                    _msg = (
                        f"Lightcurve has {self._xdata_raw.shape[0]} points,"
                        f" which exceeds max_samples={max_samples}. "
                        "Execution may be slow. Consider reducing "
                        "max_samples_per_band to reduce the total size of "
                        "the lightcurve."
                    )
                    warnings.warn(_msg, UserWarning, stacklevel=2)
            else:
                # 1D light curve: subsample the whole array if it exceeds the
                # limit.
                n_total = self._xdata_raw.shape[0]
                if n_total > max_samples:
                    t_np = self._xdata_raw.detach().cpu().numpy()
                    idx = subsample_lightcurve(
                        t_np,
                        max_samples=max_samples,
                        max_gap_fraction=mgf,
                        random_seed=subsample_seed,
                    )
                    warnings.warn(
                        f"Lightcurve has {n_total} points, which exceeds "
                        f"max_samples={max_samples}. Retaining a random "
                        f"subsample of {len(idx)} points. "
                        "Set max_samples=None to disable subsampling.",
                        UserWarning,
                        stacklevel=2,
                    )
                    idx_t = torch.as_tensor(
                        idx,
                        dtype=torch.long,
                        device=self._xdata_raw.device,
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
        cls, tab, file_format="votable", xcol="x", ycol="y", yerrcol="yerr", **kwargs
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

        x, y, yerr = cls._drop_nonfinite_rows(x, y, yerr)

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
                # MLS failed for any reason; fall back gracefully but warn the user.

                if num_mixtures is None:
                    num_mixtures = 4
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
                    print(step, mindelta, 1 / (mindelta / 2))

                    # we want to sample a set of frequencies that are spaced
                    # in the range covered by the gaussian mixture, but we
                    # want to sample them densely enough to resolve the
                    # narrowest gaussian so we want a minimum frequency

                    freq = torch.arange(
                        1 / (self._xdata_raw.max() - self._xdata_raw.min()).item(),
                        1 / (mindelta / 2),
                        step.item(),
                    )
                    print(freq.shape)

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
        print(self.x_fine_transformed.shape)
        print(self.expanded_test_x.shape)
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
