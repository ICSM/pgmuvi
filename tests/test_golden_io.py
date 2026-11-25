"""Round-trip tests for Lightcurve input/output (roadmap 1.4).

Current NaN behaviour, pinned here and not to be changed in this PR:

* The ``Lightcurve`` constructor drops rows with a non-finite x, y or yerr,
  so a light curve built with NaN rows never holds them.
* ``from_csv`` and ``from_table`` drop non-finite rows from the file or table
  and emit a ``UserWarning``; the band labels are filtered with the same mask.
* ``from_csv`` and ``from_table`` build ``torch.Tensor`` (float32) data, so
  arrays come back in float32 even when the source light curve was float64.
"""

import numpy as np
import pytest
import torch

from pgmuvi.datasets import get_path
from pgmuvi.lightcurve import Lightcurve

N = 12


def _make_1d():
    t = np.linspace(0.0, 11.0, N)
    y = np.sin(t)
    yerr = np.linspace(0.05, 0.15, N)
    y[3] = np.nan
    return Lightcurve(torch.tensor(t), torch.tensor(y), torch.tensor(yerr))


def _make_2d():
    t = np.linspace(0.0, 11.0, N)
    x = np.c_[np.r_[t, t], np.r_[np.full(N, 0.5), np.full(N, 1.5)]]
    y = np.r_[np.sin(t), np.cos(t)]
    yerr = np.full(2 * N, 0.1)
    y[5] = np.nan
    band = np.array(["g"] * N + ["r"] * N)
    return Lightcurve(
        torch.tensor(x), torch.tensor(y), torch.tensor(yerr), band=band
    )


def _check(lc, back):
    assert back._xdata_raw.shape == lc._xdata_raw.shape
    np.testing.assert_allclose(
        back._xdata_raw.numpy(), lc._xdata_raw.numpy(), rtol=1e-6
    )
    np.testing.assert_allclose(
        back._ydata_raw.numpy(), lc._ydata_raw.numpy(), rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        back._yerr_raw.numpy(), lc._yerr_raw.numpy(), rtol=1e-6
    )


def test_constructor_drops_nan_rows():
    assert len(_make_1d()._xdata_raw) == N - 1
    lc = _make_2d()
    assert len(lc._xdata_raw) == 2 * N - 1
    assert len(lc.band) == 2 * N - 1


@pytest.mark.parametrize("maker", [_make_1d, _make_2d], ids=["1d", "2d"])
def test_csv_round_trip(maker, tmp_path):
    lc = maker()
    path = tmp_path / "lc.csv"
    lc.to_csv(path)
    back = Lightcurve.from_csv(path)
    _check(lc, back)
    if lc.band is not None:
        assert list(back.band) == list(lc.band)


@pytest.mark.parametrize("maker", [_make_1d, _make_2d], ids=["1d", "2d"])
def test_table_round_trip(maker):
    lc = maker()
    back = Lightcurve.from_table(lc.to_table())
    _check(lc, back)


@pytest.mark.parametrize(
    "maker",
    [
        _make_1d,
        pytest.param(
            _make_2d,
            marks=pytest.mark.xfail(
                strict=True,
                reason="VOTable stores the (N, 2) cell as (2, N) and from_table "
                "does not undo it: 2D x comes back scrambled. Fix is out of "
                "scope for 1.4; remove this mark when it is fixed.",
            ),
        ),
    ],
    ids=["1d", "2d"],
)
def test_votable_round_trip(maker, tmp_path):
    lc = maker()
    path = tmp_path / "lc.xml"
    lc.write_votable(path)
    back = Lightcurve.from_table(path, file_format="votable")
    _check(lc, back)


def test_csv_with_nan_rows_dropped_with_warning(tmp_path):
    path = tmp_path / "nan.csv"
    path.write_text(
        "time,wavelength,flux,flux_error,band\n"
        "0.0,0.5,1.0,0.1,g\n"
        "1.0,0.5,nan,0.1,g\n"
        "2.0,1.5,3.0,0.1,r\n"
        "3.0,1.5,4.0,0.1,r\n"
    )
    with pytest.warns(UserWarning):
        lc = Lightcurve.from_csv(path)
    assert lc._xdata_raw.shape == (3, 2)
    assert list(lc.band) == ["g", "r", "r"]


def test_table_with_nan_rows_dropped_with_warning():
    from astropy.table import Table

    tab = Table(
        {
            "x": [[0.0, 0.5], [1.0, 0.5], [2.0, 1.5], [3.0, 1.5]],
            "y": [1.0, 2.0, np.nan, 4.0],
            "yerr": [0.1, 0.1, 0.1, 0.1],
            "band": ["g", "g", "r", "r"],
        }
    )
    with pytest.warns(UserWarning):
        lc = Lightcurve.from_table(tab)
    assert lc._xdata_raw.shape == (3, 2)
    assert list(lc.band) == ["g", "g", "r"]


def test_alfori_from_csv_snapshot(golden):
    path = get_path("AlfOriAAVSO_Vband.csv")
    # The default load subsamples at random to 1000 points (DEC-9, item 2.6),
    # so only the count is pinned for it; values use max_samples=None.
    with pytest.warns(UserWarning):
        assert len(Lightcurve.from_csv(path)._xdata_raw) == 1000
    lc = Lightcurve.from_csv(path, max_samples=None)
    x, y = lc._xdata_raw, lc._ydata_raw
    golden(
        "io_alfori",
        {
            "n": len(x),
            "x_first3": x[:3],
            "x_last3": x[-3:],
            "y_first3": y[:3],
            "y_last3": y[-3:],
            "bands": sorted(set(map(str, lc.band))) if lc.band is not None else None,
            "has_yerr": hasattr(lc, "_yerr_raw"),
        },
    )
