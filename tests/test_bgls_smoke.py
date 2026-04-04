
import math
import torch

from pgmuvi.bgls import fit_bgls


def test_single_band_smoke():
    torch.manual_seed(0)
    t = torch.sort(100 * torch.rand(200))[0]
    f0 = 0.123
    y = 1.3 * torch.sin(2 * math.pi * f0 * t + 0.4) + 0.2 * torch.randn(200)
    dy = 0.2 * torch.ones(200)
    res = fit_bgls(
        t,
        y,
        dy,
        grid_kwargs={"minimum_frequency": 0.01, "maximum_frequency": 1.0, "n_frequency": 2000},
    )
    assert abs(res.peak_frequency - f0) < 0.01


def test_multiband_smoke():
    torch.manual_seed(1)
    rows = []
    vals = []
    errs = []
    f0 = 0.071
    for i, band in enumerate([500.0, 600.0, 700.0]):
        n = 80 + 10 * i
        t = torch.sort(200 * torch.rand(n))[0]
        amp = 1.0 + 0.4 * i
        phase = 0.2 * i
        y = amp * torch.sin(2 * math.pi * f0 * t + phase) + 0.3 * torch.randn(n)
        dy = 0.3 * torch.ones(n)
        rows.append(torch.column_stack([t, torch.full_like(t, band)]))
        vals.append(y)
        errs.append(dy)
    x = torch.cat(rows)
    y = torch.cat(vals)
    dy = torch.cat(errs)
    res = fit_bgls(
        x,
        y,
        dy,
        grid_kwargs={"minimum_frequency": 0.01, "maximum_frequency": 0.5, "n_frequency": 1500},
    )
    assert abs(res.peak_frequency - f0) < 0.01
