"""Toy test for pgmuvi issue #108.

Multiband LPV with a large amplitude dynamic range and an optional phase lag.
Usage: python amp_test3.py <seed> <lag_cycles>
"""

import math
import sys
import torch
import gpytorch as gp
import numpy as np
from gpytorch.kernels import (
    Kernel,
    RBFKernel,
    ScaleKernel,
    IndexKernel,
    SpectralMixtureKernel as SMK,
)

torch.manual_seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
torch.set_default_dtype(torch.float64)
LAG = (
    float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
)  # phase lag (cycles) in reddest band
P = 400.0
T = 2000.0
lams = torch.tensor([0.55, 1.25, 2.2, 3.4, 4.6])  # micron
amps = torch.tensor([3.0, 1.0, 0.4, 0.15, 0.05])  # flux amplitude (dynamic range 60)
lags = (
    torch.tensor([0.0, 0.02, 0.05, 0.1, 0.15]) * LAG / 0.15
    if LAG > 0
    else torch.zeros(5)
)
nb = torch.tensor([60, 25, 25, 15, 40])


def truth(t, b):
    ph = 2 * math.pi * (t / P - lags[b])
    return amps[b] * (torch.cos(ph) + 0.3 * torch.cos(2 * ph + 0.7)) + 2.0 * amps[b]


X, Y, E = [], [], []
for b in range(5):
    t = torch.sort(torch.rand(nb[b]) * T)[0]
    if b == 3:
        t = t[(t < 700) | (t > 1300)]  # gap in a faint band
    e = 0.03 * amps[b] * torch.ones_like(t)
    X.append(
        torch.stack(
            [t / T, torch.full_like(t, math.log(lams[b])), torch.full_like(t, b)], -1
        )
    )
    Y.append(truth(t, b) + e * torch.randn_like(t))
    E.append(e)
X = torch.cat(X)
Y = torch.cat(Y)
E = torch.cat(E)
ymu, ysd = Y.mean(), Y.std()
Yn = (Y - ymu) / ysd
En = E / ysd
f0 = T / P  # frequency in transformed time units
band_logsd = torch.stack([Yn[X[:, 2] == b].std().log() for b in range(5)])
band_mean = torch.stack([Yn[X[:, 2] == b].mean() for b in range(5)])


def sm_time(nmix=2):
    k = SMK(num_mixtures=nmix, ard_num_dims=1, active_dims=[0])
    k.mixture_means = torch.tensor([f0, 2 * f0])[:nmix].view(nmix, 1, 1)
    k.mixture_scales = torch.full((nmix, 1, 1), 0.02)
    k.mixture_weights = torch.ones(nmix) / nmix if nmix > 1 else torch.tensor([0.01])
    return k


class AmpMod(Kernel):
    """k = a(b) a(b') * base;  a = exp(per-band log-amplitude) (piecewise)."""

    def __init__(self, base, nbands, band_dim=2):
        super().__init__()
        self.base = base
        self.band_dim = band_dim
        self.register_parameter("log_amp", torch.nn.Parameter(band_logsd.clone()))

    def forward(self, x1, x2, diag=False, **kw):
        a1 = self.log_amp[x1[..., self.band_dim].long()].exp()
        a2 = self.log_amp[x2[..., self.band_dim].long()].exp()
        K = self.base(x1, x2, diag=diag)
        if diag:
            return a1 * a2 * K
        return K.to_dense() * a1.unsqueeze(-1) * a2.unsqueeze(-2)


class ChromaticSM(Kernel):
    """Phase+amplitude-modulated SM (CSM/MOSM-like, per band):
    k = sum_q a_qb a_qb' exp(-2pi^2 tau^2 v_q) cos(2pi mu_q tau + phi_qb - phi_qb')"""

    def __init__(self, nmix, nbands):
        super().__init__()
        self.register_parameter(
            "log_amp",
            torch.nn.Parameter(
                torch.stack([band_logsd, band_logsd - 1.2]).clone()[:nmix]
            ),
        )
        self.register_parameter("phi", torch.nn.Parameter(torch.zeros(nmix, nbands)))
        self.register_parameter(
            "raw_mu", torch.nn.Parameter(torch.tensor([f0, 2 * f0])[:nmix].log())
        )
        self.register_parameter(
            "raw_sd", torch.nn.Parameter(torch.full((nmix,), math.log(0.02)))
        )

    def forward(self, x1, x2, diag=False, **kw):
        b1, b2 = x1[..., 2].long(), x2[..., 2].long()
        t1, t2 = x1[..., 0], x2[..., 0]
        tau = (t1 - t2) if diag else (t1.unsqueeze(-1) - t2.unsqueeze(-2))
        out = 0
        for q in range(self.log_amp.shape[0]):
            a1, a2 = self.log_amp[q, b1].exp(), self.log_amp[q, b2].exp()
            p1, p2 = self.phi[q, b1], self.phi[q, b2]
            if not diag:
                a1, p1, a2, p2 = a1[:, None], p1[:, None], a2[None], p2[None]
            mu, sd = self.raw_mu[q].exp(), self.raw_sd[q].exp()
            out = out + a1 * a2 * torch.exp(
                -2 * math.pi**2 * tau**2 * sd**2
            ) * torch.cos(2 * math.pi * mu * tau + p1 - p2)
        return out


class M(gp.models.ExactGP):
    def __init__(s, x, y, lik, k, mean):
        super().__init__(x, y, lik)
        s.k = k
        s.m = mean

    def forward(s, x):
        return gp.distributions.MultivariateNormal(s.m(x), s.k(x))


class BandMean(gp.means.Mean):
    def __init__(s, nb):
        super().__init__()
        s.c = torch.nn.Parameter(band_mean.clone())

    def forward(s, x):
        return s.c[x[..., 2].long()]


def build(name):
    if name == "separable_rbf":  # current pgmuvi WavelengthDependent-style
        wl = ScaleKernel(RBFKernel(active_dims=[1]))
        wl.base_kernel.lengthscale = 1.0
        return sm_time() * wl
    if name == "ampmod":
        wl = RBFKernel(active_dims=[1])
        wl.lengthscale = 3.0
        return AmpMod(sm_time() * wl, 5)
    if name == "icm_rank1":
        return sm_time() * IndexKernel(5, rank=1, active_dims=[2])
    if name == "icm_rank2":
        return sm_time() * IndexKernel(5, rank=2, active_dims=[2])
    if name == "chromatic_sm":
        return ChromaticSM(2, 5)
    if name == "chromatic_sm+indep":
        ind = sm_time(1) * IndexKernel(5, rank=1, active_dims=[2])
        ind.kernels[1].covar_factor.data.zero_()
        ind.kernels[1].raw_var.data.fill_(-6.0)
        return ChromaticSM(2, 5) + ind
    if name == "twoD_sm":
        k = SMK(num_mixtures=2, ard_num_dims=2, active_dims=[0, 1])
        k.mixture_means = torch.tensor([[f0, 0.01], [2 * f0, 0.01]]).view(2, 1, 2)
        k.mixture_scales = torch.tensor([[0.05, 0.3], [0.05, 0.3]]).view(2, 1, 2)
        return k


def fit(name, constmean=False):
    lik = gp.likelihoods.FixedNoiseGaussianLikelihood(
        En**2, learn_additional_noise=False
    )
    m = M(
        X, Yn, lik, build(name), gp.means.ConstantMean() if constmean else BandMean(5)
    )
    m.train()
    lik.train()
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    mll = gp.mlls.ExactMarginalLogLikelihood(lik, m)
    for i in range(800):
        opt.zero_grad()
        loss = -mll(m(X), Yn)
        loss.backward()
        opt.step()
    m.eval()
    lik.eval()
    res = []
    with torch.no_grad(), gp.settings.fast_pred_var():
        for b in range(5):
            tg = torch.linspace(0, T, 400)
            xg = torch.stack(
                [
                    tg / T,
                    torch.full_like(tg, math.log(lams[b])),
                    torch.full_like(tg, b),
                ],
                -1,
            )
            f = m(xg)
            mu = f.mean * ysd + ymu
            sd = f.stddev * ysd
            tr = truth(tg, b)
            gapm = (tg > 800) & (tg < 1200)
            sel = gapm if b == 3 else torch.ones_like(gapm)
            res.append(
                (
                    float(sd[sel].mean() / amps[b]),
                    float(((mu - tr)[sel] ** 2).mean().sqrt() / amps[b]),
                    float(((mu - tr).abs() < 2 * sd)[sel].double().mean()),
                )
            )
    info = ""
    k = m.k
    for nm, p in k.named_parameters():
        if (
            "log_amp" in nm
            or "raw_mu" in nm
            or "mixture_means" in nm
            or "phi" in nm
            or "lengthscale" in nm
        ):
            info += f" {nm}={np.round(p.detach().numpy().ravel(),2).tolist()}"
    return float(loss) * len(Yn), res, sum(p.numel() for p in m.parameters()), info


print(
    f"LAG={LAG} cycles at reddest band; columns per band: "
    "mean(pred sd)/amp, RMSE/amp, 2sigma coverage"
)
for name0 in [
    "separable_rbf",
    "ampmod",
    "icm_rank1",
    "chromatic_sm",
    "chromatic_sm+indep",
]:
    name = name0.split("/")[0]
    try:
        nll, res, npar, info = fit(name, constmean="/const" in name0)
        print(
            f"{name0:20s} -logML={nll:9.1f} npar={npar:3d} | "
            + " | ".join(f"{a:.3f},{r:.3f},{c:.2f}" for a, r, c in res)
        )
        pass
    except Exception as e:
        print(name, "FAILED", type(e).__name__, str(e)[:100])
