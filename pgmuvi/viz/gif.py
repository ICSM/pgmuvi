from __future__ import annotations
import contextlib
import io
import json
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch
try:
    import imageio.v2 as imageio
except (ImportError, ModuleNotFoundError):
    import imageio
from ..utils.helpers import _clean_errors
import logging
log = logging.getLogger("pgmuvi.viz.gif")

def make_sed_gif_phase(object_id: str,
                       lam_unique: np.ndarray,
                       band_predictors: list,
                       t: np.ndarray, y: np.ndarray, yerr: np.ndarray,
                       period_days: float, t0: float, rcfg, outdir: str,
                       is_mag_val: bool, overlay_dphi: float, n_phase: int,
                       cycles: int | None, model_label: str,
                       meta_sidecar: dict | None=None,
                       lam_full: np.ndarray=None) -> str:
    """
    Build a phase-folded SED GIF.
      - Model drives the SED at every phase for every band (smooth evolution).
      - Optional overlay: one *median* data point per band per frame,
        computed from observations with phase within ±overlay_dphi (in phase).
      - Y-limits are computed robustly from DATA ONLY (same policy as time-GIF).
    """
    if lam_full is None:
        raise ValueError('make_sed_gif_phase requires lam_full (wavelength per data point).')  # noqa: E501
    finite_y = np.isfinite(y)
    if rcfg.sed_loglog and (not is_mag_val):
        base = y[finite_y & (y > 0)]
        if base.size:
            y_p1, y_p99 = np.nanpercentile(base, [1.0, 99.0])
            y_lo, y_hi = (max(y_p1 / 1.5, 1e-06), y_p99 * 1.5)
            if not np.isfinite(y_hi) or y_hi <= y_lo:
                y_lo, y_hi = (1e-06, 1.0)
        else:
            y_lo, y_hi = (1e-06, 1.0)
    else:
        base = y[finite_y]
        if base.size:
            y_p1, y_p99 = np.nanpercentile(base, [1.0, 99.0])
            pad = 0.05 * (y_p99 - y_p1) if np.isfinite(y_p99 - y_p1) else 1.0
            y_lo = float(y_p1 - pad)
            y_hi = float(y_p99 + pad)
            if not np.isfinite(y_hi) or y_hi <= y_lo:
                y_lo, y_hi = (float(np.nanmin(base) - 1.0),
                              float(np.nanmax(base) + 1.0))
        else:
            y_lo, y_hi = (-1.0, 1.0)
    phases = np.linspace(0.0, 1.0, int(max(4, n_phase)), endpoint=False)
    if rcfg.phase_no_average:
        k_list = [int(np.round((np.nanmedian(t) - t0) / period_days))]
    else:
        tmin, tmax = (float(np.nanmin(t)), float(np.nanmax(t)))
        k0 = int(np.floor((tmin - t0) / period_days)) - 1
        k1 = int(np.ceil((tmax - t0) / period_days)) + 1
        k_range = list(range(k0, k1 + 1))
        if cycles is not None and cycles > 0 and (len(k_range) > cycles):
            k_center = int(np.round((np.nanmedian(t) - t0) / period_days))
            half = cycles // 2
            k_list = list(range(k_center - half, k_center - half + cycles))
        else:
            k_list = k_range
    frames = []
    for phi in phases:
        y_frame = np.full(len(lam_unique), np.nan, float)
        for i in range(len(lam_unique)):
            t_eval = np.asarray([t0 + (k + phi) * period_days for k in k_list], float)
            mu = np.asarray(band_predictors[i](t_eval), float)
            mu = mu[np.isfinite(mu)]
            if mu.size:
                y_frame[i] = float(np.nanmean(mu))
        overlay_vals = []
        if overlay_dphi and overlay_dphi > 0:
            ph_all = _phase_of(t, period_days, t0)
            lo = (phi - overlay_dphi) % 1.0
            hi = (phi + overlay_dphi) % 1.0
            in_bin = ((ph_all >= lo) & (ph_all <= hi)
                      if hi >= lo else (ph_all >= lo) | (ph_all <= hi))
            for lv in lam_unique:
                m_band = lam_full == lv
                sel = in_bin & m_band
                if np.any(sel):
                    y_med = np.nanmedian(y[sel])
                    eb_clean, _ = _clean_errors(yerr[sel], 0.0)
                    e_med = np.nanmedian(eb_clean)
                    overlay_vals.append((lv, float(y_med), float(e_med)))
        fig, ax = plt.subplots(figsize=(6, 4))
        valid = np.isfinite(y_frame)
        lam_plot = lam_unique[valid]
        y_plot = y_frame[valid]
        if rcfg.sed_loglog and (not is_mag_val):
            pos = y_plot > 0
            lam_plot = lam_plot[pos]
            y_plot = y_plot[pos]
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.plot(lam_plot, y_plot, '-o', lw=2, ms=4, label='model')
        if overlay_dphi and overlay_vals:
            for j, (lv, y_med, e_med) in enumerate(overlay_vals):
                if rcfg.sed_loglog and (not is_mag_val) and (y_med <= 0):
                    continue
                ax.errorbar([lv], [y_med], yerr=[e_med], fmt='o', ms=4, alpha=0.85,
                            label='data (median)' if j == 0 else None)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Magnitude' if is_mag_val else 'Flux')
        ax.set_title(f'{object_id}: SED @ phase={phi:0.2f} (P={period_days:.2f} d, t0={t0:.1f}, {model_label})')  # noqa: E501
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
    gif_path = os.path.join(outdir,
                            f'{object_id}_sed_phase_P{period_days:.2f}_t0{t0:.1f}.gif')
    imageio.mimsave(gif_path, frames, duration=0.12)
    meta = {
        'object_id': object_id,
        'P_days': period_days,
        't0': t0,
        'n_phase': n_phase,
        'overlay_dphi': overlay_dphi,
        'cycles': None if cycles is None else int(cycles),
        'model_label': model_label,
        'sed_loglog': bool(rcfg.sed_loglog),
    }
    if meta_sidecar:
        meta |= meta_sidecar
        meta['provenance'] = meta.get('provenance',
                                      meta.get('period_provenance', '')) or ''
    with contextlib.suppress(Exception):
        _debug_dump_json(os.path.join(outdir, f'{object_id}_sed_phase_meta.json'), meta)
    return gif_path

def _phase_of(t, P, t0):
    """Return phase in [0,1) given time array t, period P, and reference epoch t0."""
    return np.mod((np.asarray(t, float) - float(t0)) / float(P), 1.0)

def _debug_dump_json(path: str, payload: dict):
    """
    JSON dump that tolerates numpy types.
    """

    def _default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().numpy().tolist()
        return str(o)
    try:
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2, default=_default)
    except Exception as e:
        warnings.warn(f'[debug] failed to write {path}: {e}',
                      RuntimeWarning,
                      stacklevel=2)



def _harmonize_cli(args):
    """
    Normalize/resolve potentially conflicting CLI options with minimal surprises.
    - Legacy --multitask vs --fit-mode
    - LMC/ICM-only knobs in other modes
    - Bootstrap only for independent fits
    - Auto-enable phase GIF when phase-* options are supplied
    """
    if getattr(args, 'multitask', 'none') != 'none':
        if args.fit_mode in ('icm', 'lmc'):
            log.info(f'[note] --multitask {args.multitask} is redundant with --fit-mode={args.fit_mode}.')  # noqa: E501
        else:
            log.warning(f"[warn] --multitask={args.multitask} conflicts with --fit-mode={args.fit_mode}. Forcing fit-mode to '{args.multitask}'.")  # noqa: E501
            args.fit_mode = args.multitask
    elif args.fit_mode in ('icm', 'lmc'):
        log.info(f'[note] --multitask {args.fit_mode} is implied by --fit-mode={args.fit_mode}.')  # noqa: E501
        args.multitask = args.fit_mode
    if (getattr(args, 'multitask', 'none') in ('icm', 'lmc') and
        args.fit_mode in ('icm', 'lmc') and args.multitask != args.fit_mode):
        log.warning(f'[warn] --multitask={args.multitask} conflicts with --fit-mode={args.fit_mode}; overriding --multitask → {args.fit_mode}.')  # noqa: E501
        args.multitask = args.fit_mode
    if getattr(args, 'y_space', 'logflux') not in ('logflux', 'flux'):
        log.warning("[warn] --y-space unsupported; forcing to 'logflux'.")
        args.y_space = 'logflux'
    if hasattr(args, 'sed_loglog'):
        log.info('[note] --sed-loglog is deprecated and ignored; SED GIFs are always log-log.')  # noqa: E501
        with contextlib.suppress(Exception):
            delattr(args, 'sed_loglog')
    if args.fit_mode != 'lmc':
        if getattr(args, 'lmc_Q', 2) != 2:
            log.info('[note] --lmc-Q is ignored unless --fit-mode=lmc.')
        if getattr(args, 'lmc_M', 128) != 128:
            log.info('[note] --lmc-M is ignored unless --fit-mode=lmc.')
    if args.fit_mode != 'icm' and getattr(args, 'icm_rank', 2) != 2:
        log.info('[note] --icm-rank is ignored unless --fit-mode=icm.')
        log.warning(f'[warn] bootstrap is not supported for joint fits (fit-mode={args.fit_mode}). Forcing --bootstrap=0.')  # noqa: E501
        args.bootstrap = 0
    phase_opts = any([getattr(args, 'phase_P', None) is not None,
                      getattr(args, 'phase_t0', None) is not None,
                      getattr(args, 'phase_n', 60) != 60,
                      (getattr(args, 'phase_overlay_dphi', None) is not None and
                       args.phase_overlay_dphi > 0),
                      getattr(args, 'phase_cycles', None) is not None,
                      getattr(args, 'phase_no_average', False)])
    if phase_opts and (not getattr(args, 'phase_gif', False)):
        log.info('[note] phase options provided without --phase-gif; enabling phase GIF.')  # noqa: E501
        args.phase_gif = True
    return args
