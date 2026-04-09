"""
Bayesian change point detection utils using https://github.com/hildensia/bayesian_changepoint_detection 

"""

import bayesian_changepoint_detection.offline_changepoint_detection as offcd







def load_data(bed_power, layers):
    
    surface_elevation = layers['standard:surface']['wgs84']
    bed_elevation = layers['standard:bottom']['wgs84']
    
    # return dict(
    #     x_bp  = bp["along_track"].values / 1000,
    #     amp   = bp["bed_power_dB"].values,
    #     x_bed = bt["along_track"].values / 1000,
    #     h_bed = bt["wgs84"].values,
    #     x_sf  = sf["along_track"].values / 1000,
    #     h_sf  = sf["wgs84"].values,
    # )



# =============================================================================
# FEATURE COMPUTATION
# =============================================================================
# def compute_features(d):
#     """
#     Compute four normalised features on the bed-power grid.

#     f_amp   : filtered amplitude level
#     f_dA    : amplitude gradient (NaN artifact region masked)
#     f_dfree : flotation residual Δfree
#     f_dsurf : ice surface slope

#     Each is normalised to zero mean, unit std in the floating section.
#     All arrays have the same length as x_bp.
#     """
#     x_bp, amp = d["x_bp"], d["amp"]
#     x_bed, h_bed = d["x_bed"], d["h_bed"]
#     x_sf, h_sf = d["x_sf"], d["h_sf"]

#     nan_mask = np.isnan(amp)

#     # Filtered amplitude
#     amp_i    = pd.Series(amp).interpolate().ffill().bfill().values
#     b, a     = butter(AMP_ORDER, AMP_WN, btype="low", analog=False)
#     amp_f    = filtfilt(b, a, amp_i)
#     amp_f_out = amp_f.copy(); amp_f_out[nan_mask] = np.nan

#     # Amplitude gradient with NaN artifact masking
#     dA = np.gradient(amp_f, x_bp)
#     dA[(x_bp > GRADIENT_ARTIFACT_LO) & (x_bp < GRADIENT_ARTIFACT_HI)] = np.nan
#     dA[np.abs(dA) > GRADIENT_MAX_ABS] = np.nan
#     dA = pd.Series(dA).interpolate(limit=5).ffill().bfill().values

#     # Flotation residual Δfree = h_surf − H·(1 − ρ_ice/ρ_sw)
#     # > 0  → grounded,  ≈ 0  → floating,  < 0  → below flotation
#     h_sf_on_bed   = np.interp(x_bed, x_sf, h_sf)
#     H_bed         = h_sf_on_bed - h_bed
#     b2, a2        = butter(ELEV_ORDER, SLOPE_WN, btype="low", analog=False)
#     h_surf_sm     = filtfilt(b2, a2, h_sf_on_bed)
#     H_sm          = h_surf_sm - filtfilt(b2, a2, h_bed)
#     delta_free_sm = h_surf_sm - H_sm * (1 - RHO_ICE / RHO_SW)
#     delta_free_bp = np.interp(x_bp, x_bed, delta_free_sm)

#     # Surface slope
#     h_sf_bp_sm = filtfilt(b2, a2, np.interp(x_bp, x_sf, h_sf))
#     d_surf_bp  = np.gradient(h_sf_bp_sm, x_bp)

#     # Smoothed bed for plotting
#     hbed_bp  = np.interp(x_bp, x_bed, h_bed, left=np.nan, right=np.nan)
#     hbed_i   = pd.Series(hbed_bp).interpolate().ffill().bfill().values
#     be, ae   = butter(AMP_ORDER, ELEV_WN, btype="low", analog=False)
#     hbed_bw  = filtfilt(be, ae, hbed_i)
#     hbed_bw_out = hbed_bw.copy(); hbed_bw_out[np.isnan(hbed_bp)] = np.nan

#     d.update(dict(
#         nan_mask       = nan_mask,
#         amp_f          = amp_f,
#         amp_f_out      = amp_f_out,
#         dA             = dA,
#         delta_free_bp  = delta_free_bp,
#         delta_free_bed = np.interp(x_bed, x_bp, delta_free_bp),
#         d_surf_bp      = d_surf_bp,
#         H_bp           = np.interp(x_bp, x_bed, H_bed),
#         h_sf_bp        = np.interp(x_bp, x_sf, h_sf),
#         hbed_bw        = hbed_bw,
#         hbed_bw_out    = hbed_bw_out,
#         H_bed          = H_bed,
#     ))
#     return d


def run_bocpd(d, gz_lo_km, search_lo_km=DEFAULT_SEARCH_LO):
    """
    Run Fearnhead (2006) offline BOCPD with three likelihood models.

    The package's offline_changepoint_detection(data, prior, likelihood)
    returns (Q, P, Pcp) where:
        Pcp[j, t] = log P(data has j+1 CPs, last one at index t)
    Pcp has shape (n-1, n-1), so:
        cp_prob = exp(Pcp).sum(axis=0)  has length n-1
        aligns to x[:-1] (CP between samples t-1 and t)

    Three likelihood models are run:
        1D Gaussian  : offcd.gaussian_obs_log_likelihood  — amplitude only
        IFM          : offcd.ifm_obs_log_likelihood       — 4 features, indep.
        FullCov      : offcd.fullcov_obs_log_likelihood   — 4 features, full cov

    Their log-posteriors are combined with weights W_GAUSS, W_IFM, W_FCOV.
    """
    x_bp = d["x_bp"]
    win  = (x_bp >= search_lo_km) & (x_bp < gz_lo_km)
    x_win = x_bp[win]
    n     = int(win.sum())

    signal = np.column_stack([
        d["f_amp"][win],
        d["f_dA"][win],
        d["f_dfree"][win],
        d["f_dsurf"][win],
    ])

    # Uniform prior: const_prior(r, l) = 1/l  — equal probability per position
    prior = partial(offcd.const_prior, l=n + 1)

    print("Running 1D Gaussian model...")
    _, _, Pcp1 = offcd.offline_changepoint_detection(
        d["f_amp"][win], prior, offcd.gaussian_obs_log_likelihood,
        truncate=TRUNCATE)
    cp_gauss = np.exp(Pcp1).sum(0); cp_gauss /= cp_gauss.sum()

    print("Running IFM (4-feature independent) model...")
    _, _, Pcp3 = offcd.offline_changepoint_detection(
        signal, prior, offcd.ifm_obs_log_likelihood, truncate=TRUNCATE)
    cp_ifm = np.exp(Pcp3).sum(0); cp_ifm /= cp_ifm.sum()

    print("Running FullCov (4-feature covariance) model...")
    _, _, Pcp2 = offcd.offline_changepoint_detection(
        signal, prior, offcd.fullcov_obs_log_likelihood, truncate=TRUNCATE)
    cp_fcov = np.exp(Pcp2).sum(0); cp_fcov /= cp_fcov.sum()

    # Pcp has shape (n-1, n-1) → align to x_win[:-1]
    x_cp = x_win[:-1]

    # Combine in log-space with reliability weights
    log_comb = (W_GAUSS * np.log(cp_gauss + 1e-15)
                + W_IFM  * np.log(cp_ifm   + 1e-15)
                + W_FCOV * np.log(cp_fcov  + 1e-15)) / (W_GAUSS + W_IFM + W_FCOV)
    log_comb -= np.logaddexp.reduce(log_comb)
    cp_comb  = np.exp(log_comb)
    cp_comb  = uniform_filter1d(cp_comb, size=SMOOTH_SIZE)
    cp_comb /= cp_comb.sum()

    # MAP and credible intervals
    n1  = len(x_cp)
    cdf = np.cumsum(cp_comb)
    GP_km  = float(x_cp[np.argmax(cp_comb)])
    lo68   = float(x_cp[np.searchsorted(cdf, 0.160)])
    hi68   = float(x_cp[min(np.searchsorted(cdf, 0.840), n1 - 1)])
    lo95   = float(x_cp[np.searchsorted(cdf, 0.025)])
    hi95   = float(x_cp[min(np.searchsorted(cdf, 0.975), n1 - 1)])

    return dict(
        x_cp     = x_cp,
        x_win    = x_win,
        cp_gauss = cp_gauss,
        cp_ifm   = cp_ifm,
        cp_fcov  = cp_fcov,
        cp_comb  = cp_comb,
        gp_km    = GP_km,
        lo68=lo68, hi68=hi68,
        lo95=lo95, hi95=hi95,
        search_lo_km = search_lo_km,
        gz_lo_km     = gz_lo_km,
    )





















