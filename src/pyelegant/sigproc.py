import numpy as np


def get_amp_corr_fac_for_window(window):
    if window == "rect":
        amp_corr_fac = 1.0 / np.sinc(0.0)  # = 1.0
    elif window == "sine":
        amp_corr_fac = 1.0 / (
            (np.sinc(0.0 + 0.5) + np.sinc(0.0 - 0.5)) / 2.0
        )  # ~ 0.63662
    elif window == "sine_squared":
        amp_corr_fac = 1.0 / (np.sinc(0.0) / (1.0 - 0.0**2) / 2.0)  # = 0.5
    else:
        raise NotImplementedError

    return amp_corr_fac


def apply_window(array, window):
    array = np.array(array)
    n = array.size

    if window == "sine":
        sin = np.sin(np.linspace(0, n - 1, n) / (n - 1) * np.pi)
        return sin * array
    elif window == "rect":
        return array
    elif window == "sine_squared":
        sin = np.sin(np.linspace(0, n - 1, n) / (n - 1) * np.pi)
        return sin * sin * array
    else:
        print(" ")
        print("Valid window strings are:")
        print('   "rect", "sine", "sine_squared"')
        print(" ")
        raise ValueError(f"Unexpected window: {window}")


def dft(x, req_tune):
    """ """

    n = len(x)
    k = req_tune * n
    indexes = np.array(range(n))
    exp_vec = np.exp(-2.0 * np.pi * 1j * indexes / n * k)

    return np.sum(x * exp_vec)


def golden(func, args=(), brack=None, abstol=1e-12, full_output=0):
    """
    Copied from scipy.optimize.golden(). The only difference is the use of
    `abstol`, instead of the fractional precision tolerance `tol`.

    Given a function of one-variable and a possible bracketing interval,
    return the minimum of the function isolated to a fractional precision of
    tol.

    Parameters
    ----------
    func : callable func(x,*args)
        Objective function to minimize.
    args : tuple
        Additional arguments (if present), passed to func.
    brack : tuple
        Triple (a,b,c), where (a<b<c) and func(b) <
        func(a),func(c).  If bracket consists of two numbers (a,
        c), then they are assumed to be a starting interval for a
        downhill bracket search (see `bracket`); it doesn't always
        mean that obtained solution will satisfy a<=x<=c.
    tol : float
        x tolerance stop criterion
    full_output : bool
        If True, return optional outputs.

    Notes
    -----
    Uses analog of bisection method to decrease the bracketed
    interval.

    """
    if brack is None:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(func, args=args)
    elif len(brack) == 2:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(
            func, xa=brack[0], xb=brack[1], args=args
        )
    elif len(brack) == 3:
        xa, xb, xc = brack
        if xa > xc:  # swap so xa < xc can be assumed
            dum = xa
            xa = xc
            xc = dum
        if not ((xa < xb) and (xb < xc)):
            raise ValueError("Not a bracketing interval.")
        fa = func(*((xa,) + args))
        fb = func(*((xb,) + args))
        fc = func(*((xc,) + args))
        if not ((fb < fa) and (fb < fc)):
            raise ValueError("Not a bracketing interval.")
        funcalls = 3
    else:
        raise ValueError("Bracketing interval must be length 2 or 3 sequence.")

    _gR = 0.61803399
    _gC = 1.0 - _gR
    x3 = xc
    x0 = xa
    if abs(xc - xb) > abs(xb - xa):
        x1 = xb
        x2 = xb + _gC * (xc - xb)
    else:
        x2 = xb
        x1 = xb - _gC * (xb - xa)
    f1 = func(*((x1,) + args))
    f2 = func(*((x2,) + args))
    funcalls += 2
    while abs(x2 - x1) > abstol:
        if f2 < f1:
            x0 = x1
            x1 = x2
            x2 = _gR * x1 + _gC * x3
            f1 = f2
            f2 = func(*((x2,) + args))
        else:
            x3 = x2
            x2 = x1
            x1 = _gR * x2 + _gC * x0
            f2 = f1
            f1 = func(*((x1,) + args))
        funcalls += 1
    if f1 < f2:
        xmin = x1
        fval = f1
    else:
        xmin = x2
        fval = f2
    if full_output:
        return xmin, fval, funcalls
    else:
        return xmin


def _dft_peak_finding_objective(
    new_nu, xwin, n, amp_corr_fac, dft_peak_nus, dft_peak_As
):
    """"""

    dft_win = dft(xwin, new_nu)
    new_A = np.abs(dft_win) * 2.0 / n * amp_corr_fac

    dft_peak_nus.append(new_nu)
    dft_peak_As.append(new_A)

    return -new_A


def get_dft_peak(x_rect, init_nu, window="sine", resolution=1e-8, return_fft_spec=True):
    """
    `x_rect` must be a rectangular-windowed vector.

    If `x_rect` is a real-valued vector, the returned amplitude vector will be
    the amplitude of one-sided spectrum (i.e., the two-sided spectrum multiplied
    by 2 except for the DC component). If it is a complex-valued vector, the
    returned amplitude will that of the two-sided spectrum.
    """

    n = len(x_rect)
    nus = np.fft.fftfreq(n)

    x_det = x_rect - np.mean(x_rect)

    ff = np.fft.fft(x_det)
    phis = np.angle(ff)

    xwin = apply_window(x_det, window)

    amp_corr_fac = get_amp_corr_fac_for_window(window)

    ffwin = np.fft.fft(xwin)
    A = np.abs(ffwin) * 2.0 / n * amp_corr_fac

    dft_win = dft(xwin, init_nu)
    new_A = np.abs(dft_win) * 2.0 / n * amp_corr_fac
    peak_inds = np.argsort(np.abs(nus - init_nu))[:2]
    if A[peak_inds[0]] >= A[peak_inds[1]]:
        peak_ind = peak_inds[0]
    else:
        peak_ind = peak_inds[1]

    if peak_ind == len(A) - 1:
        dft_peak_nus = np.array([nus[peak_ind], nus[peak_ind - 1], init_nu])
        dft_peak_As = np.array([A[peak_ind], A[peak_ind - 1], new_A])
    elif peak_ind == 0:
        dft_peak_nus = np.array([nus[peak_ind], nus[peak_ind + 1], init_nu])
        dft_peak_As = np.array([A[peak_ind], A[peak_ind + 1], new_A])
    else:
        if A[peak_ind - 1] > A[peak_ind + 1]:
            dft_peak_nus = np.array([nus[peak_ind], nus[peak_ind - 1], init_nu])
            dft_peak_As = np.array([A[peak_ind], A[peak_ind - 1], new_A])
        else:
            dft_peak_nus = np.array([nus[peak_ind], nus[peak_ind + 1], init_nu])
            dft_peak_As = np.array([A[peak_ind], A[peak_ind + 1], new_A])

    try:
        backup = dict(dft_peak_nus=dft_peak_nus.copy(), dft_peak_As=dft_peak_As.copy())

        if (
            (dft_peak_As[2] < dft_peak_As[0])
            and (dft_peak_As[2] < dft_peak_As[1])
            and (
                (dft_peak_nus[0] < dft_peak_nus[2] < dft_peak_nus[1])
                or (dft_peak_nus[1] < dft_peak_nus[2] < dft_peak_nus[0])
            )
        ):
            dft_peak_nus = dft_peak_nus.tolist()
            dft_peak_As = dft_peak_As.tolist()
            out = golden(
                _dft_peak_finding_objective,
                args=(xwin, n, amp_corr_fac, dft_peak_nus, dft_peak_As),
                brack=np.sort(dft_peak_nus),
                abstol=resolution,
                full_output=1,
            )
        else:
            raise ValueError("Invalid bracket. Use slower but robust method.")

    except:
        dft_peak_nus = backup["dft_peak_nus"]
        dft_peak_As = backup["dft_peak_As"]

        # 25% slower (but more robust) than the one that uses "golden()"
        sort_inds = np.argsort(dft_peak_As)[::-1]
        dft_peak_As = dft_peak_As[sort_inds]
        dft_peak_nus = dft_peak_nus[sort_inds]
        dnu = nus[1] - nus[0]
        while dnu > resolution:
            new_nu = dft_peak_nus[0] + dnu / 2.0
            dft_win = dft(xwin, new_nu)
            new_A = np.abs(dft_win) * 2.0 / n * amp_corr_fac
            dft_peak_nus = np.append(dft_peak_nus, new_nu)
            dft_peak_As = np.append(dft_peak_As, new_A)
            new_nu = dft_peak_nus[0] - dnu / 2.0
            dft_win = dft(xwin, new_nu)
            new_A = np.abs(dft_win) * 2.0 / n * amp_corr_fac
            dft_peak_nus = np.append(dft_peak_nus, new_nu)
            dft_peak_As = np.append(dft_peak_As, new_A)
            sort_inds = np.argsort(dft_peak_As)[::-1]
            dft_peak_As = dft_peak_As[sort_inds]
            dft_peak_nus = dft_peak_nus[sort_inds]
            dnu /= 2.0

    dft_peak_nu = dft_peak_nus[np.argmax(dft_peak_As)]
    _dft = dft(xwin, dft_peak_nu)
    dft_peak_A = np.abs(_dft) * 2.0 / n * amp_corr_fac
    if np.iscomplex(x_rect[0]):
        dft_peak_A /= 2.0
    dft_peak_phi = np.angle(_dft)

    out = dict(A=dft_peak_A, nu=dft_peak_nu, phi=dft_peak_phi)
    if return_fft_spec:
        out["fft_nus"] = nus
        out["fft_As"] = A

    return out


def find_closest_fft_peak(x_rect, init_nu, window="sine"):
    """
    `x_rect` must be a rectangular-windowed vector.
    """

    n = len(x_rect)
    nus = np.fft.fftfreq(n)

    x_det = x_rect - np.mean(x_rect)

    xwin = apply_window(x_det, window)

    amp_corr_fac = get_amp_corr_fac_for_window(window)

    ffwin = np.fft.fft(xwin)
    A = np.abs(ffwin) * 2.0 / n * amp_corr_fac

    closest_index = np.argmin(np.abs(nus - init_nu))
    #
    downturns = np.where(np.diff(A[: (closest_index + 1)][::-1]) < 0.0)[0]
    if downturns.size == 0:
        us_peak_ind = 0
    else:
        us_peak_ind = closest_index - downturns[0]
    #
    downturns = np.where(np.diff(A[closest_index:]) < 0.0)[0]
    if downturns.size == 0:
        ds_peak_ind = A.size - 1
    else:
        ds_peak_ind = closest_index + downturns[0]
    #
    if us_peak_ind == ds_peak_ind:
        peak_ind = us_peak_ind
    elif A[us_peak_ind] >= A[ds_peak_ind]:
        peak_ind = us_peak_ind
    else:
        peak_ind = ds_peak_ind

    return dict(A=A[peak_ind], nu=nus[peak_ind])


def get_fft_peak_indexes(n_peaks, fft_amplitudes, min_nu=0.0, max_nu=1.0):
    """
    "fft_amplitudes" must be the full FFT amplitude vector, i.e., the length of
    "fft_amplitudes" and the length of the original time-domain array must be
    the same.
    """

    if n_peaks < 1:
        raise ValueError('"n_peaks" must be a positive integer.')

    if not (
        (min_nu >= 0.0) and (min_nu <= 1.0) and (max_nu >= 0.0) and (max_nu <= 1.0)
    ):
        raise ValueError('"min_nu" and "max_nu" must be between 0 and 1.')
    if min_nu >= max_nu:
        raise ValueError('"min_nu" must be smaller than "max_nu"')

    As = fft_amplitudes.copy()

    n = len(As)

    nus = np.fft.fftfreq(n)
    nus[nus < 0.0] += 1.0  # = [0, ..., (0.5 - 1/n), 0.5, ..., (1.0-1/n)]

    in_range = np.logical_and((nus >= min_nu), (nus <= max_nu))

    use_fast_algo = True

    peak_inds = []
    for i in range(n_peaks):
        peak_iLeft, peak_iRight = None, None

        if use_fast_algo:
            index_offset = np.where(in_range)[0][0]
            sort_inds = np.argsort(As[in_range])[::-1] + index_offset
            peak_height = 0.0
            for iMid in sort_inds:
                if iMid == 0:
                    if (As[0] > peak_height) and (As[0] > As[1]):
                        peak_height = As[0]
                        peak_index = 0
                        peak_iLeft = peak_iRight = 1
                        break
                elif iMid == n - 1:
                    if (As[n - 1] > peak_height) and (As[n - 1] > As[n - 2]):
                        peak_height = As[n - 1]
                        peak_index = n - 1
                        peak_iLeft = peak_iRight = n - 2
                        break
                else:
                    iLeft = iMid - 1
                    iRight = iMid + 1
                    if (
                        (As[iMid] > peak_height)
                        and (As[iMid] > As[iLeft])
                        and (As[iMid] > As[iRight])
                    ):
                        peak_height = As[iMid]
                        peak_index = iMid
                        peak_iLeft = iLeft
                        peak_iRight = iRight
                        break
        else:
            peak_height = 0.0
            for iMid in np.array(range(n))[in_range]:
                if iMid == 0:
                    if (As[0] > peak_height) and (As[0] > As[1]):
                        peak_height = As[0]
                        peak_index = 0
                        peak_iLeft = peak_iRight = 1
                elif iMid == n - 1:
                    if (As[n - 1] > peak_height) and (As[n - 1] > As[n - 2]):
                        peak_height = As[n - 1]
                        peak_index = n - 1
                        peak_iLeft = peak_iRight = n - 2
                else:
                    iLeft = iMid - 1
                    iRight = iMid + 1
                    if (
                        (As[iMid] > peak_height)
                        and (As[iMid] > As[iLeft])
                        and (As[iMid] > As[iRight])
                    ):
                        peak_height = As[iMid]
                        peak_index = iMid
                        peak_iLeft = iLeft
                        peak_iRight = iRight

        if (peak_iLeft is not None) and (peak_iRight is not None):
            # Make peak flat to allow for finding a next peak
            if As[peak_iLeft] > As[peak_iRight]:
                As[peak_index] = As[peak_iLeft]
            else:
                As[peak_index] = As[peak_iRight]
        else:
            peak_index = np.nan

        peak_inds.append(peak_index)

    return peak_inds


def get_approx_tunes(detrended_x_vec, detrended_y_vec, window="sine_squared"):
    x = detrended_x_vec
    y = detrended_y_vec

    n_turns = x.size

    amp_corr_fac = get_amp_corr_fac_for_window(window)

    ff_win = np.fft.fft(apply_window(x, window))
    Ax_win = np.abs(ff_win) * 2.0 / n_turns * amp_corr_fac

    ff_win = np.fft.fft(apply_window(y, window))
    Ay_win = np.abs(ff_win) * 2.0 / n_turns * amp_corr_fac

    min_nu = 0.0
    if isinstance(x[0], complex):
        max_nu = 1.0
    else:
        max_nu = 0.5

    n_peaks = 2
    interp_nus = {}

    interp_nus["x"] = np.full((n_peaks,), np.nan)
    As = Ax_win
    peak_inds = get_fft_peak_indexes(n_peaks, As, min_nu=min_nu, max_nu=max_nu)
    for i, p_i in enumerate(peak_inds):
        interp_nus["x"][i] = get_fft_peak_interp_nu(p_i, As, window=window)

    interp_nus["y"] = np.full((n_peaks,), np.nan)
    As = Ay_win
    peak_inds = get_fft_peak_indexes(n_peaks, As, min_nu=min_nu, max_nu=max_nu)
    for i, p_i in enumerate(peak_inds):
        interp_nus["y"][i] = get_fft_peak_interp_nu(p_i, As, window=window)

    return interp_nus


def get_fft_peak_interp_nu(peak_index, fft_amplitudes, window="sine_squared"):
    """
    "fft_amplitudes" must be the full FFT amplitude vector, i.e., the length of
    "fft_amplitudes" and the length of the original time-domain array must be
    the same.

    "window" must be the window used to obtain the passed "fft_amplitudes".
    Using a different window may result in an inaccurate amplitude and tune
    interpolation.

    Also, it is usutally recommended to use "sine" or "sine_squared" window
    for amplitude/tune estimation, as they tend to give better estimation than
    "rect" window.
    """

    if np.isnan(peak_index):
        interp_nu = np.nan
        return interp_nu

    k = peak_index
    As = fft_amplitudes

    n = len(As)

    if k == 0:
        k_plus_1 = 1
        direction = 1.0
    elif k == n - 1:
        k_plus_1 = n - 2
        direction = -1.0
    elif As[k - 1] >= As[k + 1]:
        k_plus_1 = k - 1
        direction = -1.0
    else:
        k_plus_1 = k + 1
        direction = 1.0

    kf = float(k)

    if window == "rect":
        interp_nu = (
            kf / n + np.arctan(As[k_plus_1] / (As[k] + As[k_plus_1])) / n * direction
        )
    elif (window == "sine") or (window == "sine_squared"):
        if window == "sine":
            l = 1
        else:
            l = 2
        interp_nu = (
            kf / n
            + ((l + 1) * As[k_plus_1] / (As[k] + As[k_plus_1]) - l / 2.0)
            / n
            * direction
        )
    else:
        raise ValueError(f'Unexpected "window": {window}')

    return interp_nu


def _check_nu0_range(nux0_range, nuy0_range):
    """"""

    if nux0_range is None:
        lower_nux = True
        nux0_range = [0.0, 0.5]
    elif len(nux0_range) != 2:
        raise ValueError(
            f'Length of "nux0_range" list must be 2. Length is {len(nux0_range)}.'
        )
    elif nux0_range[0] > nux0_range[1]:
        raise ValueError(
            (
                "nux0_range: 1st element must be smaller than 2nd element. "
                f'nux0_range is "{nux0_range}".'
            )
        )
    elif (nux0_range[0] <= 0.5) and (nux0_range[1] <= 0.5):
        lower_nux = True
    elif (nux0_range[0] >= 0.5) and (nux0_range[1] >= 0.5):
        lower_nux = False
    else:
        raise ValueError(
            (
                "Min. and max. nux0 must be either both below 0.5 "
                f'or both above 0.5. nux0_range is "{nux0_range}".'
            )
        )

    if nuy0_range is None:
        lower_nuy = True
        nuy0_range = [0.0, 0.5]
    elif len(nuy0_range) != 2:
        raise ValueError(
            f'Length of "nuy0_range" list must be 2. Length is {len(nuy0_range)}.'
        )
    elif nuy0_range[0] > nuy0_range[1]:
        raise ValueError(
            (
                "nuy0_range: 1st element must be smaller than 2nd element. "
                f'nuy0_range is "{nuy0_range}".'
            )
        )
    elif (nuy0_range[0] <= 0.5) and (nuy0_range[1] <= 0.5):
        lower_nuy = True
    elif (nuy0_range[0] >= 0.5) and (nuy0_range[1] >= 0.5):
        lower_nuy = False
    else:
        raise ValueError(
            (
                "Min. and max. nuy0 must be either both below 0.5 "
                f'or both above 0.5. nuy0_range is "{nuy0_range}".'
            )
        )

    return lower_nux, lower_nuy, nux0_range, nuy0_range


def select_fundamental_tunes(
    nux_array,
    nuy_array,
    kick_type,
    max_sync_tune=1e-3,
    min_nu_distance=0.02,
    nux0_range=None,
    nuy0_range=None,
):
    """
    Determine horizontal and vertical tunes from a single set of turn-by-turn
    (TbT) horizontal and vertical data obtained by using either horizontal
    kicker only, vertical kicker only, or both horiz. & vert. kickers.

    For the "both kickers" case, it is simple to extract tune in each plane.
    Estimate nux from horiz. TbT data and nuy from vert. TbT data, with the
    restriction of "max_sync_tune", "nux0_range", and "nuy0_range", if specified.

    For the "horiz. kicker only" case, first, the horizontal tune is determined
    from horizontal TbT data. This should be relatively easily done as the
    horizontal signal is usually strong due to the fact that the beam was
    horizontally excited.

    The only main competing peak would be the synchrotron tune peak.
    This can be excluded by specifying `max_sync_tune`. Any peak found
    less than this tune will be consideredas a synchrotron tune peak,
    and thus excluded from the horizontal tune peak candidates.

    Once the horizontal tune is determined, the vertical tune is determined
    from the vertical TbT data. This is more difficult as the vertical
    tune is extracted through coupling from horizontal excitation.

    In most cases, the horizonotal tune peak also exists in the vertical
    spectrum. By specifying `min_nu_distance`, the code will identify any
    peak within the range of (nux - min_du_distance) and (nux + min_nu_distance)
    as the horizontal tune peak. The highest peak that is not considered as
    the synchrotron or the horizontal tune peak will be determined as the
    vertical tune peak.

    If this algorithm fails to correctly identiy the horizontal and vertical
    tune peaks, you can use `nux0_range` and `nuy0_range` to restrict the peak
    search range.

    The "vert. kicker only" case is similar to the "horiz. kicker only". The
    determination order is simply reversed.
    """

    if kick_type not in ("h", "v", "hv"):
        raise ValueError('"kick_type" must be either "h", "v", or "hv".')

    lower_nux, lower_nuy, nux0_range, nuy0_range = _check_nu0_range(
        nux0_range, nuy0_range
    )

    if not lower_nux:
        nux_array = 1.0 - nux_array
    if not lower_nuy:
        nuy_array = 1.0 - nuy_array

    nux0_min, nux0_max = np.min(nux0_range), np.max(nux0_range)
    nuy0_min, nuy0_max = np.min(nuy0_range), np.max(nuy0_range)

    if kick_type == "h":
        # Estimate nux from x-tbt and then nuy from y-tbt with restriction of
        # nuy being not too close from nux

        nux0 = np.nan
        for nux in nux_array:
            if lower_nux:
                if nux < max_sync_tune:
                    continue
            else:
                if nux > 1.0 - max_sync_tune:
                    continue
            if (nux >= nux0_min) and (nux <= nux0_max):
                nux0 = nux
                break

        nuy0 = np.nan
        for nuy in nuy_array:
            if lower_nuy:
                if nuy < max_sync_tune:
                    continue
            else:
                if nuy > 1.0 - max_sync_tune:
                    continue
            if abs(nuy - nux0) < min_nu_distance:
                continue
            if (nuy >= nuy0_min) and (nuy <= nuy0_max):
                nuy0 = nuy
                break
    elif kick_type == "v":
        # Estimate nuy from y-tbt and then nux from x-tbt with restriction of nux
        # being not too close from nuy

        nuy0 = np.nan
        for nuy in nuy_array:
            if lower_nuy:
                if nuy < max_sync_tune:
                    continue
            else:
                if nuy > 1.0 - max_sync_tune:
                    continue
            if (nuy >= nuy0_min) and (nuy <= nuy0_max):
                nuy0 = nuy
                break

        nux0 = np.nan
        for nux in nux_array:
            if lower_nux:
                if nux < max_sync_tune:
                    continue
            else:
                if nux > 1.0 - max_sync_tune:
                    continue
            if abs(nux - nuy0) < min_nu_distance:
                continue
            if (nux >= nux0_min) and (nux <= nux0_max):
                nux0 = nux
                break

    else:  # Estimate nux from x-tbt & nuy from y-tbt
        nux0 = np.nan
        for nux in nux_array:
            if lower_nux:
                if nux < max_sync_tune:
                    continue
            else:
                if nux > 1.0 - max_sync_tune:
                    continue
            if (nux >= nux0_min) and (nux <= nux0_max):
                nux0 = nux
                break

        nuy0 = np.nan
        for nuy in nuy_array:
            if lower_nuy:
                if nuy < max_sync_tune:
                    continue
            else:
                if nuy > 1.0 - max_sync_tune:
                    continue
            if (nuy >= nuy0_min) and (nuy <= nuy0_max):
                nuy0 = nuy
                break

    return nux0, nuy0


def get_linear_freq_components(
    x_vec,
    y_vec,
    kick_type="hv",
    window="sine_squared",
    max_sync_tune=1e-3,
    min_nu_distance=0.02,
    nu_resolution=1e-5,
    nux0_range=None,
    nuy0_range=None,
):
    assert not isinstance(x_vec[0], complex)  # Must be "real"

    approx_tunes = get_approx_tunes(x_vec, y_vec, window=window)

    nux0_approx, nuy0_approx = select_fundamental_tunes(
        approx_tunes["x"],
        approx_tunes["y"],
        kick_type,
        max_sync_tune=max_sync_tune,
        min_nu_distance=min_nu_distance,
        nux0_range=nux0_range,
        nuy0_range=nuy0_range,
    )

    if np.isnan(nux0_approx) and (nux0_range is not None):
        nux0_approx = np.mean(nux0_range)
    if np.isnan(nuy0_approx) and (nuy0_range is not None):
        nuy0_approx = np.mean(nuy0_range)

    # refine the fundamental peak nu with DFT
    _kwargs = dict(resolution=nu_resolution, window=window)
    nux0_peak = get_dft_peak(x_vec, nux0_approx, **_kwargs)
    nuy0_peak = get_dft_peak(y_vec, nuy0_approx, **_kwargs)

    nux0 = nux0_peak["nu"]
    nuy0 = nuy0_peak["nu"]

    hamp, vamp = {}, {}
    hphi, vphi = {}, {}

    n_turns = x_vec.size

    amp_corr_fac = get_amp_corr_fac_for_window(window)

    # Extract main (fundamental) peak info in H-plane
    nx, ny = 1, 0
    hamp[(nx, ny)] = nux0_peak["A"]
    hphi[(nx, ny)] = nux0_peak["phi"]

    # Extract info on secondary peak after subtracting main peak component in H-plane
    x_sub = (
        x_vec
        - (
            hamp[(1, 0)]
            * np.exp(
                (2.0 * np.pi * nux0 * np.array(range(n_turns)) + hphi[(1, 0)]) * 1j
            )
        ).real
    )
    x_sub_win = apply_window(x_sub, window)
    nx, ny = 0, 1
    _dft = dft(x_sub_win, nx * nux0 + ny * nuy0)
    hamp[(nx, ny)] = np.abs(_dft) * 2.0 / n_turns * amp_corr_fac
    hphi[(nx, ny)] = np.angle(_dft)

    # Extract main (fundamental) peak info in V-plane
    nx, ny = 0, 1
    vamp[(nx, ny)] = nuy0_peak["A"]
    vphi[(nx, ny)] = nuy0_peak["phi"]

    # Extract info on secondary peak after subtracting main peak component in V-plane
    y_sub = (
        y_vec
        - (
            vamp[(0, 1)]
            * np.exp(
                (2.0 * np.pi * nuy0 * np.array(range(n_turns)) + vphi[(0, 1)]) * 1j
            )
        ).real
    )
    y_sub_win = apply_window(y_sub, window)
    nx, ny = 1, 0
    _dft = dft(y_sub_win, nx * nux0 + ny * nuy0)
    vamp[(nx, ny)] = np.abs(_dft) * 2.0 / n_turns * amp_corr_fac
    vphi[(nx, ny)] = np.angle(_dft)

    return dict(nux0=nux0, nuy0=nuy0, hamp=hamp, vamp=vamp, hphi=hphi, vphi=vphi)


def get_linear_freq_components_from_xy_matrices(
    x_matrix,
    y_matrix,
    kick_type="hv",
    window="sine_squared",
    max_sync_tune=1e-3,
    min_nu_distance=0.02,
    nu_resolution=1e-5,
    nux0_range=None,
    nuy0_range=None,
):
    X = x_matrix
    Y = y_matrix

    nBPM = X.shape[1]

    nans = np.full((nBPM,), np.nan)

    nux0s = nans.copy()
    nuy0s = nans.copy()
    fx1 = dict(amp=nans.copy(), phi=nans.copy())
    fx2 = dict(amp=nans.copy(), phi=nans.copy())
    fy1 = dict(amp=nans.copy(), phi=nans.copy())
    fy2 = dict(amp=nans.copy(), phi=nans.copy())

    for i in range(nBPM):
        out = get_linear_freq_components(
            X[:, i],
            Y[:, i],
            kick_type=kick_type,
            window=window,
            max_sync_tune=max_sync_tune,
            min_nu_distance=min_nu_distance,
            nu_resolution=nu_resolution,
            nux0_range=nux0_range,
            nuy0_range=nuy0_range,
        )

        nux0s[i] = out["nux0"]
        nuy0s[i] = out["nuy0"]

        fx1["amp"][i] = out["hamp"][(1, 0)]
        fx1["phi"][i] = out["hphi"][(1, 0)]
        fx2["amp"][i] = out["hamp"][(0, 1)]
        fx2["phi"][i] = out["hphi"][(0, 1)]

        fy1["amp"][i] = out["vamp"][(0, 1)]
        fy1["phi"][i] = out["vphi"][(0, 1)]
        fy2["amp"][i] = out["vamp"][(1, 0)]
        fy2["phi"][i] = out["vphi"][(1, 0)]

    Cx1 = fx1["amp"] * np.exp(1j * fx1["phi"])
    Cy1 = fy1["amp"] * np.exp(1j * fy1["phi"])
    Cx2 = fx2["amp"] * np.exp(1j * fx2["phi"])
    Cy2 = fy2["amp"] * np.exp(1j * fy2["phi"])

    return dict(
        nux=nux0s,
        nuy=nuy0s,
        x1amp=fx1["amp"],
        x1phi=fx1["phi"],
        x1C=Cx1,
        x2amp=fx2["amp"],
        x2phi=fx2["phi"],
        x2C=Cx2,
        y1amp=fy1["amp"],
        y1phi=fy1["phi"],
        y1C=Cy1,
        y2amp=fy2["amp"],
        y2phi=fy2["phi"],
        y2C=Cy2,
    )


def _get_sqrt2J_from_ps_coords(x, px, y, py, Ascrinv):

    xhat, pxhat, yhat, pyhat = Ascrinv @ np.array([x, px, y, py])
    sqrt2Jx = np.sqrt(xhat**2 + pxhat**2)
    sqrt2Jy = np.sqrt(yhat**2 + pyhat**2)

    return sqrt2Jx, sqrt2Jy


def get_linopt_observations(
    lin_freq_comps, model_beta, model_alpha, fxy1_closed=True, use_dfxy1=True
):

    betax0_bpms = model_beta["bpms"]["x"]
    betay0_bpms = model_beta["bpms"]["y"]

    betax0_inj = model_beta["inj_pt"]["x"]
    betay0_inj = model_beta["inj_pt"]["y"]

    alphax0_inj = model_alpha["inj"]["x"]
    alphay0_inj = model_alpha["inj"]["y"]

    Ascr = np.zeros((4, 4))
    Ascr[0, 0] = np.sqrt(betax0_inj)
    Ascr[1, 0] = -alphax0_inj / np.sqrt(betax0_inj)
    Ascr[1, 1] = 1 / np.sqrt(betax0_inj)
    Ascr[2, 2] = np.sqrt(betay0_inj)
    Ascr[3, 2] = -alphay0_inj / np.sqrt(betay0_inj)
    Ascr[3, 3] = 1 / np.sqrt(betay0_inj)
    Ascrinv = np.linalg.inv(Ascr)

    fx1 = dict(amp=lin_freq_comps["x1amp"])
    fy1 = dict(amp=lin_freq_comps["y1amp"])

    if fxy1_closed:
        meas_avg_sqrt2Jx = np.mean(fx1["amp"][:-1] / np.sqrt(betax0_bpms))
        meas_avg_sqrt2Jy = np.mean(fy1["amp"][:-1] / np.sqrt(betay0_bpms))

    else:
        meas_avg_sqrt2Jx = np.mean(fx1["amp"] / np.sqrt(betax0_bpms))
        meas_avg_sqrt2Jy = np.mean(fy1["amp"] / np.sqrt(betay0_bpms))

    x0 = meas_avg_sqrt2Jx * np.sqrt(betax0_inj)
    y0 = meas_avg_sqrt2Jy * np.sqrt(betay0_inj)
    px0 = py0 = 0.0

    sqrt2Jx_unscaled, sqrt2Jy_unscaled = _get_sqrt2J_from_ps_coords(
        x0, px0, y0, py0, Ascrinv
    )


def unwrap_montonically_increasing(phase_rad, tune_above_half=False):

    two_pi = 2.0 * np.pi

    if tune_above_half:
        phase_rad = two_pi - phase_rad

    phase_rad = np.unwrap(phase_rad)

    while True:
        decreasing = np.diff(phase_rad) < 0.0
        if not np.any(decreasing):
            break

        i_start = np.where(decreasing)[0][0] + 1

        phase_rad[i_start:] += 2 * np.pi

    return phase_rad


def wrap_within_pi(phase_rad):

    two_pi = 2.0 * np.pi

    try:
        len(phase_rad)

        while np.any(phase_rad > np.pi):
            inds = phase_rad > np.pi
            phase_rad[inds] -= two_pi
        while np.any(phase_rad < -np.pi):
            inds = phase_rad < -np.pi
            phase_rad[inds] += two_pi

    except:
        while (phase_rad > np.pi) or (phase_rad < -np.pi):
            if phase_rad > np.pi:
                phase_rad -= two_pi
            elif phase_rad < -np.pi:
                phase_rad += two_pi

    return phase_rad
