import numpy as np

def apply_window(array, window_type):
    """
    """

    array = np.array(array)
    n = array.size

    if window_type == 'sine':
        sin = np.sin(np.linspace(0, n-1, n) / (n-1) * np.pi)
        return sin * array
    elif window_type == 'rect':
        return array
    elif window_type == 'sine_squared':
        sin = np.sin(np.linspace(0, n-1, n) / (n-1) * np.pi)
        return sin * sin * array
    else:
        print(' ')
        print('Valid window_type strings are:')
        print('   "rect", "sine", "sine_squared"')
        print(' ')
        raise ValueError(f'Unexpected window_type: {window_type}')

def dft(x, req_tune):
    """
    """

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
        xa,xb,xc,fa,fb,fc,funcalls = bracket(func, args=args)
    elif len(brack) == 2:
        xa,xb,xc,fa,fb,fc,funcalls = bracket(func, xa=brack[0], xb=brack[1], args=args)
    elif len(brack) == 3:
        xa,xb,xc = brack
        if (xa > xc):  # swap so xa < xc can be assumed
            dum = xa; xa=xc; xc=dum
        if not ((xa < xb) and (xb < xc)):
            raise ValueError("Not a bracketing interval.")
        fa = func(*((xa,)+args))
        fb = func(*((xb,)+args))
        fc = func(*((xc,)+args))
        if not ((fb<fa) and (fb < fc)):
            raise ValueError("Not a bracketing interval.")
        funcalls = 3
    else:
        raise ValueError("Bracketing interval must be length 2 or 3 sequence.")

    _gR = 0.61803399
    _gC = 1.0-_gR
    x3 = xc
    x0 = xa
    if (abs(xc-xb) > abs(xb-xa)):
        x1 = xb
        x2 = xb + _gC*(xc-xb)
    else:
        x2 = xb
        x1 = xb - _gC*(xb-xa)
    f1 = func(*((x1,)+args))
    f2 = func(*((x2,)+args))
    funcalls += 2
    while abs(x2-x1) > abstol:
    #while (abs(x3-x0) > tol*(abs(x1)+abs(x2))):
        if (f2 < f1):
            x0 = x1; x1 = x2; x2 = _gR*x1 + _gC*x3
            f1 = f2; f2 = func(*((x2,)+args))
        else:
            x3 = x2; x2 = x1; x1 = _gR*x2 + _gC*x0
            f2 = f1; f1 = func(*((x1,)+args))
        funcalls += 1
    if (f1 < f2):
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
    new_nu, xwin, n, amp_corr_fac, dft_peak_nus, dft_peak_As):
    """"""

    dft_win = dft(xwin, new_nu)
    new_A  = np.abs(dft_win) * 2.0 / n * amp_corr_fac

    dft_peak_nus.append(new_nu)
    dft_peak_As.append(new_A)

    return -new_A

def getDftPeak(x_rect, init_nu, window='sine', resolution=1e-8,
               return_fft_spec=True):
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

    if window == 'rect':
        amp_corr_fac = 1.0 / np.sinc(0.0) # = 1.0
    elif window == 'sine':
        amp_corr_fac = 1.0 / \
            ((np.sinc(0.0 + 0.5) + np.sinc(0.0 - 0.5))/2.0) # ~ 0.63662
    elif window == 'sine_squared':
        amp_corr_fac = 1.0 / \
            (np.sinc(0.0) / (1.0 - 0.0**2) / 2.0) # = 0.5

    ffwin = np.fft.fft(xwin)
    A = np.abs(ffwin) * 2.0 / n * amp_corr_fac

    dft_win = dft(xwin, init_nu)
    new_A   = np.abs(dft_win) * 2.0 / n * amp_corr_fac
    peak_inds = np.argsort(np.abs(nus - init_nu))[:2]
    if A[peak_inds[0]] >= A[peak_inds[1]]:
        peak_ind = peak_inds[0]
    else:
        peak_ind = peak_inds[1]
    if A[peak_ind-1] > A[peak_ind+1]:
        dft_peak_nus = np.array([nus[peak_ind], nus[peak_ind-1], init_nu])
        dft_peak_As  = np.array([A[peak_ind]  , A[peak_ind-1]  , new_A])
    else:
        dft_peak_nus = np.array([nus[peak_ind], nus[peak_ind+1], init_nu])
        dft_peak_As  = np.array([A[peak_ind]  , A[peak_ind+1]  , new_A])

    try:
        backup = dict(dft_peak_nus=dft_peak_nus.copy(),
                      dft_peak_As=dft_peak_As.copy())

        if (dft_peak_As[2] < dft_peak_As[0]) and \
           (dft_peak_As[2] < dft_peak_As[1]) and \
           ((dft_peak_nus[0] < dft_peak_nus[2] < dft_peak_nus[1]) or
            (dft_peak_nus[1] < dft_peak_nus[2] < dft_peak_nus[0])):

            dft_peak_nus = dft_peak_nus.tolist()
            dft_peak_As  = dft_peak_As.tolist()
            out = golden(
                _dft_peak_finding_objective, args=(xwin, n, amp_corr_fac, dft_peak_nus, dft_peak_As),
                brack=np.sort(dft_peak_nus), abstol=resolution, full_output=1)
        else:
            raise ValueError('Invalid bracket. Use slower but robust method.')

    except:
        dft_peak_nus = backup['dft_peak_nus']
        dft_peak_As  = backup['dft_peak_As']

        # 25% slower (but more robust) than the one that uses "golden()"
        sort_inds = np.argsort(dft_peak_As)[::-1]
        dft_peak_As   = dft_peak_As[sort_inds]
        dft_peak_nus  = dft_peak_nus[sort_inds]
        dnu = nus[1]-nus[0]
        while dnu > resolution:
            new_nu = dft_peak_nus[0] + dnu/2.0
            dft_win = dft(xwin, new_nu)
            new_A  = np.abs(dft_win) * 2.0 / n * amp_corr_fac
            dft_peak_nus = np.append(dft_peak_nus , new_nu)
            dft_peak_As  = np.append(dft_peak_As  , new_A)
            new_nu = dft_peak_nus[0] - dnu/2.0
            dft_win = dft(xwin, new_nu)
            new_A  = np.abs(dft_win) * 2.0 / n * amp_corr_fac
            dft_peak_nus = np.append(dft_peak_nus , new_nu)
            dft_peak_As  = np.append(dft_peak_As  , new_A)
            sort_inds = np.argsort(dft_peak_As)[::-1]
            dft_peak_As  = dft_peak_As[sort_inds]
            dft_peak_nus = dft_peak_nus[sort_inds]
            dnu /= 2.0

    dft_peak_nu = dft_peak_nus[np.argmax(dft_peak_As)]
    _dft = dft(xwin, dft_peak_nu)
    dft_peak_A = np.abs(_dft) * 2.0 / n * amp_corr_fac
    if np.iscomplex(x_rect[0]): dft_peak_A /= 2.0
    dft_peak_phi = np.angle(_dft)

    out = dict(A=dft_peak_A, nu=dft_peak_nu, phi=dft_peak_phi)
    if return_fft_spec:
        out['fft_nus'] = nus
        out['fft_As'] = A

    return out

