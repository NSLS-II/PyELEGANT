import numpy as np

#----------------------------------------------------------------------
def calcSVD(M):
    """"""

    U, sv, VT = np.linalg.svd(M, full_matrices=0, compute_uv=1)

    return U, sv, VT

#----------------------------------------------------------------------
def calcTruncSVMatrix(sv, rcond=1e-6, nsv=None, disp=0):
    """
    Returns a truncated singluar value matrix for the given singular value
    vector "sv", which must be a 1-D vector.

    "rcond" will override "nsv", if both are specified. If both of them are
    `None`, all singular values will be kept.
    """

    norm_sv = sv / sv[0]

    n = len(sv)

    if disp >= 3:
        print('\n* Normalized Singular Values:')
        print(norm_sv)
        print(' ')

    if (disp >= 2) and np.any(sv == 0.0):
        print('### WARNING ### Zero singular values detected!')
        n_zero = np.where(sv == 0.0)[0].size
        n_nonzero = n - n_zero
        print(f'Number of non-zero singular values = {n_nonzero:d}')
        print(f'Number of     zero singular values = {n_zero:d}')
        print(' ')

    if rcond is not None:
        nsv = np.sum(norm_sv >= rcond)

        if (disp >= 2) and (np.min(norm_sv) < rcond):
            print(('# Info # Near-zero normalized singular values '
                   f'(<{rcond:.3e}) detected!'))
            n_ok = np.sum(norm_sv >= rcond)
            n_notok = n - n_ok
            print(f'Number of above-threshold singular values = {n_ok:d}')
            print(f'Number of below-threshold singular values = {n_notok:d}')
            print(' ')

    if nsv is None: nsv = n

    S_inv_trunc = np.zeros((n,n))
    S_inv_trunc_square = np.diagflat(1.0/sv[:nsv])
    S_inv_trunc[:nsv, :nsv] = S_inv_trunc_square

    if (disp >= 1) and (nsv != n):
        print(f'* Using only {nsv:d} out of {n:d} singular values.')

    return S_inv_trunc

#----------------------------------------------------------------------
def getPseudoInv(U, Sinv, VT):
    """
    """

    return VT.T @ Sinv @ U.T
