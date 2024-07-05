import numpy as np


def principal_angles(A, B, rowspace=False, grassmann_distance=True):
    """Calculate principal angles and Grassmann distance of two subspaces
    A, B: (n, p) and (n, q) matrices whose column spaces are subspaces of R^n.
    rowspace: whether use row space of A, B instead, when A, B are (p, n), (q, n).
    grassmann_distance: whether return the Grassmann distance, i.e., L2 norm of
        the principal angles. The principal angles range between [0, pi/2].
    Return: vector of principal angles of size min(p, q), Grassmann distance
    """
    axis = 1 if rowspace else 0
    A = A / np.linalg.norm(A, axis=axis, keepdims=True)
    B = B / np.linalg.norm(B, axis=axis, keepdims=True)
    sig = np.linalg.svd(A @ B.T if rowspace else A.T @ B, compute_uv=False)
    P_angles = np.arccos(sig)
    if grassmann_distance:
        dist = np.linalg.norm(P_angles)
    return (P_angles, dist) if grassmann_distance else P_angles


def phase_locking_value(phase, unbiased=True, axis=-1):
    """Calculate phase locking value given list of phases of each unit
    phase: phase values of a single unit or list of phase values of units
        Phase values of each unit can be nd-array where spikes are along
        the specified axis. Arrays for different units must have the same
        shape except the specified axis.
    """
    single = isinstance(phase[0], float)
    if single:
        phase = [phase]
    N = []
    resultant = []
    for pha in phase:
        pha = np.asarray(pha)
        shape = pha.shape
        N.append(shape[axis])
        resultant.append(np.sum(np.exp(1j * pha), axis=axis))
    N = np.reshape(N, [-1] + [1] * (len(shape) - 1))
    resultant = np.stack(resultant)
    shape = list(shape)
    shape.pop(axis)
    plv = np.zeros([N.size] + shape)
    idx = np.nonzero(N > 1)[0]
    if unbiased:
        plv2 = (resultant[idx] * resultant[idx].conj()).real / N[idx]
        plv[idx] = (np.fmax(plv2 - 1, 0) / (N[idx] - 1)) ** 0.5
    else:
        plv[idx] = np.abs(resultant[idx]) / N[idx]
    if single:
        plv = plv[0]
    return plv
