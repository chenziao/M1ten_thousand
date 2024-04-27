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
