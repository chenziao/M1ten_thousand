import numpy as np

def jump_u(u, U):
    return U + (1 - U) * u


def drop_u(u, U):
    return (u - U) / (1 - U)


def solve(A11, A12, A21, A22, b1, b2):
    """Solve 2 x 2 linear equations"""
    den = A11 * A22 - A12 * A21
    x1 = (A22 * b1 - A12 * b2) / den
    x2 = (A11 * b2 - A21 * b1) / den
    return x1, x2


def estimate_steady_state(rate, U=0.5, tau_d=0., tau_f=0.):
    rate = np.asarray(rate)
    shape = rate.shape
    rate = rate.ravel()
    assert all(rate > 0)
    if tau_f > 0:
        L = tau_f * U * rate
        m_u = L / (1 + L)
        m_u2 = m_u * (1 - 1 / (L + 2 / (2 - U)))
    else:
        m_u = np.zeros_like(rate)
        m_u2 = m_u.copy()
    if tau_d > 0:
        M = tau_d * U * rate
        if tau_f > 0:
            Ab = dict(
                A11 = 1 + 1 / M,
                A12 = 1 / U - 1,
                A21 = (U - 1) * (1 - (1 / U - 1) * (m_u2 - m_u)),
                A22 = 3 - 2 * U + 1 / L + 1 / M + (1 - U) * (1 / U - 1),
                b1 = (1 + L / U) / (1 + L),
                b2 = U + m_u * ((1 / U - U) * (2 - U) * L + 5 - U) / ((2 - U) * L + 2)
            )
            m_Q, m_Qu = solve(**Ab)
            m_Q2 = M / ((2 - U) * M + 2) * (U + 2 * (1 - U) * (m_u + m_Q) \
                + (1 - U) * (1 / U - 1) * m_u2 + 2 * (1 - U) * (1 - 2 * U) / U * m_Qu)
        else:
            m_Q = M / (1 + M)
            m_Q2 = m_Q * (1 - 1 / (M + 2 / (2 - U)))
            m_Qu = np.zeros_like(rate)
    else:
        m_Q = np.zeros_like(rate)
        m_Qu, m_Q2 = m_Q.copy(), m_Q.copy()
    m_P = jump_u(m_u, U=U) - U * m_Q - (1 - U) * m_Qu
    vars = dict(m_u=m_u, m_u2=m_u2, m_Q=m_Q, m_Q2=m_Q2, m_Qu=m_Qu, m_P=m_P)
    vars = {k: np.reshape(v, shape) for k, v in vars.items()}
    return vars