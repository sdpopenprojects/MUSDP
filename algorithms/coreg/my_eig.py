import numpy as np
from scipy.linalg import eigh
from sklearn.utils.extmath import svd_flip


def myeigh_v1(A, B=None, rank=None, eval_descending=True):
    r"""
    Solves a symmetric eigenvector or genealized eigenvector problem.

        .. math:
            A v = \lambda v

    or

        .. math:
            A v = \labmda B v

    where A (and B) are symmetric (hermetian).

    Parameters
    ----------
    A: array-like, shape (n x n)
        The LHS matrix.

    B: None, array-like, shape (n x n)
        The (optional) RHS matrix.

    rank: None, int
        Number of components to compute.

    eval_descending: bool
        Whether or not to compute largest or smallest eigenvalues.
        If True, will compute largest rank eigenvalues and
        eigenvalues are returned in descending order. Otherwise,
        computes smallest eigenvalues and returns them in ascending order.

    Output
    ------
    evals : numpy.ndarray, shape (rank,)
        Solution eigenvalues

    evecs : numpy.ndarray, shape (n, rank)
        Solution eigenvectors
    """

    if rank is not None:
        n_max_evals = A.shape[0]

        if eval_descending:
            eigvals_idxs = (n_max_evals - rank, n_max_evals - 1)
        else:
            eigvals_idxs = (0, rank - 1)
    else:
        eigvals_idxs = None

    evals, evecs = eigh(a=A, b=B, subset_by_index=eigvals_idxs)

    if eval_descending:
        ev_reordering = np.argsort(-evals)
        evals = evals[ev_reordering]
        evecs = evecs[:, ev_reordering]

    evecs = svd_flip(evecs, evecs.T)[0]

    return evals, evecs


def myeigh_v2(A):
    d, u = np.linalg.eigh(A)
    # d, u = np.linalg.eig(A)

    d_reordering = np.argsort(-d)
    u = u[:, d_reordering]
    d = d[d_reordering]

    d = d.real
    u = u.real

    return d, u
