from typing import Any

import numpy as np
from scipy.signal import correlate
from surface_potential_analysis.state_vector.plot import (
    _calculate_total_offsset_multiplications,
)


def calculate_M_using_correlation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Compute the full correlation of A and B
    corr = correlate(A, B, mode="full")

    # The result includes negative lags, so we need to take the second half (i >= 0)
    N = len(A)
    return corr[N - 1 :]  # Take from lag 0 onwards (positive lags)


def calculate_M(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    N = len(A)
    M = np.zeros(N)

    for i in range(N):
        # For each i, calculate the sum over the range 0 to N-i
        M[i] = np.sum(A[: N - i] * B[i:N])

    return M


A = np.array([1, 2, 3, 4])
B = np.array([5, 6, 7, 8])

M0 = calculate_M(A, B)
M1 = calculate_M_using_correlation(A, B)
# print(M0, M1)


def _calculate_total_offsset_multiplications_complex(
    lhs: np.ndarray[Any, np.dtype[Any]],
    rhs: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """Calculate sum_i^N-i A_i B_i+N for all N.

    Parameters
    ----------
    lhs : np.ndarray[Any, np.dtype[np.float64]]
    rhs : np.ndarray[Any, np.dtype[np.float64]]

    Returns
    -------
    np.ndarray[Any, np.dtype[np.float64]]

    """
    re_re = correlate(np.real(lhs), np.real(rhs), mode="full")[lhs.size - 1 :]
    re_im = correlate(np.real(lhs), np.imag(rhs), mode="full")[lhs.size - 1 :]
    im_re = correlate(np.imag(lhs), np.real(rhs), mode="full")[lhs.size - 1 :]
    im_im = correlate(np.imag(lhs), np.imag(rhs), mode="full")[lhs.size - 1 :]
    return re_re - im_im + 1j * (re_im + im_re)


scattered_k = 2 * np.pi / 4
stacked = np.array([[0, 1, 2, 3, 4, 5]])
exp_positions = np.exp(1j * scattered_k * stacked)
# print(stacked[:, 0] - stacked[:, 1])
print(exp_positions)
# convolution_j = \sum_i^N-j e^(ik.x_i+j) e^(-ik.x_i)
convolution = np.apply_along_axis(
    lambda m: _calculate_total_offsset_multiplications(np.conj(m), m),
    axis=1,
    arr=exp_positions,
)
print(convolution)
isf = (convolution) / np.arange(1, 6 + 1)[::-1]  # type: ignore unknown
print(isf)

convolution = np.apply_along_axis(
    lambda m: _calculate_total_offsset_multiplications_complex(np.conj(m), m),
    axis=1,
    arr=exp_positions,
)
isf = (convolution) / np.arange(1, 6 + 1)[::-1]  # type: ignore unknown
print(isf)
# out_basis = EvenlySpacedTimeBasis(basis[1].n, 1, 0, basis[1].dt * (basis[1].n))  # type: ignore unknown
