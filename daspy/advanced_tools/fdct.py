# Purpose: Fast Discrete Curvelet Transform
# Author: Minzhe Hu
# Date: 2024.4.11
# Email: hmz2018@mail.ustc.edu.cn
# Modified from
# http://www.curvelet.org/download-secure.php?file=CurveLab-2.1.3.tar.gz
# (matlab version)
import numpy as np
from numpy.fft import fftshift, ifftshift, fft2, ifft2


def _round(x):
    return np.round(x).astype(int)


def _floor(x):
    return np.floor(x).astype(int)


def _ceil(x):
    return np.ceil(x).astype(int)


def fdct_wrapping_window(x):
    """
    Creates the two halves of a C**inf compactly supported window.

    :param x: vector or matrix of abscissae, the relevant ones from 0 to 1.
    :return: vector or matrix containing samples of the left, resp. right half
        of the window.
    """

    # Initialize the variables
    wr = np.zeros_like(x)
    wl = np.zeros_like(x)

    # Set values close to zero to zero
    x[np.abs(x) < 2**-52] = 0

    # Calculate wr and wl
    wr[(x > 0) & (x < 1)] = np.exp(
        1 - 1. / (1 - np.exp(1 - 1. / x[(x > 0) & (x < 1)])))
    wr[x <= 0] = 1
    wl[(x > 0) & (x < 1)] = np.exp(
        1 - 1. / (1 - np.exp(1 - 1. / (1 - x[(x > 0) & (x < 1)]))))
    wl[x >= 1] = 1

    # Normalize wr and wl
    normalization = np.sqrt(wl**2 + wr**2)
    wr = wr / normalization
    wl = wl / normalization

    return wl, wr


def fdct_wrapping(x, is_real=False, finest=2,
                  nbscales=None, nbangles_coarse=16):
    """
    Fast Discrete Curvelet Transform via wedge wrapping.

    :param x: np.array. M-by-N matrix.
    :param is_real: bool. Type of the transform, False for complex-valued
        curvelets and True for real-valued curvelets.
    :param finest: int. Chooses one of two possibilities for the coefficients at
        the finest level: 1 for curvelets and 2 for wavelets.
    :param nbscales: int. Number of scales including the coarsest wavele
        level. Default set to ceil(log2(min(M,N)) - 3).
    :param nbangles_coarse: int. Number of angles at the 2nd coarsest level,
        minimum 8, must be a multiple of 4.
    :return: 2-D list of np.ndarray. Array of curvelet coefficients.
        C[j][l][k1,k2] is the coefficient at scale j(from finest to coarsest
        scale), angle l(starts at the top-left corner and increases clockwise),
        position k1, k2(size varies with j and l). If is_real is 1, there are
        two types of curvelets, 'cosine' and 'sine'. For a given scale j, the
        'cosine' coefficients are stored in the first two quadrants (low values
        of l), the 'sine' coefficients in the last two quadrants (high values of
        l).
    """
    X = fftshift(fft2(ifftshift(x))) / np.sqrt(x.size)
    N1, N2 = X.shape
    if nbscales is None:
        nbscales = _ceil(np.log2(min(N1, N2)) - 3)

    # Initialization: data structure
    nbangles = [1] + [nbangles_coarse * 2 ** ((nbscales - i) // 2)
                      for i in range(nbscales, 1, -1)]
    if finest == 2:
        nbangles[-1] = 1

    C = []
    for j in range(nbscales):
        C.append([None] * nbangles[j])

    # Loop: pyramidal scale decomposition
    M1 = N1 / 3
    M2 = N2 / 3

    if finest == 1:
        # Initialization: smooth periodic extension of high frequencies
        bigN1 = 2 * _floor(2 * M1) + 1
        bigN2 = 2 * _floor(2 * M2) + 1
        equiv_index_1 = (_floor(N1 / 2) - _floor(2 * M1) +
                         np.arange(bigN1)) % N1
        equiv_index_2 = (_floor(N2 / 2) - _floor(2 * M2) +
                         np.arange(bigN2)) % N2
        X = X[np.ix_(equiv_index_1, equiv_index_2)]

        window_length_1 = _floor(2 * M1) - _floor(M1) - (N1 % 3 == 0)
        window_length_2 = _floor(2 * M2) - _floor(M2) - (N2 % 3 == 0)
        coord_1 = np.linspace(0, 1, window_length_1)
        coord_2 = np.linspace(0, 1, window_length_2)
        wl_1, wr_1 = fdct_wrapping_window(coord_1)
        wl_2, wr_2 = fdct_wrapping_window(coord_2)

        lowpass_1 = np.concatenate((wl_1, np.ones(2 * _floor(M1) + 1), wr_1))
        if N1 % 3 == 0:
            lowpass_1 = np.concatenate(([0], lowpass_1, [0]))

        lowpass_2 = np.concatenate((wl_2, np.ones(2 * _floor(M2) + 1), wr_2))
        if N2 % 3 == 0:
            lowpass_2 = np.concatenate(([0], lowpass_2, [0]))

        lowpass = np.outer(lowpass_1, lowpass_2)
        Xlow = X * lowpass
        scales = np.arange(nbscales, 1, -1)

    else:
        M1 /= 2
        M2 /= 2

        window_length_1 = _floor(2 * M1) - _floor(M1)
        window_length_2 = _floor(2 * M2) - _floor(M2)
        coord_1 = np.linspace(0, 1, window_length_1)
        coord_2 = np.linspace(0, 1, window_length_2)
        wl_1, wr_1 = fdct_wrapping_window(coord_1)
        wl_2, wr_2 = fdct_wrapping_window(coord_2)

        lowpass_1 = np.concatenate((wl_1, np.ones(2 * _floor(M1) + 1), wr_1))
        lowpass_2 = np.concatenate((wl_2, np.ones(2 * _floor(M2) + 1), wr_2))
        lowpass = np.outer(lowpass_1, lowpass_2)
        hipass = np.sqrt(1 - lowpass ** 2)

        Xlow_index_1 = np.arange(-_floor(2 * M1),
                                 _floor(2 * M1) + 1) + _ceil((N1 + 1) / 2) - 1
        Xlow_index_2 = np.arange(-_floor(2 * M2),
                                 _floor(2 * M2) + 1) + _ceil((N2 + 1) / 2) - 1
        Xlow = X[np.ix_(Xlow_index_1, Xlow_index_2)] * lowpass
        Xhi = X.copy()
        Xhi[np.ix_(Xlow_index_1, Xlow_index_2)] *= hipass

        C[nbscales - 1][0] = fftshift(ifft2(ifftshift(Xhi))
                                      ) * np.sqrt(Xhi.size)
        if is_real:
            C[nbscales - 1][0] = C[nbscales - 1][0].real

        scales = np.arange(nbscales - 1, 1, -1)
    for j in scales - 1:
        M1 /= 2
        M2 /= 2
        window_length_1 = _floor(2 * M1) - _floor(M1)
        window_length_2 = _floor(2 * M2) - _floor(M2)
        coord_1 = np.linspace(0, 1, window_length_1)
        coord_2 = np.linspace(0, 1, window_length_2)
        wl_1, wr_1 = fdct_wrapping_window(coord_1)
        wl_2, wr_2 = fdct_wrapping_window(coord_2)

        lowpass_1 = np.concatenate((wl_1, np.ones(2 * _floor(M1) + 1), wr_1))
        lowpass_2 = np.concatenate((wl_2, np.ones(2 * _floor(M2) + 1), wr_2))
        lowpass = np.outer(lowpass_1, lowpass_2)
        hipass = np.sqrt(1 - lowpass ** 2)

        Xhi = Xlow.copy()
        Xlow_index_1 = np.arange(-_floor(2 * M1),
                                 _floor(2 * M1) + 1) + _floor(4 * M1)
        Xlow_index_2 = np.arange(-_floor(2 * M2),
                                 _floor(2 * M2) + 1) + _floor(4 * M2)
        Xlow = Xlow[np.ix_(Xlow_index_1, Xlow_index_2)]
        Xhi[np.ix_(Xlow_index_1, Xlow_index_2)] = Xlow * hipass
        Xlow *= lowpass

        # Loop: angular decomposition
        l = -1
        nbquadrants = 2 + 2 * (not is_real)
        nbangles_perquad = nbangles[j] // 4
        for quadrant in range(1, nbquadrants + 1):
            M_horiz = (M1, M2)[quadrant % 2]
            M_vert = (M2, M1)[quadrant % 2]
            wedge_ticks_left = _round(
                np.linspace(
                    0,
                    1,
                    nbangles_perquad +
                    1) *
                _floor(
                    4 *
                    M_horiz) +
                1)
            wedge_ticks_right = 2 * _floor(4 * M_horiz) + 2 - wedge_ticks_left
            if nbangles_perquad % 2:
                wedge_ticks = np.concatenate(
                    (wedge_ticks_left, wedge_ticks_right[::-1]))
            else:
                wedge_ticks = np.concatenate(
                    (wedge_ticks_left, wedge_ticks_right[-2::-1]))

            wedge_endpoints = wedge_ticks[1:-1:2]
            wedge_midpoints = (wedge_endpoints[:-1] + wedge_endpoints[1:]) / 2
            # Left corner wedge
            l += 1
            first_wedge_endpoint_vert = _round(
                _floor(4 * M_vert) / nbangles_perquad + 1)
            length_corner_wedge = _floor(4 * M_vert) - _floor(M_vert) + \
                _ceil(first_wedge_endpoint_vert / 4)
            Y_corner = np.arange(length_corner_wedge) + 1
            XX, YY = np.meshgrid(
                np.arange(2 * _floor(4 * M_horiz) + 1) + 1, Y_corner)
            width_wedge = wedge_endpoints[1] + wedge_endpoints[0] - 1
            slope_wedge = (_floor(4 * M_horiz) + 1 -
                           wedge_endpoints[0]) / _floor(4 * M_vert)
            left_line = _round(
                2 - wedge_endpoints[0] + slope_wedge * (Y_corner - 1))
            wrapped_data = np.zeros(
                (length_corner_wedge, width_wedge), dtype=complex)
            wrapped_XX = np.zeros(
                (length_corner_wedge, width_wedge), dtype=int)
            wrapped_YY = np.zeros(
                (length_corner_wedge, width_wedge), dtype=int)
            first_row = _floor(4 * M_vert) + 2 - \
                _ceil((length_corner_wedge + 1) / 2) + \
                (length_corner_wedge + 1) % 2 * (quadrant - 2 == quadrant % 2)
            first_col = _floor(4 * M_horiz) + 2 - _ceil((width_wedge + 1) / 2) \
                + (width_wedge + 1) % 2 * (quadrant - 3 == (quadrant - 3) % 2)
            for row in Y_corner - 1:
                cols = left_line[row] + \
                    (np.arange(width_wedge) - (left_line[row] - first_col)) \
                    % width_wedge
                admissible_cols = _round(0.5 * (cols + 1 + abs(cols - 1))) - 1
                new_row = (row - first_row + 1) % length_corner_wedge
                wrapped_data[new_row, :] = Xhi[row,
                                               admissible_cols] * (cols > 0)
                wrapped_XX[new_row, :] = XX[row, admissible_cols]
                wrapped_YY[new_row, :] = YY[row, admissible_cols]
            slope_wedge_right = (_floor(4 * M_horiz) + 1 -
                                 wedge_midpoints[0]) / _floor(4 * M_vert)
            mid_line_right = wedge_midpoints[0] + \
                slope_wedge_right * (wrapped_YY - 1)
            coord_right = 0.5 + _floor(4 * M_vert) / \
                (wedge_endpoints[1] - wedge_endpoints[0]) * \
                (wrapped_XX - mid_line_right) / \
                (_floor(4 * M_vert) + 1 - wrapped_YY)
            C2 = 1 / (1 / (2 * (_floor(4 * M_horiz)) / (wedge_endpoints[0] -
                1) - 1) + 1 / (2 * (_floor(4 * M_vert)) / (
                first_wedge_endpoint_vert - 1) - 1))
            C1 = C2 / (2 * (_floor(4 * M_vert)) /
                       (first_wedge_endpoint_vert - 1) - 1)
            wrapped_XX[(wrapped_XX - 1) / _floor(4 * M_horiz) +
                       (wrapped_YY - 1) / _floor(4 * M_vert) == 2] += 1
            coord_corner = C1 + C2 * ((wrapped_XX - 1) / _floor(4 * M_horiz) -
                (wrapped_YY - 1) / _floor(4 * M_vert)) / (2 -
                ((wrapped_XX - 1) / _floor(4 * M_horiz) + (wrapped_YY - 1) /
                _floor(4 * M_vert)))
            wl_left, _ = fdct_wrapping_window(coord_corner)
            _, wr_right = fdct_wrapping_window(coord_right)
            wrapped_data = wrapped_data * wl_left * wr_right
            if not is_real:
                wrapped_data = np.rot90(wrapped_data, -(quadrant - 1))
                C[j][l] = fftshift(ifft2(ifftshift(wrapped_data))) * \
                    np.sqrt(wrapped_data.size)
            else:
                wrapped_data = np.rot90(wrapped_data, -(quadrant - 1))
                x = fftshift(ifft2(ifftshift(wrapped_data))) * \
                    np.sqrt(wrapped_data.size)
                C[j][l] = np.sqrt(2) * x.real
                C[j][l + nbangles[j] // 2] = np.sqrt(2) * x.imag

            # Regular wedges
            length_wedge = _floor(4 * M_vert) - _floor(M_vert)
            Y = np.arange(length_wedge) + 1
            first_row = _floor(4 * M_vert) + 2 - _ceil((length_wedge + 1) / 2) \
                + (length_wedge + 1) % 2 * (quadrant - 2 == quadrant % 2)
            for subl in range(1, nbangles_perquad - 1):
                l += 1
                width_wedge = wedge_endpoints[subl +
                                              1] - wedge_endpoints[subl - 1] + 1
                slope_wedge = ((_floor(4 * M_horiz) + 1) -
                               wedge_endpoints[subl]) / _floor(4 * M_vert)
                left_line = _round(
                    wedge_endpoints[subl - 1] + slope_wedge * (Y - 1))
                wrapped_data = np.zeros(
                    (length_wedge, width_wedge), dtype=complex)
                wrapped_XX = np.zeros((length_wedge, width_wedge), dtype=int)
                wrapped_YY = np.zeros((length_wedge, width_wedge), dtype=int)
                first_col = _floor(4 * M_horiz) + 2 - \
                    _ceil((width_wedge + 1) / 2) + \
                    (width_wedge + 1) % 2 * (quadrant - 3 == (quadrant - 3) % 2)
                for row in Y - 1:
                    cols = left_line[row] + (np.arange(width_wedge) -
                        (left_line[row] - first_col)) % width_wedge - 1
                    new_row = (row - first_row + 1) % length_wedge
                    wrapped_data[new_row, :] = Xhi[row, cols]
                    wrapped_XX[new_row, :] = XX[row, cols]
                    wrapped_YY[new_row, :] = YY[row, cols]
                slope_wedge_left = ((_floor(4 * M_horiz) + 1) -
                    wedge_midpoints[subl - 1]) / _floor(4 * M_vert)
                mid_line_left = wedge_midpoints[subl - 1] + \
                    slope_wedge_left * (wrapped_YY - 1)
                coord_left = 0.5 + _floor(4 * M_vert) / \
                    (wedge_endpoints[subl] - wedge_endpoints[subl - 1]) * \
                    (wrapped_XX - mid_line_left) / \
                    (_floor(4 * M_vert) + 1 - wrapped_YY)
                slope_wedge_right = ((_floor(4 * M_horiz) + 1) -
                                     wedge_midpoints[subl]) / _floor(4 * M_vert)
                mid_line_right = wedge_midpoints[subl] + \
                    slope_wedge_right * (wrapped_YY - 1)
                coord_right = 0.5 + _floor(4 * M_vert) / \
                    (wedge_endpoints[subl + 1] - wedge_endpoints[subl]) * \
                    (wrapped_XX - mid_line_right) / \
                    (_floor(4 * M_vert) + 1 - wrapped_YY)

                wl_left, _ = fdct_wrapping_window(coord_left)
                _, wr_right = fdct_wrapping_window(coord_right)
                wrapped_data = wrapped_data * wl_left * wr_right
                if not is_real:
                    wrapped_data = np.rot90(wrapped_data, -(quadrant - 1))
                    C[j][l] = fftshift(ifft2(ifftshift(wrapped_data))) * \
                        np.sqrt(wrapped_data.size)
                else:
                    wrapped_data = np.rot90(wrapped_data, -(quadrant - 1))
                    x = fftshift(ifft2(ifftshift(wrapped_data))) * \
                        np.sqrt(wrapped_data.size)
                    C[j][l] = np.sqrt(2) * x.real
                    C[j][l + nbangles[j] // 2] = np.sqrt(2) * x.imag

            # Right corner wedge
            l += 1
            width_wedge = 4 * _floor(4 * M_horiz) + 3 - \
                wedge_endpoints[-1] - wedge_endpoints[-2]
            slope_wedge = ((_floor(4 * M_horiz) + 1) -
                           wedge_endpoints[-1]) / _floor(4 * M_vert)
            left_line = _round(
                wedge_endpoints[-2] + slope_wedge * (Y_corner - 1))
            wrapped_data = np.zeros((length_corner_wedge, width_wedge),
                                    dtype=complex)
            wrapped_XX = np.zeros((length_corner_wedge, width_wedge), dtype=int)
            wrapped_YY = np.zeros((length_corner_wedge, width_wedge), dtype=int)
            first_row = _floor(4 * M_vert) + 2 - \
                _ceil((length_corner_wedge + 1) / 2) + \
                (length_corner_wedge + 1) % 2 * (quadrant - 2 == quadrant % 2)
            first_col = _floor(4 * M_horiz) + 2 - _ceil((width_wedge + 1) / 2) + \
                (width_wedge + 1) % 2 * (quadrant - 3 == (quadrant - 3) % 2)
            for row in Y_corner - 1:
                cols = left_line[row] + (np.arange(width_wedge) -
                    (left_line[row] - first_col)) % width_wedge
                admissible_cols = _round(0.5 * (cols + 2 * _floor(4 * M_horiz)
                    + 1 - np.abs(cols - (2 * _floor(4 * M_horiz) + 1)))) - 1
                new_row = (row - first_row + 1) % length_corner_wedge
                wrapped_data[new_row, :] = Xhi[row, admissible_cols] * \
                    (cols <= (2 * _floor(4 * M_horiz) + 1))
                wrapped_XX[new_row, :] = XX[row, admissible_cols]
                wrapped_YY[new_row, :] = YY[row, admissible_cols]

            slope_wedge_left = ((_floor(4 * M_horiz) + 1) -
                                wedge_midpoints[-1]) / _floor(4 * M_vert)
            mid_line_left = wedge_midpoints[-1] + \
                slope_wedge_left * (wrapped_YY - 1)
            coord_left = 0.5 + _floor(4 * M_vert) / \
                (wedge_endpoints[-1] - wedge_endpoints[-2]) * \
                (wrapped_XX - mid_line_left) / \
                (_floor(4 * M_vert) + 1 - wrapped_YY)
            C2 = -1 / (2 * (_floor(4 * M_horiz)) / (wedge_endpoints[-1] - 1) -
                       1 + 1 / (2 * (_floor(4 * M_vert)) /
                                (first_wedge_endpoint_vert - 1) - 1))
            C1 = -C2 * (2 * (_floor(4 * M_horiz)) /
                        (wedge_endpoints[-1] - 1) - 1)
            wrapped_XX[(wrapped_XX - 1) / _floor(4 * M_horiz) ==
                       (wrapped_YY - 1) / _floor(4 * M_vert)] -= 1
            coord_corner = C1 + C2 * (2 - ((wrapped_XX - 1) /
                _floor(4 * M_horiz) + (wrapped_YY - 1) / _floor(4 * M_vert))) \
                / ((wrapped_XX - 1) / _floor(4 * M_horiz) - (wrapped_YY - 1) /
                _floor(4 * M_vert))
            wl_left, _ = fdct_wrapping_window(coord_left)
            _, wr_right = fdct_wrapping_window(coord_corner)
            wrapped_data = wrapped_data * wl_left * wr_right
            if not is_real:
                wrapped_data = np.rot90(wrapped_data, -(quadrant - 1))
                C[j][l] = fftshift(ifft2(ifftshift(wrapped_data))
                                   ) * np.sqrt(wrapped_data.size)
            else:
                wrapped_data = np.rot90(wrapped_data, -(quadrant - 1))
                x = fftshift(ifft2(ifftshift(wrapped_data))) * \
                    np.sqrt(wrapped_data.size)
                C[j][l] = np.sqrt(2) * x.real
                C[j][l + nbangles[j] // 2] = np.sqrt(2) * x.imag

            if quadrant < nbquadrants:
                Xhi = np.rot90(Xhi)
    # Coarsest wavelet level
    C[0][0] = fftshift(ifft2(ifftshift(Xlow))) * np.sqrt(Xlow.size)
    if is_real:
        C[0][0] = C[0][0].real

    return C


def ifdct_wrapping(C, is_real=False, size=None):
    """
    Inverse Fast Discrete Curvelet Transform via wedge wrapping. This is in fact
    the adjoint, also the pseudo-inverse

    :param C: 2-D list of np.ndarray. Array of curvelet coefficients.
    :param is_real: bool. Type of the transform, False for complex-valued
        curvelets and True for real-valued curvelets.
    :param size: tuple of ints. Size of the image to be recovered (not necessary
        if finest = 2)
    :return: 2-D np.ndarray.
    """
    nbscales = len(C)
    nbangles_coarse = len(C[1])
    nbangles = [1] + [nbangles_coarse * 2 ** ((nbscales - i) // 2)
                      for i in range(nbscales, 1, -1)]
    if len(C[-1]) == 1:
        finest = 2
        nbangles[nbscales - 1] = 1
    else:
        finest = 1

    if size is None:
        if finest == 1:
            raise ValueError("Require output size.")
        else:
            N1, N2 = C[-1][0].shape
    else:
        N1, N2 = size

    M1 = N1 / 3
    M2 = N2 / 3

    if finest == 1:
        # Initialization: preparing the lowpass filter at finest scale
        window_length_1 = _floor(2 * M1) - _floor(M1) - (N1 % 3 == 0)
        window_length_2 = _floor(2 * M2) - _floor(M2) - (N2 % 3 == 0)
        coord_1 = np.linspace(0, 1, window_length_1)
        coord_2 = np.linspace(0, 1, window_length_2)
        wl_1, wr_1 = fdct_wrapping_window(coord_1)
        wl_2, wr_2 = fdct_wrapping_window(coord_2)

        lowpass_1 = np.concatenate((wl_1, np.ones(2 * _floor(M1) + 1), wr_1))
        if N1 % 3 == 0:
            lowpass_1 = np.concatenate(([0], lowpass_1, [0]))

        lowpass_2 = np.concatenate((wl_2, np.ones(2 * _floor(M2) + 1), wr_2))
        if N2 % 3 == 0:
            lowpass_2 = np.concatenate(([0], lowpass_2, [0]))

        lowpass = np.outer(lowpass_1, lowpass_2)
        scales = np.arange(nbscales, 1, -1)
    else:
        M1 /= 2
        M2 /= 2

        window_length_1 = _floor(2 * M1) - _floor(M1)
        window_length_2 = _floor(2 * M2) - _floor(M2)
        coord_1 = np.linspace(0, 1, window_length_1)
        coord_2 = np.linspace(0, 1, window_length_2)
        wl_1, wr_1 = fdct_wrapping_window(coord_1)
        wl_2, wr_2 = fdct_wrapping_window(coord_2)

        lowpass_1 = np.concatenate((wl_1, np.ones(2 * _floor(M1) + 1), wr_1))
        lowpass_2 = np.concatenate((wl_2, np.ones(2 * _floor(M2) + 1), wr_2))
        lowpass = np.outer(lowpass_1, lowpass_2)
        hipass_finest = np.sqrt(1 - lowpass ** 2)

        scales = np.arange(nbscales - 1, 1, -1)

    bigN1 = 2 * _floor(2 * M1) + 1
    bigN2 = 2 * _floor(2 * M2) + 1
    X = np.zeros((bigN1, bigN2), dtype=complex)

    # Loop: pyramidal reconstruction

    Xj_topleft_1 = 1
    Xj_topleft_2 = 1
    for j in scales - 1:
        M1 /= 2
        M2 /= 2

        window_length_1 = _floor(2 * M1) - _floor(M1)
        window_length_2 = _floor(2 * M2) - _floor(M2)
        coord_1 = np.linspace(0, 1, window_length_1)
        coord_2 = np.linspace(0, 1, window_length_2)
        wl_1, wr_1 = fdct_wrapping_window(coord_1)
        wl_2, wr_2 = fdct_wrapping_window(coord_2)

        lowpass_1 = np.concatenate((wl_1, np.ones(2 * _floor(M1) + 1), wr_1))
        lowpass_2 = np.concatenate((wl_2, np.ones(2 * _floor(M2) + 1), wr_2))
        lowpass_next = np.outer(lowpass_1, lowpass_2)
        hipass = np.sqrt(1 - lowpass_next ** 2)
        Xj = np.zeros((2 * _floor(4 * M1) + 1, 2 * _floor(4 * M2) + 1),
                      dtype=complex)

        # Loop: angles
        l = -1
        nbquadrants = 2 + 2 * (not is_real)
        nbangles_perquad = nbangles[j] // 4
        for quadrant in range(1, nbquadrants + 1):
            M_horiz = (M1, M2)[quadrant % 2]
            M_vert = (M2, M1)[quadrant % 2]
            wedge_ticks_left = _round(np.linspace(0, 1, nbangles_perquad + 1) *
                                      _floor(4 * M_horiz) + 1)
            wedge_ticks_right = 2 * _floor(4 * M_horiz) + 2 - wedge_ticks_left
            if nbangles_perquad % 2:
                wedge_ticks = np.concatenate(
                    (wedge_ticks_left, wedge_ticks_right[::-1]))
            else:
                wedge_ticks = np.concatenate(
                    (wedge_ticks_left, wedge_ticks_right[-2::-1]))
            wedge_endpoints = wedge_ticks[1:-1:2]
            wedge_midpoints = (wedge_endpoints[:-1] + wedge_endpoints[1:]) / 2

            # Left corner wedge
            l += 1
            first_wedge_endpoint_vert = _round(_floor(4 * M_vert) /
                                               nbangles_perquad + 1)
            length_corner_wedge = _floor(4 * M_vert) - _floor(M_vert) + \
                _ceil(first_wedge_endpoint_vert / 4)
            Y_corner = np.arange(length_corner_wedge) + 1
            [XX, YY] = np.meshgrid(np.arange(1, 2 * _floor(4 * M_horiz) + 2),
                                   Y_corner)
            width_wedge = wedge_endpoints[1] + wedge_endpoints[0] - 1
            slope_wedge = (_floor(4 * M_horiz) + 1 -
                           wedge_endpoints[0]) / _floor(4 * M_vert)
            left_line = _round(2 - wedge_endpoints[0] +
                               slope_wedge * (Y_corner - 1))
            wrapped_XX = np.zeros((length_corner_wedge, width_wedge), dtype=int)
            wrapped_YY = np.zeros((length_corner_wedge, width_wedge), dtype=int)
            first_row = _floor(4 * M_vert) + \
                2 - _ceil((length_corner_wedge + 1) / 2) + \
                (length_corner_wedge + 1) % 2 * (quadrant - 2 == quadrant % 2)
            first_col = _floor(4 * M_horiz) + 2 - _ceil((width_wedge + 1) / 2) \
                + (width_wedge + 1) % 2 * (quadrant - 3 == (quadrant - 3) % 2)
            for row in Y_corner - 1:
                cols = left_line[row] + (np.arange(width_wedge) -
                    (left_line[row] - first_col)) % width_wedge
                new_row = (row - first_row + 1) % length_corner_wedge
                admissible_cols = _round(0.5 * (cols + 1 + abs(cols - 1))) - 1
                wrapped_XX[new_row, :] = XX[row, admissible_cols]
                wrapped_YY[new_row, :] = YY[row, admissible_cols]

            slope_wedge_right = (_floor(4 * M_horiz) + 1 - wedge_midpoints[0]) \
                / _floor(4 * M_vert)
            mid_line_right = wedge_midpoints[0] + \
                slope_wedge_right * (wrapped_YY - 1)
            coord_right = 0.5 + _floor(4 * M_vert) / (wedge_endpoints[1] -
                wedge_endpoints[0]) * (wrapped_XX - mid_line_right) / \
                (_floor(4 * M_vert) + 1 - wrapped_YY)
            C2 = 1 / (1 / (2 * (_floor(4 * M_horiz)) /
                (wedge_endpoints[0] - 1) - 1) + 1 / (2 * (_floor(4 * M_vert))
                / (first_wedge_endpoint_vert - 1) - 1))
            C1 = C2 / (2 * (_floor(4 * M_vert)) /
                       (first_wedge_endpoint_vert - 1) - 1)
            wrapped_XX[(wrapped_XX - 1) / _floor(4 * M_horiz) +
                       (wrapped_YY - 1) / _floor(4 * M_vert) == 2] += 1
            coord_corner = C1 + C2 * ((wrapped_XX - 1) / _floor(4 * M_horiz) -
                (wrapped_YY - 1) / _floor(4 * M_vert)) / (2 - ((wrapped_XX - 1)
                / _floor(4 * M_horiz) + (wrapped_YY - 1) / _floor(4 * M_vert)))
            wl_left, _ = fdct_wrapping_window(coord_corner)
            _, wr_right = fdct_wrapping_window(coord_right)

            if not is_real:
                wrapped_data = fftshift(fft2(ifftshift(C[j][l]))) / \
                    np.sqrt(C[j][l].size)
                wrapped_data = np.rot90(wrapped_data, quadrant - 1)
            else:
                x = C[j][l] + 1j * C[j][l + nbangles[j] // 2]
                wrapped_data = fftshift(fft2(ifftshift(x))) / \
                    np.sqrt(x.size * 2)
                wrapped_data = np.rot90(wrapped_data, quadrant - 1)

            wrapped_data = wrapped_data * wl_left * wr_right
            # Unwrapping data
            for row in Y_corner - 1:
                cols = left_line[row] + (np.arange(width_wedge) -
                    (left_line[row] - first_col)) % width_wedge
                admissible_cols = _round(0.5 * (cols + 1 + abs(cols - 1))) - 1
                new_row = (row - first_row + 1) % length_corner_wedge
                Xj[row, admissible_cols] += wrapped_data[new_row, :]
                # We use the following property: in an assignment A(B) = C where
                # B and C are vectors, if some value x repeats in B, then the
                # last occurrence of x is the one corresponding to the eventual
                # assignment.

            # Regular wedges
            length_wedge = _floor(4 * M_vert) - _floor(M_vert)
            Y = np.arange(length_wedge) + 1
            first_row = _floor(4 * M_vert) + 2 - _ceil((length_wedge + 1) / 2) \
                + (length_wedge + 1) % 2 * (quadrant - 2 == quadrant % 2)
            for subl in range(1, nbangles_perquad - 1):
                l += 1
                width_wedge = wedge_endpoints[subl + 1] - \
                    wedge_endpoints[subl - 1] + 1
                slope_wedge = ((_floor(4 * M_horiz) + 1) -
                               wedge_endpoints[subl]) / _floor(4 * M_vert)
                left_line = _round(wedge_endpoints[subl - 1] +
                                   slope_wedge * (Y - 1))
                wrapped_XX = np.zeros((length_wedge, width_wedge), dtype=int)
                wrapped_YY = np.zeros((length_wedge, width_wedge), dtype=int)
                first_col = _floor(4 * M_horiz) + 2 - \
                    _ceil((width_wedge + 1) / 2) + \
                    (width_wedge + 1) % 2 * (quadrant - 3 == (quadrant - 3) % 2)
                for row in Y - 1:
                    cols = left_line[row] + (np.arange(width_wedge) -
                        (left_line[row] - first_col)) % width_wedge - 1
                    new_row = (row - first_row + 1) % length_wedge
                    wrapped_XX[new_row, :] = XX[row, cols]
                    wrapped_YY[new_row, :] = YY[row, cols]

                slope_wedge_left = ((_floor(4 * M_horiz) + 1) -
                    wedge_midpoints[subl - 1]) / _floor(4 * M_vert)
                mid_line_left = wedge_midpoints[subl - 1] + \
                    slope_wedge_left * (wrapped_YY - 1)
                coord_left = 0.5 + _floor(4 * M_vert) / (wedge_endpoints[subl]
                    - wedge_endpoints[subl - 1]) * \
                    (wrapped_XX - mid_line_left) / \
                    (_floor(4 * M_vert) + 1 - wrapped_YY)
                slope_wedge_right = ((_floor(4 * M_horiz) + 1) -
                                     wedge_midpoints[subl]) / _floor(4 * M_vert)
                mid_line_right = wedge_midpoints[subl] + \
                    slope_wedge_right * (wrapped_YY - 1)
                coord_right = 0.5 + _floor(4 * M_vert) / \
                    (wedge_endpoints[subl + 1] - wedge_endpoints[subl]) * \
                    (wrapped_XX - mid_line_right) / \
                    (_floor(4 * M_vert) + 1 - wrapped_YY)
                wl_left, _ = fdct_wrapping_window(coord_left)
                _, wr_right = fdct_wrapping_window(coord_right)
                if not is_real:
                    wrapped_data = fftshift(fft2(ifftshift(C[j][l]))) / \
                        np.sqrt(C[j][l].size)
                    wrapped_data = np.rot90(wrapped_data, quadrant - 1)
                else:
                    x = C[j][l] + 1j * C[j][l + nbangles[j] // 2]
                    wrapped_data = fftshift(
                        fft2(ifftshift(x))) / np.sqrt(x.size * 2)
                    wrapped_data = np.rot90(wrapped_data, quadrant - 1)

                wrapped_data = wrapped_data * wl_left * wr_right

                # Unwrapping data
                for row in Y - 1:
                    cols = left_line[row] + (np.arange(width_wedge) -
                        (left_line[row] - first_col)) % width_wedge - 1
                    new_row = (row + 1 - first_row) % length_wedge
                    Xj[row, cols] += wrapped_data[new_row, :]

            # Right corner wedge
            l += 1
            width_wedge = 4 * _floor(4 * M_horiz) + 3 - \
                wedge_endpoints[-1] - wedge_endpoints[-2]
            slope_wedge = ((_floor(4 * M_horiz) + 1) -
                           wedge_endpoints[-1]) / _floor(4 * M_vert)
            left_line = _round(
                wedge_endpoints[-2] + slope_wedge * (Y_corner - 1))
            wrapped_XX = np.zeros(
                (length_corner_wedge, width_wedge), dtype=int)
            wrapped_YY = np.zeros(
                (length_corner_wedge, width_wedge), dtype=int)
            first_row = _floor(4 * M_vert) + 2 - \
                _ceil((length_corner_wedge + 1) / 2) + \
                (length_corner_wedge + 1) % 2 * (quadrant - 2 == quadrant % 2)
            first_col = _floor(4 * M_horiz) + 2 - _ceil((width_wedge + 1) / 2) \
                + (width_wedge + 1) % 2 * (quadrant - 3 == (quadrant - 3) % 2)
            for row in Y_corner - 1:
                cols = left_line[row] + (np.arange(width_wedge) -
                    (left_line[row] - first_col)) % width_wedge
                admissible_cols = _round(0.5 * (cols + 2 * _floor(4 * M_horiz)
                    + 1 - np.abs(cols - (2 * _floor(4 * M_horiz) + 1)))) - 1
                new_row = (row - first_row + 1) % length_corner_wedge
                wrapped_XX[new_row, :] = XX[row, admissible_cols]
                wrapped_YY[new_row, :] = YY[row, admissible_cols]

            slope_wedge_left = ((_floor(4 * M_horiz) + 1) -
                                wedge_midpoints[-1]) / _floor(4 * M_vert)
            mid_line_left = wedge_midpoints[-1] + \
                slope_wedge_left * (wrapped_YY - 1)
            coord_left = 0.5 + _floor(4 * M_vert) / \
                (wedge_endpoints[-1] - wedge_endpoints[-2]) * \
                (wrapped_XX - mid_line_left) / \
                (_floor(4 * M_vert) + 1 - wrapped_YY)
            C2 = -1 / (2 * (_floor(4 * M_horiz)) / (wedge_endpoints[-1] - 1)
                       - 1 + 1 / (2 * (_floor(4 * M_vert)) /
                                  (first_wedge_endpoint_vert - 1) - 1))
            C1 = -C2 * (2 * (_floor(4 * M_horiz)) /
                        (wedge_endpoints[-1] - 1) - 1)

            wrapped_XX[(wrapped_XX - 1) / _floor(4 * M_horiz) ==
                       (wrapped_YY - 1) / _floor(4 * M_vert)] -= 1
            coord_corner = C1 + C2 * (2 - ((wrapped_XX - 1) /
                _floor(4 * M_horiz) + (wrapped_YY - 1) / _floor(4 * M_vert))) \
                / ((wrapped_XX - 1) / _floor(4 * M_horiz) - (wrapped_YY - 1) /
                _floor(4 * M_vert))
            wl_left, _ = fdct_wrapping_window(coord_left)
            _, wr_right = fdct_wrapping_window(coord_corner)

            if not is_real:
                wrapped_data = fftshift(
                    fft2(ifftshift(C[j][l]))) / np.sqrt(C[j][l].size)
                wrapped_data = np.rot90(wrapped_data, quadrant - 1)
            else:
                x = C[j][l] + 1j * C[j][l + nbangles[j] // 2]
                wrapped_data = fftshift(
                    fft2(ifftshift(x))) / np.sqrt(x.size * 2)
                wrapped_data = np.rot90(wrapped_data, quadrant - 1)

            wrapped_data = wrapped_data * wl_left * wr_right

            # Unwrapping data
            for row in Y_corner - 1:
                cols = left_line[row] + (np.arange(width_wedge) -
                    (left_line[row] - first_col)) % width_wedge
                admissible_cols = _round(1 / 2 * (cols + 2 * _floor(4 * M_horiz)
                    + 1 - abs(cols - (2 * _floor(4 * M_horiz) + 1)))) - 1
                new_row = (row + 1 - first_row) % length_corner_wedge
                Xj[row, np.flip(admissible_cols)] += wrapped_data[new_row, ::-1]
                # We use the following property: in an assignment A[B] = C where
                # B and C are vectors, if some value x repeats in B, then the
                # last occurrence of x is the one corresponding to the eventual
                # assignment.

            Xj = np.rot90(Xj)

        Xj *= lowpass
        Xj_index1 = np.arange(-_floor(2 * M1),
                              _floor(2 * M1) + 1) + _floor(4 * M1)
        Xj_index2 = np.arange(-_floor(2 * M2),
                              _floor(2 * M2) + 1) + _floor(4 * M2)

        Xj[np.ix_(Xj_index1, Xj_index2)] *= hipass

        loc_1 = Xj_topleft_1 + np.arange(2 * _floor(4 * M1) + 1) - 1
        loc_2 = Xj_topleft_2 + np.arange(2 * _floor(4 * M2) + 1) - 1
        X[np.ix_(loc_1, loc_2)] += Xj

        # Preparing for loop reentry or exit
        Xj_topleft_1 += _floor(4 * M1) - _floor(2 * M1)
        Xj_topleft_2 += _floor(4 * M2) - _floor(2 * M2)

        lowpass = lowpass_next

    if is_real:
        Y = X
        X = np.rot90(X, 2)
        X = X + np.conj(Y)

    # Coarsest wavelet level
    M1 = M1 / 2
    M2 = M2 / 2
    Xj = fftshift(fft2(ifftshift(C[0][0]))) / np.sqrt(C[0][0].size)
    loc_1 = Xj_topleft_1 + np.arange(2 * _floor(4 * M1) + 1) - 1
    loc_2 = Xj_topleft_2 + np.arange(2 * _floor(4 * M2) + 1) - 1
    X[np.ix_(loc_1, loc_2)] += Xj * lowpass

    # Finest level
    M1 = N1 / 3
    M2 = N2 / 3
    if finest == 1:
        # Folding back onto N1-by-N2 matrix
        shift_1 = _floor(2 * M1) - _floor(N1 / 2)
        shift_2 = _floor(2 * M2) - _floor(N2 / 2)
        Y = X[:, np.arange(N2) + shift_2]
        Y[:, np.arange(N2 - shift_2, N2)] += X[:, :shift_2]
        Y[:, :shift_2] += X[:, N2 + shift_2:N2 + 2 * shift_2]
        X = Y[np.arange(N1) + shift_1, :]
        X[np.arange(N1 - shift_1, N1), :] += Y[:shift_1, :]
        X[:shift_1, :] += Y[N1 + shift_1:N1 + 2 * shift_1, :]
    else:
        # Extension to a N1-by-N2 matrix
        Y = fftshift(fft2(ifftshift(C[nbscales - 1][0]))) / \
            np.sqrt(C[nbscales - 1][0].size)
        X_topleft_1 = _ceil((N1 + 1) / 2) - _floor(M1)
        X_topleft_2 = _ceil((N2 + 1) / 2) - _floor(M2)
        loc_1 = X_topleft_1 + np.arange(2 * _floor(M1) + 1) - 1
        loc_2 = X_topleft_2 + np.arange(2 * _floor(M2) + 1) - 1
        Y[np.ix_(loc_1, loc_2)] = Y[np.ix_(loc_1, loc_2)] * hipass_finest + X
        X = Y

    x = fftshift(ifft2(ifftshift(X))) * np.sqrt(X.size)
    if is_real:
        x = np.real(x)

    return x
