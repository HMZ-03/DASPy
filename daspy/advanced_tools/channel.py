# Purpose: Several functions for analysis data quality and geometry of channels
# Author: Minzhe Hu, Zefeng Li
# Date: 2024.11.18
# Email: hmz2018@mail.ustc.edu.cn
import numpy as np
from copy import deepcopy
from geographiclib.geodesic import Geodesic
from pyproj import Proj


def robust_polyfit(data, deg, thresh):
    """
    Fit a curve with a robust weighted polynomial.

    :param data: 1-dimensional array.
    :param deg: int. Degree of the fitting polynomial
    :param thresh: int or float. Defined MAD multiple of outliers.
    :return: Fitting data
    """
    nch = len(data)
    channels = np.arange(nch)
    p_coef = np.polyfit(channels, data, deg)
    p_fit = np.poly1d(p_coef)
    old_data = p_fit(channels)
    mse = 1

    # robust fitting until the fitting curve changes < 0.1% at every point.
    while mse > 0.001:
        rsl = abs(data - old_data)
        mad = np.median(rsl)
        weights = np.zeros(nch)
        weights[rsl < thresh * mad] = 1
        p_coef = np.polyfit(channels, data, deg, w=weights)
        p_fit = np.poly1d(p_coef)
        new_data = p_fit(channels)
        mse = np.nanmax(np.abs((new_data - old_data) / old_data))
        old_data = new_data

    return new_data, weights


def _continuity_checking(lst1, lst2, adjacent=2, toleration=2):
    lst1_raw = deepcopy(lst1)
    for chn in lst1_raw:
        discont = [a for a in lst2 if abs(a - chn) <= adjacent]
        if len(discont) >= adjacent * 2 + 1 - toleration:
            lst1.remove(chn)
            lst2.append(chn)

    return lst1, lst2


def channel_checking(data, deg=10, thresh=5, continuity=True, adjacent=2,
                     toleration=2, mode='low', verbose=False):
    """
    Use the energy of each channel to determine which channels are bad.

    :param data: 2-dimensional np.ndarray. Axis 0 is channel number and axis 1
        is time series.
    :param deg: int. Degree of the fitting polynomial
    :param thresh: int or float. The MAD multiple of bad channel energy lower
        than good channels.
    :param continuity: bool. Perform continuity checks on bad channels and good
        channels.
    :param adjacent: int. The number of nearby channels for continuity checks.
    :param toleration: int. The number of discontinuous channel allowed in each
        channel (including itself) in the continuity check.
    :param mode: str. 'low' means bad channels have low amplitude, 'high' means
        bad channels have high amplitude, and 'both' means bad channels are
        likely to have low or high amplitude.
    :return: Good channels and bad channels.
    """
    nch = len(data)
    energy = np.log10(np.sum(data**2, axis=1))
    energy[energy == -np.inf] = -308

    # Remove abnormal value by robust polynomial fitting.
    fitted_energy, weights = robust_polyfit(energy, deg, thresh)
    deviation = energy - fitted_energy

    # Iterate eliminates outliers.
    mad = np.median(abs(deviation[weights > 0]))
    if mode == 'low':
        bad_chn = np.argwhere(deviation < -thresh * mad).ravel().tolist()
    elif mode == 'high':
        bad_chn = np.argwhere(deviation > thresh * mad).ravel().tolist()
    elif mode == 'high':
        bad_chn = np.argwhere(deviation < -thresh * mad).ravel().tolist() + \
                np.argwhere(deviation > thresh * mad).ravel().tolist()
    good_chn = list(set(range(nch)) - set(bad_chn))

    if continuity:
        # Discontinuous normal value are part of bad channels.
        good_chn, bad_chn = _continuity_checking(good_chn, bad_chn,
                                                 adjacent=adjacent,
                                                 toleration=toleration)

        # Discontinuous outliers are usually not bad channels.
        bad_chn, good_chn = _continuity_checking(bad_chn, good_chn,
                                                 adjacent=adjacent,
                                                 toleration=toleration)

    bad_chn = np.sort(np.array(bad_chn))
    good_chn = np.sort(np.array(good_chn))
    if verbose:
        return good_chn, bad_chn, energy, fitted_energy - thresh * mad

    return good_chn, bad_chn


def _channel_location(track_pt):
    track, tn = track_pt[:, :-1], track_pt[:, -1]
    dim = track.shape[1]
    l_track = np.sqrt(np.sum(np.diff(track, axis=0) ** 2, axis=1))
    l_track_cum = np.hstack(([0], np.cumsum(l_track)))
    idx_kp = np.where(tn >= 0)[0]

    interp_ch = []
    chn = np.floor(tn[idx_kp[0]]).astype(int)
    interp_ch.append([*track[idx_kp[0]], chn])
    if abs(chn - tn[idx_kp[0]]) > 1e-6:
        chn += 1

    seg_interval = []
    for i in range(1, len(idx_kp)):
        # calculate actual interval between known-channel points
        istart, iend = idx_kp[i - 1], idx_kp[i]
        n_chn_kp = tn[iend] - tn[istart]
        d_interp = (l_track_cum[iend] - l_track_cum[istart]) / n_chn_kp
        seg_interval.append([tn[istart], tn[iend], d_interp])

        l_res = 0  # remaining fiber length before counting the next segment
        # consider if the given channelnumber is not an integer
        chn_res = tn[istart] - int(tn[istart])
        if d_interp == 0:
            while chn < int(tn[iend]):
                chn += 1
                interp_ch.append([*track[istart, :], chn])
            continue
        for j in range(istart, iend):
            l_start = l_track[j] + l_res

            # if tp segment length is large for more than one interval, get the
            # channel loc
            if l_start >= d_interp * (1 - chn_res - 1e-6):
                # floor int, num of channel available
                n_chn_tp = int(l_start / d_interp + chn_res)
                l_new = (np.arange(n_chn_tp) + 1 - chn_res) * d_interp - \
                    l_res  # channel distance from segment start

                # interpolate the channel loc
                t_new = np.zeros((len(l_new), dim))
                for d in range(dim):
                    t_new[:, d] = np.interp(l_new, [0, l_track[j]],
                                            [track[j, d], track[j + 1, d]])

                # remaining length to add to next segment
                l_res = l_start - n_chn_tp * d_interp

                # write interpolated channel loc
                for ti in t_new:
                    chn += 1
                    interp_ch.append([*ti, chn])

                # handle floor int problem when l_start/d_interp is near an
                # interger
                if (d_interp - l_res) / d_interp < 1e-6:
                    chn += 1
                    interp_ch.append([*track[j + 1, :], int(tn[j + 1])])
                    l_res = 0
                chn_res = 0
            # if tp segment length is not enough for one interval, simply add
            # the length to next segment
            elif l_start < d_interp:
                l_res = l_start

    if abs(tn[iend] - int(tn[iend])) > 1e-6:
        chn += 1
        interp_ch.append([*track[iend, :], chn])

    return np.array(seg_interval), np.array(interp_ch)


def location_interpolation(known_pt, track_pt=None, dx=2, data_type='lonlat',
                           verbose=False):
    """
    Interpolate to obtain the positions of all channels.

    :param known_pt: np.ndarray. Points with known channel numbers. Each row
        includes 2 or 3 coordinates and a channel number.
    :param track_pt: np.ndarray. Optional fiber spatial track points without
        channel numbers. Each row includes 2 or 3 coordinates. Please ensure
        that the track points are arranged in increasing order of track number.
        If track points is not dense enough, please insert the coordinates of
        known points into track points in order.
    :param dx: Known points far from the track (> dx) will be excluded.
        Recommended setting is channel interval. The unit is m.
    :param data_type: str. Coordinate type. 'lonlat' ('lonlatheight') for
        longitude, latitude in degree (and height in meters), 'xy' ('xyz') for
        x, y (and z) in meters.
    :param verbose: bool. If True, return interpoleted channel location and
        segment interval.
    :return: Interpoleted channel location if verbose is False.
    """
    known_pt = known_pt[known_pt[:,-1].argsort()]
    dim = known_pt.shape[1] - 1
    if 'lonlat' in data_type:
        zone = np.floor((max(known_pt[:,0]) + min(known_pt[:,0])) / 2 / 6)\
            .astype(int) + 31
        DASProj = Proj(proj='utm', zone=zone, ellps='WGS84',
                       preserve_units=False)
        known_pt[:, 0], known_pt[:, 1] = DASProj(known_pt[:, 0], known_pt[:, 1])
    else:
        assert 'xy' in data_type, ('data_type should be \'lonlat\',\''
                                   'lonlatheight\', \'xy\' or \'xyz\'')

    if track_pt is None:
        seg_interval, interp_ch = _channel_location(known_pt)
    else:
        K = len(known_pt)
        T = len(track_pt)
        track_pt = np.c_[track_pt, np.zeros(T) - 1]
        if 'lonlat' in data_type:
            track_pt[:, 0], track_pt[:, 1] = DASProj(track_pt[:, 0],
                                                     track_pt[:, 1])

        # insert the known points into the fiber track data
        matrix = [np.tile(track_pt[:, d], (K, 1)) -
                  np.tile(known_pt[:, d], (T, 1)).T for d in range(dim)]

        dist = np.sqrt(np.sum(np.array(matrix) ** 2, axis=0))
        for k in range(K):
            if min(dist[k]) < dx:
                t_list = np.sort(np.where(dist[k] == min(dist[k]))[0])
                for t in t_list:
                    if track_pt[t, -1] == -1:
                        track_pt[t, -1] = known_pt[k, -1]
                        last_pt = t
                        break

        # interpolation with regular spacing along the fiber track
        try:
            track_pt = track_pt[:last_pt + 1]
        except NameError:
            print('All known points are too far away from the track points. If '
                  'they are reliable, they can be merged in sequence as track '
                  'points to input')
            return None

        seg_interval, interp_ch = _channel_location(track_pt)

    if data_type == 'lonlat':
        interp_ch[:, 0], interp_ch[:, 1] = \
            DASProj(interp_ch[:, 0], interp_ch[:, 1], inverse=True)

    if verbose:
        return interp_ch, seg_interval
    return interp_ch


def _xcorr(x, y):
    N = len(x)
    meanx = np.mean(x)
    meany = np.mean(y)
    stdx = np.std(np.asarray(x))
    stdy = np.std(np.asarray(y))
    c = np.sum((y - meany) * (x - meanx)) / (N * stdx * stdy)
    return c


def _horizontal_angle_change(geo, gap=10):
    nch = len(geo)
    angle = np.zeros(nch)
    for i in range(1, nch - 1):
        lon, lat = geo[i]
        lon_s, lat_s = geo[max(i - gap, 0)]
        lon_e, lat_e = geo[min(i + gap, nch - 1)]
        azi_s = Geodesic.WGS84.Inverse(lat_s, lon_s, lat, lon)['azi1']
        azi_e = Geodesic.WGS84.Inverse(lat, lon, lat_e, lon_e)['azi1']
        dazi = azi_e - azi_s
        if abs(dazi) > 180:
            dazi = -np.sign(dazi) * (360 - abs(dazi))
        angle[i] = dazi

    return angle


def _vertical_angle_change(geo, gap=10):
    nch = len(geo)
    angle = np.zeros(nch)
    for i in range(1, nch - 1):
        lon, lat, dep = geo[i]
        lon_s, lat_s, dep_s = geo[max(i - gap, 0)]
        lon_e, lat_e, dep_e = geo[min(i + gap, nch - 1)]
        s12_s = Geodesic.WGS84.Inverse(lat_s, lon_s, lat, lon)['s12']
        theta_s = np.arctan((dep - dep_s) / s12_s) / np.pi * 180
        s12_e = Geodesic.WGS84.Inverse(lat, lon, lat_e, lon_e)['s12']
        theta_e = np.arctan((dep_e - dep) / s12_e) / np.pi * 180
        angle[i] = theta_e - theta_s

    return angle


def _local_maximum_indexes(data, thresh):
    idx = np.where(data > thresh)[0]
    if len(idx):
        i = list(np.where(np.diff(idx) > 1)[0] + 1)
        if len(idx) - 1 not in i:
            i.append(len(idx) - 1)
        b = 0
        max_idx = []
        for e in i:
            max_idx.append(idx[b] + np.argmax(data[idx[b]:idx[e]]))
            b = e
        return max_idx
    else:
        return []


def turning_points(data, data_type='coordinate', thresh=5, depth_info=False,
                   channel_gap=3):
    """
    Seek turning points in the DAS channel.

    :param data: numpy.ndarray. Data used to seek turning points.
    :param data_type: str. If data_type is 'coordinate', data should include
        longitude and latitude (first two columns), and can also include depth
        (last column). If data_type is 'waveform', data should be continuous
        waveform, preferably containing signal with strong coherence
        (earthquake, traffic signal, etc.).
    :param thresh: For coordinate data, when the angle of the optical cables on
        both sides centered on a certain point exceeds thresh, it is considered
        an turning point. For waveform, thresh means the MAD multiple of
        adjacent channel cross-correlation values lower than their median.
    :param depth_info: bool. Optional if data_type is 'coordinate'. Whether
        depth (in meters) is included in the coordinate data and need to be
        used.
    :param channel_gap: int. Optional if data_type is 'coordinate'. The smaller
        the value is, the finer the segmentation will be. It is recommended to
        set it to half the ratio of gauge length and channel interval.
    :return: list. Channel index of turning points.
    """
    if data_type == 'coordinate':
        angle = _horizontal_angle_change(data[:, :2], gap=channel_gap)
        turning_h = _local_maximum_indexes(abs(angle), thresh)

        if depth_info:
            angle = _vertical_angle_change(data, gap=channel_gap)
            turning_v = _local_maximum_indexes(abs(angle), thresh)
            return turning_h, turning_v

        return turning_h

    elif data_type == 'waveform':
        nch = len(data)
        cc = np.zeros(nch - 1)
        for i in range(nch - 1):
            cc[i] = _xcorr(data[i], data[i + 1])
        median = np.median(cc)
        mad = np.median(abs(cc - median))

        return np.argwhere(cc < median - thresh * mad)[0]

    else:
        raise ValueError('Data_type should be \'coordinate\' or \'waveform\'.')


def _equally_spacing(channels, dist, dx):
    if len(channels) > 20:
        return _equally_spacing_2(channels, dist, dx)
    else:
        return _equally_spacing_1(channels, dist, dx)


def _equally_spacing_1(channels, dist, dx):
    nch = len(channels)
    residual = np.inf
    for i in range(2 ** (nch - 2)):
        state = bin(i)[2:].rjust(nch - 2, '0')
        dist_new = [dist[0]]
        channels_new = [channels[0]]
        idx = 0
        for j, s in enumerate(state):
            if s == '0':
                dist_new[idx] += dist[j+1]
            else:
                dist_new.append(dist[j+1])
                channels_new.append(channels[j+1])
                idx += 1
        res = sum([abs(d - dx) for d in dist_new])    
        if res < residual:
            residual = res
            dist_equal = dist_new
            channels_equal = channels_new
    channels_equal.append(channels[-1])
    return channels_equal, dist_equal


def _equally_spacing_2(channels, dist, dx):
    channels_equal = [channels[0]]
    dist_equal = []
    i = 0
    while i < len(dist):
        d = dist[i]
        if d < dx and i < len(dist) - 1:
            d1 = d + dist[i + 1]
            while d1 < dx and i < len(dist) - 2:
                d = d1
                i += 1
                d1 += dist[i + 1]
            if abs(d - dx) <= abs(d1 - dx):
                channels_equal.append(channels[i + 1])
                dist_equal.append(d)
            else:
                i += 1
                channels_equal.append(channels[i + 1])
                dist_equal.append(d1)
        else:
            channels_equal.append(channels[i + 1])
            dist_equal.append(d)
        i += 1

    return channels_equal, dist_equal


def equally_spaced_channels(geometry, dx, depth_info=False, verbose=False):
    """
    Find equally spaced channel numbers based on known DAS latitude and
    longitude.

    :param geometry: numpy.ndarray. DAS geometry used to filter equally spaced
        channels. It needs to consist of longitude, latitude (and depth) or
        channel number, longitude, latitude (and depth).
    :param dx: Channel interval.
    :param depth_info: bool. Whether depth (in meters) is included in the
        geometry and needed to be used.
    :param verbose: bool. If True, return channel numbers for equally spaced
        channels and channel intervals.
    :return: Channel numbers for equally spaced channels if verbose is False.
    """
    nch = len(geometry)
    if geometry.shape[1] == 2 + int(depth_info):
        channels = np.arange(nch).astype(int)
    else:
        geometry = geometry[geometry[:, 0].argsort()]
        channels = geometry[:, 0].astype(int)
        geometry = geometry[:, 1:]

    dist = np.zeros(nch - 1)
    for i in range(nch - 1):
        lon0, lat0 = geometry[i, :2]
        lon1, lat1 = geometry[i+1, :2]
        d = Geodesic.WGS84.Inverse(lat0, lon0, lat1, lon1)['s12']
        if depth_info:
            dist[i] = np.sqrt(d**2 + (geometry[i+1, 2] - geometry[i, 2]) ** 2)
        else:
            dist[i] = d

    channels_equal = [channels[0]]
    dist_equal = []
    channels_seg = []
    dist_seg = []
    flag = False
    for i in range(1, nch-1):
        if dist[i-1] + dist[i] <= dx * 1.5:
            channels_seg.append(channels[i-1])
            dist_seg.append(dist[i-1])
        else:
            if len(channels_seg):
                channels_seg.extend(channels[i-1:i+1])
                dist_seg.append(dist[i-1])
                channels_seg, dist_seg = _equally_spacing(channels_seg,
                                                          dist_seg, dx)
                dist_equal.extend(dist_seg)
                channels_equal.extend(channels_seg[1:])
                channels_seg = []
                dist_seg = []
                flag = False
            else:
                if flag:
                    channels_equal.append(channels[i-1])
                    dist_equal.append(dist[i-1])
                else:
                    flag = True

    if len(channels_seg):
        channels_seg.extend(channels[i:i+2])
        dist_seg.append(dist[i])
        channels_seg, dist_seg = _equally_spacing(channels_seg, dist_seg, dx)
        dist_equal.extend(dist_seg)
        channels_equal.extend(channels_seg[1:])
    else:
        channels_equal.extend(channels[-2:])
        dist_equal.extend(dist[-2:])

    if verbose:
        return channels_equal, dist_equal
    return channels_equal