#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "click>=8.1.7",
#     "napari[all]>=0.6.6",
#     "requests>=2.31.0",
#     "numpy>=1.26.4"
# ]
# ///

import cmath
import json
import requests
import struct

import numpy
numpy.seterr(all='ignore')   # Suppress divide by zero warnings, just return NaNs

# The above doesn't work, so suppress all warnings:
import warnings
warnings.filterwarnings("ignore")

import click

import napari

# AOCal binary header - this structure, followed by a numpy array of complex128 values
HEADER_FORMAT = "8s6I2d"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# Globals that will contain receiver/tile mappings after get_metadata is called
TI2TN = []     # Sorted list of tile ID numbers
TI2R = []      # One value for each tile, containing the receiver number associated with that tile
R2TLIST = {}   # A dict with receiver ID as key, and a list of tile IDs as value
CHANNELS = []  # A list of all 24 channels associated with this observation


def get_metadata(obsid):
    """
    Get metadata for the given observation ID from the MWA metadata service.

    Populates global variables TI2TN, TI2R, R2TLIST, and CHANNELS using this observation's metadata.

    :param obsid: The observation ID to pass to the web service
    :return: None
    """
    global TI2TN, TI2R, R2TLIST, CHANNELS
    result = requests.get('https://ws.mwatelescope.org/metadata/obs', data={'obs_id': obsid})
    obs_info = json.loads(result.text)
    TI2TN = obs_info['faults']['all_tiles']
    TI2TN.sort()
    TI2R = [obs_info['tdict'][str(tid)][1] for tid in TI2TN]
    for ti in range(len(TI2TN)):
        if TI2R[ti] not in R2TLIST:
            R2TLIST[TI2R[ti]] = []
        R2TLIST[TI2R[ti]].append(TI2TN[ti])
    CHANNELS = obs_info['rfstreams']['0']['frequencies']
    assert obsid == obs_info['starttime']


def get_receiver_shapedata(nchannels):
    """
    Returns the structures needed by Napari to add a 'shapes' layer containing receiver boundaries.

    These are:
        names: A list of receiver name strings, one per receiver, including the tiles on that receiver
        corners: A list of shapes, each defined by a list of four corner coordinates

    :param nchannels: Number of fine channels in the calibration file
    :return: a tuple pf (names, corners)
    """
    names, corners = [], []
    for i in range(0, len(TI2TN), 8):
        corners.append(numpy.array([[i, 0], [i, nchannels - 1], [i + 7, nchannels - 1], [i + 7, 0]], dtype=int))
        names.append('Rec%02d(%d-%d)' % (TI2R[i], R2TLIST[TI2R[i]][0], R2TLIST[TI2R[i]][0] + 7))

    return names, corners


def get_channel_shapedata(nchannels):
    """
    Returns the structures needed by Napari to add a 'shapes' layer containing coarse channel boundaries.

    These are:
        names: A list of coarse channel name strings, one per receiver
        corners: A list of shapes, each defined by a list of four corner coordinates

    :param nchannels: Number of fine channels in the calibration file
    :return: a tuple pf (names, corners)
    """
    names, corners = [], []
    ntiles = len(TI2TN)
    for i in range(0, nchannels, nchannels // 24):
        corners.append(numpy.array([[0, i], [ntiles, i], [ntiles, i + nchannels // 24], [0, i + nchannels // 24]], dtype=int))
        names.append('Chan:%d' % CHANNELS[i // (nchannels // 24)])

    return names, corners


def load_aocal(filename, divide_index=None):
    """
    Load a binary format AOCal object in a bytes string, and split into the header and data.

    If the input file has more than 768 coarse channels (40 kHz frequency resolution), then the data is
    averaged to downsample to this resolution for display.

    If divide_index is provided, then the phases of all tiles are divided by the phases of this reference tile. This
    can be used to compare AOCal files produced by hyperdrive (that don't use a reference tile) with files produced
    by the web service (that always have zero phase offset for the reference tile).

    Note that gains are NOT corrected relative to this reference tile, because the database fits have aboslute gains,
    not gains relative to the reference tile.

    The complex values are converted and returned as two numpy arrays:
        gains: The absolute value (vector lengths) of the complex values
        phases: The polar angle (in radians) of the complex values

    :param filename: file name to read
    :param divide_index: If not None, divide all complex values by the corresponding value for this tile index
    :return: A tuple of (ntiles, nchannels, gains, phases) where gains and phases are numpy ndarrays of shape (ntiles, nchannel, 4)
    """
    with open(filename, 'rb') as f:
        aocal_string = f.read()
    header = struct.unpack(HEADER_FORMAT, aocal_string[:HEADER_SIZE])
    _, _, _, _, ntiles, nchannels, _, _, _ = header
    print('Loaded file: %s - header: %s' % (filename, header))
    data = numpy.frombuffer(aocal_string[HEADER_SIZE:], dtype=numpy.complex128)
    data.shape = (1, ntiles, nchannels, 4)
    ndata = data[0, :, :, :]

    ndata = 1.0 / ndata  # Take the inverse

    # downsample by averaging to 40kHz if higher res than this:
    if nchannels > 768:
        dfactor = nchannels // 768
        ndata = ndata.reshape((ntiles, 768, dfactor, 4))
        ndata = numpy.nanmean(ndata, axis=2)
        nchannels = 768

    gains, phases = numpy.vectorize(cmath.polar)(ndata)

    gains[numpy.isinf(gains)] = 0

    if divide_index is not None:
        phases = phases - phases[divide_index]
        # gains = gains / gains[divide_index]   # Don't correct, because gains in database fit generated aocal files aren't relative to the reference tile

    print('    Gains min/max = %f %f' % (numpy.nanmin(gains), numpy.nanmax(gains)))
    print('    Phases min/max = %f %f' % (numpy.nanmin(phases), numpy.nanmax(phases)))

    return ntiles, nchannels, gains, phases


@click.group()
def cli():
    pass


@cli.command(context_settings={"ignore_unknown_options": True})
@click.argument('filename')
@click.option('--channel', type=int, default=None)
@click.option('--divide_index', type=int, default=None)
@click.option('--show_tiledata', is_flag=True)
def view(filename, channel, divide_index, show_tiledata):
    ntiles, nchannels, gains, phases = load_aocal(filename, divide_index=divide_index)
    # gains and phases have shape (ntiles, nchannels, 4)

    obsid = ''.join([x for x in filename if x.isdigit()])[:10]
    get_metadata(obsid=int(obsid))

    if channel is not None:
        if divmod(nchannels, 24)[1] == 0:    # Choose only one channel out of a 24-channel fit
            nchannels = nchannels // 24   # Only as wide as one coarse channel
            if 1 <= channel <= 24:   # Index 1-24 in the list of channels in this observation == mwax box number
                coarse_channel_index = channel - 1
            else:
                coarse_channel_index = CHANNELS.index(channel)
            gains = gains[:, (coarse_channel_index * nchannels):((coarse_channel_index + 1) * nchannels), :]
            phases = phases[:, (coarse_channel_index * nchannels):((coarse_channel_index + 1) * nchannels), :]
            delta_freq = 1.28e6 / nchannels
        else:   # Identify that we only have a single channel fit, but tell us which channel it is:
            if 1 <= channel <= 24:   # Index 1-24 in the list of channels in this observation == mwax box number
                coarse_channel_index = channel - 1
            else:
                coarse_channel_index = CHANNELS.index(channel)
            delta_freq = 1.28e6 / nchannels
        print('Showing coarse channel: %d' % CHANNELS[coarse_channel_index])
    else:
        delta_freq = 30.72e6 / nchannels
        print('Showing coarse channels: %s' % CHANNELS)

    if show_tiledata:
        for i in range(ntiles):
            delta_phase = (phases[i, 12, 0] - phases[i, 2, 0]) / 10   # Average over 10 fine channels
            if delta_phase > cmath.pi:
                delta_phase -= cmath.pi * 2
            elif delta_phase < -cmath.pi:
                delta_phase += cmath.pi * 2
            phase_slope = delta_phase / delta_freq   # In radians per Hz
            delay_m = phase_slope / (2 * cmath.pi / 2.9979e8)
            print('Tile %d: Px=%6.3f, Py=%6.3f, delay_m=%6.2f m' % (TI2TN[i], phases[i][2][0], phases[i][2][1], delay_m))

    pnames = ['Phases:X', 'Phases:XY', 'Phases:YX', 'Phases:Y']
    gnames = ['Gains:X', 'Gains:XY', 'Gains:YX', 'Gains:Y']
    cmaps = ['hsv', 'hsv', 'hsv', 'hsv']

    viewer = napari.Viewer()
    gn = viewer.add_image(gains,
                          gamma=0.5,
                          visible=False,
                          interpolation2d='nearest',
                          interpolation3d='nearest',
                          name=gnames,
                          channel_axis=2)

    pn = viewer.add_image(phases,
                          visible=False,
                          colormap=cmaps,
                          contrast_limits=[-cmath.pi, cmath.pi],
                          interpolation2d='nearest',
                          interpolation3d='nearest',
                          name=pnames,
                          channel_axis=2)

    viewer.dims.axis_labels = ('Input', 'Channel')
    viewer.axes.labels = True
    viewer.axes.visible = True

    receiver_labels = numpy.zeros(shape=(ntiles, nchannels), dtype=numpy.int8)
    tilemap = numpy.zeros(shape=(ntiles, nchannels), dtype=numpy.int32)
    for i in range(ntiles):
        receiver_labels[i, :] = TI2R[i]
        tilemap[i, :] = TI2TN[i]

    rl = viewer.add_labels(receiver_labels, name='Receiver ID', depiction='plane', blending='additive', opacity=0.0)
    rl2 = viewer.add_labels(tilemap, name='Tile ID', depiction='plane', blending='additive', opacity=0.0)

    rnames, rcorners = get_receiver_shapedata(nchannels)

    rfeatures = {'rname':rnames}
    rtext = {'string': '{rname}:',
             'anchor': 'center',
             'translation': [0, 0],
             'size': 8,
             'color': 'green'}

    # add polygons
    shapes_layer = viewer.add_shapes(rcorners,
                                     features=rfeatures,
                                     shape_type='polygon',
                                     edge_width=1,
                                     face_color='transparent',
                                     text=rtext,
                                     name='Receivers')

    if channel is None:
        cnames, ccorners = get_channel_shapedata(nchannels)

        cfeatures = {'cname': cnames}
        ctext = {'string': '{cname}:',
                 'anchor': 'center',
                 'translation': [0, 0],
                 'size': 8,
                 'color': 'green'}

        # add polygons
        shapes_layer = viewer.add_shapes(ccorners,
                                         features=cfeatures,
                                         shape_type='polygon',
                                         edge_width=1,
                                         face_color='transparent',
                                         text=ctext,
                                         name='Channels')

    viewer.reset_view()
    napari.run()


@cli.command(context_settings={"ignore_unknown_options": True})
@click.argument('filenames', nargs=2)
@click.option('--channel', type=int, default=None)
@click.option('--divide_index1', type=int, default=None)
@click.option('--divide_index2', type=int, default=None)
def diff(filenames, channel, divide_index1, divide_index2):
    ntiles1, nchannels1, gains1, phases1 = load_aocal(filenames[0], divide_index=divide_index1)
    ntiles2, nchannels2, gains2, phases2 = load_aocal(filenames[1], divide_index=divide_index2)

    obsid = ''.join([x for x in filenames[0] if x.isdigit()])[:10]
    get_metadata(obsid=int(obsid))

    if channel is not None:
        if 1 <= channel <= 24:  # Index 1-24 in the list of channels in this observation == mwax box number
            coarse_channel_index = channel - 1
        else:
            coarse_channel_index = CHANNELS.index(channel)

        if divmod(nchannels1, 24)[1] == 0:
            nchannels1 = nchannels1 // 24   # Only as wide as one coarse channel
            gains1 = gains1[:, (coarse_channel_index * nchannels1):((coarse_channel_index + 1) * nchannels1), :]
            phases1 = phases1[:, (coarse_channel_index * nchannels1):((coarse_channel_index + 1) * nchannels1), :]

        if divmod(nchannels2, 24)[1] == 0:
            nchannels2 = nchannels2 // 24   # Only as wide as one coarse channel
            gains2 = gains2[:, (coarse_channel_index * nchannels2):((coarse_channel_index + 1) * nchannels2), :]
            phases2 = phases2[:, (coarse_channel_index * nchannels2):((coarse_channel_index + 1) * nchannels2), :]

        print('Showing coarse channel: %d' % CHANNELS[coarse_channel_index])
    else:
        print('Showing coarse channels: %s' % CHANNELS)

    assert ntiles1 == ntiles2
    assert nchannels1 == nchannels2
    ntiles = ntiles1
    nchannels = nchannels1
    # gains and phases have shape (ntiles, nchannels, 2)

    gains = numpy.log10(gains1 / gains2)
    phases = numpy.minimum(abs(phases1 - phases2), cmath.pi * 2 - abs(phases1 - phases2))

    print('Differenced Gains: log10(G1/G2): min, max, stdev')
    print('   XX: %f - %f +/- %f' % (numpy.nanmin(gains[:,:,0]), numpy.nanmax(gains[:,:,0]), numpy.nanstd(gains[:,:,0])))
    print('   XY: %f - %f +/- %f' % (numpy.nanmin(gains[:,:,1]), numpy.nanmax(gains[:,:,1]), numpy.nanstd(gains[:,:,1])))
    print('   YX: %f - %f +/- %f' % (numpy.nanmin(gains[:,:,2]), numpy.nanmax(gains[:,:,2]), numpy.nanstd(gains[:,:,2])))
    print('   YY: %f - %f +/- %f' % (numpy.nanmin(gains[:,:,3]), numpy.nanmax(gains[:,:,3]), numpy.nanstd(gains[:,:,3])))
    print('Differenced Phases: P1-P2: min, max, stdev')
    print('   XX: %f - %f +/- %f' % (numpy.nanmin(phases[:,:,0]), numpy.nanmax(phases[:,:,0]), numpy.nanstd(phases[:,:,0])))
    print('   XY: %f - %f +/- %f' % (numpy.nanmin(phases[:,:,1]), numpy.nanmax(phases[:,:,1]), numpy.nanstd(phases[:,:,1])))
    print('   YX: %f - %f +/- %f' % (numpy.nanmin(phases[:,:,2]), numpy.nanmax(phases[:,:,2]), numpy.nanstd(phases[:,:,2])))
    print('   YY: %f - %f +/- %f' % (numpy.nanmin(phases[:,:,3]), numpy.nanmax(phases[:,:,3]), numpy.nanstd(phases[:,:,3])))

    pnames = ['Phases:X1-X2', 'Phases:XY1-XY2', 'Phases:YX1-YX2', 'Phases:Y1-Y2']
    gnames = ['Gains:log10(X1/X2)', 'Gains:log10(XY1/XY2)', 'Gains:log10(YX1/YX2)', 'Gains:log10(Y1/Y2)']

    viewer = napari.Viewer()
    gn = viewer.add_image(gains,
                          gamma=0.5,
                          visible=False,
                          interpolation2d='nearest',
                          interpolation3d='nearest',
                          name=gnames,
                          channel_axis=2)

    pn = viewer.add_image(phases,
                          visible=False,
                          interpolation2d='nearest',
                          interpolation3d='nearest',
                          name=pnames,
                          channel_axis=2)

    viewer.dims.axis_labels = ('Input', 'Channel')
    viewer.axes.labels = True
    viewer.axes.visible = True

    receiver_labels = numpy.zeros(shape=(ntiles, nchannels), dtype=numpy.int8)
    tilemap = numpy.zeros(shape=(ntiles, nchannels), dtype=numpy.int32)
    for i in range(ntiles):
        receiver_labels[i, :] = TI2R[i]
        tilemap[i, :] = TI2TN[i]

    rl = viewer.add_labels(receiver_labels, name='Receiver ID', depiction='plane', blending='additive', opacity=0.0)
    rl2 = viewer.add_labels(tilemap, name='Tile ID', depiction='plane', blending='additive', opacity=0.0)

    rnames, rcorners = get_receiver_shapedata(nchannels)

    rfeatures = {'rname':rnames}
    rtext = {'string': '{rname}:',
             'anchor': 'center',
             'translation': [0, 0],
             'size': 8,
             'color': 'green'}

    # add polygons
    shapes_layer = viewer.add_shapes(rcorners,
                                     features=rfeatures,
                                     shape_type='polygon',
                                     edge_width=1,
                                     face_color='transparent',
                                     text=rtext,
                                     name='Receivers')

    if channel is None:
        cnames, ccorners = get_channel_shapedata(nchannels)

        cfeatures = {'cname': cnames}
        ctext = {'string': '{cname}:',
                 'anchor': 'center',
                 'translation': [0, 0],
                 'size': 8,
                 'color': 'green'}

        # add polygons
        shapes_layer = viewer.add_shapes(ccorners,
                                         features=cfeatures,
                                         shape_type='polygon',
                                         edge_width=1,
                                         face_color='transparent',
                                         text=ctext,
                                         name='Channels')

    viewer.reset_view()
    napari.run()


if __name__ == '__main__':
    cli()
