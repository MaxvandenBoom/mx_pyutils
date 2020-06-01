"""
Functions to load MicroMed trc data
=====================================================


Copyright 2020, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)
Adapted from Fieldtrip 26-5-2020 (version Mariska, edited by Romain)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import logging
import struct
import numpy as np
from math import floor
from functions.misc import allocate_array


def read_data(filename, channels=None, sample_range=(-1, -1)):
    """
    Load the data into a matrix based on channels

    Args:
        filename (str):                 Path to the .trc file
        channels (tuple or list):       The channels to retrieve from the data; None or empty will read all channels
        sample_range (tuple):           The time-span that will be read; -1 will result in the first or last sample

    Returns:
        data (ndarray):                 A two-dimensional array with the data (format: channel x time); or
                                        None when an error occurs
    """

    #
    header = read_header(filename)

    #
    sample_range_start = sample_range[0] if sample_range[0] >= 0 else 0
    sample_range_end = sample_range[1] if sample_range[1] >= 0 else header['num_samples'] - 1
    if sample_range_end < sample_range_start:
        logging.error('Invalid \'range\' parameter, the given end-point (at ' + str(sample_range_end) + ') lies before the start-point (at ' + str(sample_range_start) + ')')
        return None
    if sample_range_end >= header['num_samples']:
        logging.error('Invalid \'range\' parameter, the given end-point (at ' + str(sample_range_end) + ') lies beyond the length of the data-set')
        return None
    num_samples = sample_range_end - sample_range_start + 1

    # retrieve the channel indices
    if channels is None or not channels:
        channels_idx = list(range(len(header['elec'])))
    else:
        # find the indices based on the channel names
        channels_idx = []
        for req_channel in channels:
            for search_channel_idx in range(len(header['elec'])):
                if header['elec'][search_channel_idx]['name'] == req_channel:
                    channels_idx.append(search_channel_idx)
                    break
        if not len(channels) == len(channels_idx):
            logging.error('Could not find all the requested channels')
            return None

    # read the segment of data
    offset = header['data_start_offset'] + (header['num_chan'] * header['bytes'] * sample_range_start)
    if header['bytes'] == 1:
        read_dtype = np.uint8
    elif header['bytes'] == 2:
        read_dtype = np.uint16
    elif header['bytes'] == 4:
        read_dtype = np.uint32
    else:
        logging.error('Unknown data type')
        return None

    try:
        data = np.fromfile(filename, dtype=read_dtype, count=header['num_chan'] * num_samples, sep='', offset=offset)
        data = np.reshape(data, (header['num_chan'], num_samples), 'F')
    except Exception as e:
        logging.error('Error while reading .trc data, message: ' + str(e))
        return None

    # allocate a buffer for the data
    ret_data = allocate_array((len(channels_idx), num_samples), dtype=np.float64)
    if ret_data is None:
        return None

    # pick the channels and convert the data
    for i in range(len(channels_idx)):
        ret_data[i, :]  = ((data[channels_idx[i], :].astype(np.float64) - header['elec'][channels_idx[i]]['logic_gnd']) /
                            (header['elec'][channels_idx[i]]['logic_max'] - header['elec'][channels_idx[i]]['logic_min'] + 1)
                            ) * (header['elec'][channels_idx[i]]['phys_max'] - header['elec'][channels_idx[i]]['phys_min'])

    return ret_data


def read_header(filename):
    """
    Load the header

    Args:
        filename (str):                 Path to the .trc file

    Returns:
        data (dict):                    A dictionary object with the header information; or None when an error occurs
    """

    header = {}

    # read the file
    f = open(filename, "rb")
    try:

        #
        # reading patient & recording info
        #
        f.seek(64)
        header['surname']       = f.read(22).decode("utf-8").strip()
        header['name']          = f.read(20).decode("utf-8").strip()

        f.seek(128)
        header['day']           = int(f.read(1)[0])
        header['month']         = int(f.read(1)[0])
        header['year']          = 1900 + int(f.read(1)[0])


        #
        # reading header info
        #
        f.seek(175)
        header['header_type']     = int(f.read(1)[0])
        if not header['header_type'] == 4:
            logging.error('*.trc file is not Micromed System98 Header type 4')
            return None

        f.seek(138)
        header['data_start_offset']         = struct.unpack("I", f.read(4))[0]
        header['num_chan']                  = struct.unpack("H", f.read(2))[0]
        header['multiplexer']               = struct.unpack("H", f.read(2))[0]
        header['rate_min']                  = struct.unpack("H", f.read(2))[0]
        header['bytes']                     = struct.unpack("H", f.read(2))[0]

        f.seek(176 + 8)
        header['code_area']                 = struct.unpack("I", f.read(4))[0]
        header['code_area_length']          = struct.unpack("I", f.read(4))[0]

        f.seek(192 + 8)
        header['electrode_area']            = struct.unpack("I", f.read(4))[0]
        header['electrode_area_length']     = struct.unpack("I", f.read(4))[0]

        f.seek(400 + 8)
        header['trigger_area']              = struct.unpack("I", f.read(4))[0]
        header['trigger_area_length']       = struct.unpack("I", f.read(4))[0]

        #
        # Retrieving electrode info
        #

        # Order
        f.seek(184)
        OrderOff = struct.unpack("L", f.read(4))[0]
        f.seek(OrderOff)
        vOrder = [0] * header['num_chan']
        for iChan in range(header['num_chan']):
            vOrder[iChan] = struct.unpack("H", f.read(2))[0]

        # electrodes
        header['elec'] = []
        f.seek(200)
        ElecOff = struct.unpack("L", f.read(4))[0]
        for iChan in range(header['num_chan']):
            f.seek(ElecOff + 128 * vOrder[iChan])
            if struct.unpack("B", f.read(1))[0] == 0:
                continue

            elec = {}
            elec['bip']         = struct.unpack("B", f.read(1))[0]
            elec['name']        = f.read(6).decode("utf-8").replace('\x00', '').strip()
            elec['ref']         = f.read(6).decode("utf-8").replace('\x00', '').strip()
            elec['logic_min']   = struct.unpack("l", f.read(4))[0]
            elec['logic_max']   = struct.unpack("l", f.read(4))[0]
            elec['logic_gnd']   = struct.unpack("l", f.read(4))[0]
            elec['phys_min']    = struct.unpack("l", f.read(4))[0]
            elec['phys_max']    = struct.unpack("l", f.read(4))[0]
            elec['unit_num']    = struct.unpack("H", f.read(2))[0]
            if elec['unit_num'] == -1:
                elec['unit_num'] = 'nV'
            elif elec['unit_num'] == 0:
                elec['unit_num'] = 'uV'
            elif elec['unit_num'] == 1:
                elec['unit_num'] = 'mV'
            elif elec['unit_num'] == 2:
                elec['unit_num'] = 'V'
            elif elec['unit_num'] == 100:
                elec['unit_num'] = '%'
            elif elec['unit_num'] == 101:
                elec['unit_num'] = 'bpm'
            elif elec['unit_num'] == 102:
                elec['unit_num'] = 'Adim.'


            f.seek(ElecOff + 128 * vOrder[iChan] + 44)
            elec['fs_coeff']        = struct.unpack("H", f.read(2))[0]

            f.seek(ElecOff + 128 * vOrder[iChan] + 90)
            elec['x_pos']           = struct.unpack("f", f.read(4))[0]
            elec['y_pos']           = struct.unpack("f", f.read(4))[0]
            elec['z_pos']           = struct.unpack("f", f.read(4))[0]

            f.seek(ElecOff + 128 * vOrder[iChan] + 102)
            elec['type']            = struct.unpack("H", f.read(2))[0]


            header['elec'].append(elec)

        #
        # determine the number of samples
        #
        f.seek(header['data_start_offset'])
        datbeg = f.tell()
        f.seek(0, 2)
        datend = f.tell()
        header['num_samples'] = (datend - datbeg) / (header['bytes'] * header['num_chan'])
        if not (header['num_samples'] % 1 == 0):
            logging.warning('rounding off the number of samples')
            header['num_samples'] = floor(header['num_samples'])
        header['num_samples'] = int(header['num_samples'])

        #
        return header

    except Exception as e:
        logging.error('Error while reading .trc header, message: ' + str(e))
        return None

    finally:
        f.close()
