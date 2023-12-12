#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Example OFDM RX
# Author: Cyrille Morin
# GNU Radio version: 3.10.6.0

from gnuradio import analog
from gnuradio import blocks
import pmt
from gnuradio import digital
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import gr, pdu
from gnuradio import uhd
import time
import numpy as np
import ofdm_rx_example_embedded_packet_comp as embedded_packet_comp  # embedded python block




class ofdm_rx_example(gr.top_block):

    def __init__(self, bps=4, fft_len=64, freq=2500000000, gain=20, samp_rate=1000000, skip_time=0.5):
        gr.top_block.__init__(self, "Example OFDM RX", catch_exceptions=True)

        ##################################################
        # Parameters
        ##################################################
        self.bps = bps
        self.fft_len = fft_len
        self.freq = freq
        self.gain = gain
        self.samp_rate = samp_rate
        self.skip_time = skip_time

        ##################################################
        # Variables
        ##################################################
        self.occupied_carriers = occupied_carriers = (list(range(-26, -21)) + list(range(-20, -7)) + list(range(-6, 0)) + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27)),)
        self.frame_amount = frame_amount = 20
        self.pilot_symbols = pilot_symbols = ((1, 1, 1, -1,),)
        self.pilot_carriers = pilot_carriers = ((-21, -7, 7, 21,),)
        self.payload_mod = payload_mod = digital.qam_constellation(constellation_points=2**bps,differential=False,mod_code=digital.utils.mod_codes.GRAY_CODE) if (bps in [2,4,8]) else digital.constellation_bpsk()
        self.length_tag_key = length_tag_key = "frame_len"
        self.header_mod = header_mod = digital.constellation_qpsk()
        self.byte_amount = byte_amount = frame_amount * len(occupied_carriers[0]) // (8//bps)
        self.sync_word2 = sync_word2 = [0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0]
        self.sync_word1 = sync_word1 = [0., 0., 0., 0., 0., 0., 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 0., 0., 0., 0., 0.]
        self.rolloff = rolloff = 0
        self.payload_equalizer = payload_equalizer = digital.ofdm_equalizer_simpledfe(fft_len, payload_mod.base(), occupied_carriers, pilot_carriers, pilot_symbols, 1)
        self.packet_length_tag_key = packet_length_tag_key = "packet_len"
        self.header_formatter = header_formatter = digital.packet_header_ofdm(occupied_carriers, n_syms=1, len_tag_key=length_tag_key, frame_len_tag_key=length_tag_key, bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=payload_mod.bits_per_symbol(), scramble_header=True)
        self.header_equalizer = header_equalizer = digital.ofdm_equalizer_simpledfe(fft_len,header_mod.base(),occupied_carriers,pilot_carriers,pilot_symbols)
        self.cp_len = cp_len = fft_len//4
        self.byte_sequence = byte_sequence = [np.random.randint(256) for x in range(byte_amount)] if np.random.seed(0) is None else 0

        ##################################################
        # Blocks
        ##################################################

        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_clock_source('internal', 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec(0))

        self.uhd_usrp_source_0.set_center_freq(freq, 0)
        self.uhd_usrp_source_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_source_0.set_gain(gain, 0)
        self.pdu_tagged_stream_to_pdu_0 = pdu.tagged_stream_to_pdu(gr.types.byte_t, packet_length_tag_key)
        self.fft_vxx_1 = fft.fft_vcc(fft_len, True, (), True, 1)
        self.fft_vxx_0_0 = fft.fft_vcc(fft_len, True, (), True, 1)
        self.embedded_packet_comp = embedded_packet_comp.blk()
        self.digital_packet_headerparser_b_0 = digital.packet_headerparser_b(header_formatter)
        self.digital_ofdm_sync_sc_cfb_0 = digital.ofdm_sync_sc_cfb(fft_len, cp_len, False, 0.96)
        self.digital_ofdm_serializer_vcc_payload = digital.ofdm_serializer_vcc(fft_len, occupied_carriers, length_tag_key, packet_length_tag_key, 0, '', True)
        self.digital_ofdm_serializer_vcc_header = digital.ofdm_serializer_vcc(fft_len, occupied_carriers, "header_len", '', 0, '', True)
        self.digital_ofdm_frame_equalizer_vcvc_1 = digital.ofdm_frame_equalizer_vcvc(payload_equalizer.base(), cp_len, length_tag_key, False, 0)
        self.digital_ofdm_frame_equalizer_vcvc_0 = digital.ofdm_frame_equalizer_vcvc(header_equalizer.base(), cp_len, "header_len", True, 1)
        self.digital_ofdm_chanest_vcvc_0 = digital.ofdm_chanest_vcvc(sync_word1, sync_word2, 1, 0, 1, False)
        self.digital_header_payload_demux_0 = digital.header_payload_demux(
            3,
            fft_len,
            cp_len,
            length_tag_key,
            "",
            True,
            gr.sizeof_gr_complex,
            "rx_time",
            int(samp_rate),
            ["rx_freq"],
            0)
        self.digital_constellation_decoder_cb_1 = digital.constellation_decoder_cb(payload_mod.base())
        self.digital_constellation_decoder_cb_0 = digital.constellation_decoder_cb(header_mod.base())
        self.blocks_skiphead_0 = blocks.skiphead(gr.sizeof_gr_complex*1, (int(samp_rate*skip_time)))
        self.blocks_repack_bits_bb_0 = blocks.repack_bits_bb(payload_mod.bits_per_symbol(), 8, packet_length_tag_key, True, gr.GR_LSB_FIRST)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_message_strobe_0 = blocks.message_strobe(pmt.cons(pmt.make_dict(), pmt.to_pmt(np.array(byte_sequence, dtype=np.uint8))), int(1000))
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, (fft_len+cp_len))
        self.analog_frequency_modulator_fc_0 = analog.frequency_modulator_fc((-2.0/fft_len))


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_message_strobe_0, 'strobe'), (self.embedded_packet_comp, 'Expected'))
        self.msg_connect((self.digital_packet_headerparser_b_0, 'header_data'), (self.digital_header_payload_demux_0, 'header_data'))
        self.msg_connect((self.pdu_tagged_stream_to_pdu_0, 'pdus'), (self.embedded_packet_comp, 'RX in'))
        self.connect((self.analog_frequency_modulator_fc_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_delay_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.digital_header_payload_demux_0, 0))
        self.connect((self.blocks_repack_bits_bb_0, 0), (self.pdu_tagged_stream_to_pdu_0, 0))
        self.connect((self.blocks_skiphead_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.blocks_skiphead_0, 0), (self.digital_ofdm_sync_sc_cfb_0, 0))
        self.connect((self.digital_constellation_decoder_cb_0, 0), (self.digital_packet_headerparser_b_0, 0))
        self.connect((self.digital_constellation_decoder_cb_1, 0), (self.blocks_repack_bits_bb_0, 0))
        self.connect((self.digital_header_payload_demux_0, 0), (self.fft_vxx_0_0, 0))
        self.connect((self.digital_header_payload_demux_0, 1), (self.fft_vxx_1, 0))
        self.connect((self.digital_ofdm_chanest_vcvc_0, 0), (self.digital_ofdm_frame_equalizer_vcvc_0, 0))
        self.connect((self.digital_ofdm_frame_equalizer_vcvc_0, 0), (self.digital_ofdm_serializer_vcc_header, 0))
        self.connect((self.digital_ofdm_frame_equalizer_vcvc_1, 0), (self.digital_ofdm_serializer_vcc_payload, 0))
        self.connect((self.digital_ofdm_serializer_vcc_header, 0), (self.digital_constellation_decoder_cb_0, 0))
        self.connect((self.digital_ofdm_serializer_vcc_payload, 0), (self.digital_constellation_decoder_cb_1, 0))
        self.connect((self.digital_ofdm_sync_sc_cfb_0, 0), (self.analog_frequency_modulator_fc_0, 0))
        self.connect((self.digital_ofdm_sync_sc_cfb_0, 1), (self.digital_header_payload_demux_0, 1))
        self.connect((self.fft_vxx_0_0, 0), (self.digital_ofdm_chanest_vcvc_0, 0))
        self.connect((self.fft_vxx_1, 0), (self.digital_ofdm_frame_equalizer_vcvc_1, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_skiphead_0, 0))


    def get_bps(self):
        return self.bps

    def set_bps(self, bps):
        self.bps = bps
        self.set_byte_amount(self.frame_amount * len(self.occupied_carriers[0]) // (8//self.bps))
        self.set_payload_mod(digital.qam_constellation(constellation_points=2**self.bps,differential=False,mod_code=digital.utils.mod_codes.GRAY_CODE) if (self.bps in [2,4,8]) else digital.constellation_bpsk())

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len
        self.set_cp_len(self.fft_len//4)
        self.set_header_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len,header_mod.base(),self.occupied_carriers,self.pilot_carriers,self.pilot_symbols))
        self.set_payload_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, payload_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols, 1))
        self.analog_frequency_modulator_fc_0.set_sensitivity((-2.0/self.fft_len))
        self.blocks_delay_0.set_dly(int((self.fft_len+self.cp_len)))

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_source_0.set_center_freq(self.freq, 0)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.uhd_usrp_source_0.set_gain(self.gain, 0)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_skip_time(self):
        return self.skip_time

    def set_skip_time(self, skip_time):
        self.skip_time = skip_time

    def get_occupied_carriers(self):
        return self.occupied_carriers

    def set_occupied_carriers(self, occupied_carriers):
        self.occupied_carriers = occupied_carriers
        self.set_byte_amount(self.frame_amount * len(self.occupied_carriers[0]) // (8//self.bps))
        self.set_header_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len,header_mod.base(),self.occupied_carriers,self.pilot_carriers,self.pilot_symbols))
        self.set_header_formatter(digital.packet_header_ofdm(self.occupied_carriers, n_syms=1, len_tag_key=self.length_tag_key, frame_len_tag_key=self.length_tag_key, bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=payload_mod.bits_per_symbol(), scramble_header=True))
        self.set_payload_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, payload_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols, 1))

    def get_frame_amount(self):
        return self.frame_amount

    def set_frame_amount(self, frame_amount):
        self.frame_amount = frame_amount
        self.set_byte_amount(self.frame_amount * len(self.occupied_carriers[0]) // (8//self.bps))

    def get_pilot_symbols(self):
        return self.pilot_symbols

    def set_pilot_symbols(self, pilot_symbols):
        self.pilot_symbols = pilot_symbols
        self.set_header_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len,header_mod.base(),self.occupied_carriers,self.pilot_carriers,self.pilot_symbols))
        self.set_payload_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, payload_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols, 1))

    def get_pilot_carriers(self):
        return self.pilot_carriers

    def set_pilot_carriers(self, pilot_carriers):
        self.pilot_carriers = pilot_carriers
        self.set_header_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len,header_mod.base(),self.occupied_carriers,self.pilot_carriers,self.pilot_symbols))
        self.set_payload_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, payload_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols, 1))

    def get_payload_mod(self):
        return self.payload_mod

    def set_payload_mod(self, payload_mod):
        self.payload_mod = payload_mod

    def get_length_tag_key(self):
        return self.length_tag_key

    def set_length_tag_key(self, length_tag_key):
        self.length_tag_key = length_tag_key
        self.set_header_formatter(digital.packet_header_ofdm(self.occupied_carriers, n_syms=1, len_tag_key=self.length_tag_key, frame_len_tag_key=self.length_tag_key, bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=payload_mod.bits_per_symbol(), scramble_header=True))

    def get_header_mod(self):
        return self.header_mod

    def set_header_mod(self, header_mod):
        self.header_mod = header_mod

    def get_byte_amount(self):
        return self.byte_amount

    def set_byte_amount(self, byte_amount):
        self.byte_amount = byte_amount
        self.set_byte_sequence([np.random.randint(256) for x in range(self.byte_amount)] if np.random.seed(0) is None else 0)

    def get_sync_word2(self):
        return self.sync_word2

    def set_sync_word2(self, sync_word2):
        self.sync_word2 = sync_word2

    def get_sync_word1(self):
        return self.sync_word1

    def set_sync_word1(self, sync_word1):
        self.sync_word1 = sync_word1

    def get_rolloff(self):
        return self.rolloff

    def set_rolloff(self, rolloff):
        self.rolloff = rolloff

    def get_payload_equalizer(self):
        return self.payload_equalizer

    def set_payload_equalizer(self, payload_equalizer):
        self.payload_equalizer = payload_equalizer

    def get_packet_length_tag_key(self):
        return self.packet_length_tag_key

    def set_packet_length_tag_key(self, packet_length_tag_key):
        self.packet_length_tag_key = packet_length_tag_key

    def get_header_formatter(self):
        return self.header_formatter

    def set_header_formatter(self, header_formatter):
        self.header_formatter = header_formatter

    def get_header_equalizer(self):
        return self.header_equalizer

    def set_header_equalizer(self, header_equalizer):
        self.header_equalizer = header_equalizer

    def get_cp_len(self):
        return self.cp_len

    def set_cp_len(self, cp_len):
        self.cp_len = cp_len
        self.blocks_delay_0.set_dly(int((self.fft_len+self.cp_len)))

    def get_byte_sequence(self):
        return self.byte_sequence

    def set_byte_sequence(self, byte_sequence):
        self.byte_sequence = byte_sequence
        self.blocks_message_strobe_0.set_msg(pmt.cons(pmt.make_dict(), pmt.to_pmt(np.array(self.byte_sequence, dtype=np.uint8))))



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-b", "--bps", dest="bps", type=intx, default=4,
        help="Set Bits per Symbol [default=%(default)r]")
    parser.add_argument(
        "-l", "--fft-len", dest="fft_len", type=intx, default=64,
        help="Set FFT Length [default=%(default)r]")
    parser.add_argument(
        "-f", "--freq", dest="freq", type=eng_float, default=eng_notation.num_to_str(float(2500000000)),
        help="Set Center frequency [default=%(default)r]")
    parser.add_argument(
        "-g", "--gain", dest="gain", type=eng_float, default=eng_notation.num_to_str(float(20)),
        help="Set RX Gain [default=%(default)r]")
    parser.add_argument(
        "-r", "--samp-rate", dest="samp_rate", type=intx, default=1000000,
        help="Set Sample rate [default=%(default)r]")
    parser.add_argument(
        "-s", "--skip-time", dest="skip_time", type=eng_float, default=eng_notation.num_to_str(float(0.5)),
        help="Set skip stime [default=%(default)r]")
    return parser


def main(top_block_cls=ofdm_rx_example, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(bps=options.bps, fft_len=options.fft_len, freq=options.freq, gain=options.gain, samp_rate=options.samp_rate, skip_time=options.skip_time)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    tb.wait()


if __name__ == '__main__':
    main()
