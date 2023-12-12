#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Example OFDM TX
# Author: Cyrille Morin
# GNU Radio version: 3.10.6.0

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




class ofdm_tx_example(gr.top_block):

    def __init__(self, bps=4, fft_len=64, freq=2500000000, gain=20, packet_freq=0.5, samp_rate=1000000):
        gr.top_block.__init__(self, "Example OFDM TX", catch_exceptions=True)

        ##################################################
        # Parameters
        ##################################################
        self.bps = bps
        self.fft_len = fft_len
        self.freq = freq
        self.gain = gain
        self.packet_freq = packet_freq
        self.samp_rate = samp_rate

        ##################################################
        # Variables
        ##################################################
        self.occupied_carriers = occupied_carriers = (list(range(-26, -21)) + list(range(-20, -7)) + list(range(-6, 0)) + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27)),)
        self.frame_amount = frame_amount = 20
        self.payload_mod = payload_mod = digital.qam_constellation(constellation_points=2**bps,differential=False,mod_code=digital.utils.mod_codes.GRAY_CODE) if (bps in [2,4,8]) else digital.constellation_bpsk()
        self.length_tag_key = length_tag_key = "frame_len"
        self.header_mod = header_mod = digital.constellation_qpsk()
        self.byte_amount = byte_amount = frame_amount * len(occupied_carriers[0]) // (8//bps)
        self.sync_word2 = sync_word2 = [0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0]
        self.sync_word1 = sync_word1 = [0., 0., 0., 0., 0., 0., 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 0., 0., 0., 0., 0.]
        self.rolloff = rolloff = 0
        self.pilot_symbols = pilot_symbols = ((1, 1, 1, -1,),)
        self.pilot_carriers = pilot_carriers = ((-21, -7, 7, 21,),)
        self.header_formatter = header_formatter = digital.packet_header_ofdm(occupied_carriers, n_syms=1, len_tag_key=length_tag_key, frame_len_tag_key=length_tag_key, bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=payload_mod.bits_per_symbol(), scramble_header=True)
        self.cp_len = cp_len = fft_len//4
        self.byte_sequence = byte_sequence = [np.random.randint(256) for x in range(byte_amount)] if np.random.seed(0) is None else 0

        ##################################################
        # Blocks
        ##################################################

        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(("", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
            length_tag_key,
        )
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0.set_time_unknown_pps(uhd.time_spec(0))

        self.uhd_usrp_sink_0.set_center_freq(freq, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_gain(gain, 0)
        self.pdu_pdu_to_tagged_stream_0 = pdu.pdu_to_tagged_stream(gr.types.byte_t, length_tag_key)
        self.fft_vxx_0 = fft.fft_vcc(fft_len, False, (), True, 1)
        self.digital_packet_headergenerator_bb_0 = digital.packet_headergenerator_bb(header_formatter.base(), length_tag_key)
        self.digital_ofdm_cyclic_prefixer_0 = digital.ofdm_cyclic_prefixer(
            fft_len,
            fft_len + cp_len,
            rolloff,
            length_tag_key)
        self.digital_ofdm_carrier_allocator_cvc_0 = digital.ofdm_carrier_allocator_cvc( fft_len, occupied_carriers, pilot_carriers, pilot_symbols, (sync_word1, sync_word2), length_tag_key, True)
        self.digital_chunks_to_symbols_xx_0_0 = digital.chunks_to_symbols_bc(np.array(payload_mod.points())/np.mean(np.abs(np.array(payload_mod.points()))), 1)
        self.digital_chunks_to_symbols_xx_0 = digital.chunks_to_symbols_bc(np.array(header_mod.points())/np.mean(np.abs(np.array(header_mod.points()))), 1)
        self.blocks_tagged_stream_mux_1 = blocks.tagged_stream_mux(gr.sizeof_gr_complex*1, length_tag_key, 0)
        self.blocks_tagged_stream_mux_0 = blocks.tagged_stream_mux(gr.sizeof_gr_complex*1, length_tag_key, 0)
        self.blocks_stream_to_tagged_stream_0 = blocks.stream_to_tagged_stream(gr.sizeof_gr_complex, 1, fft_len, length_tag_key)
        self.blocks_repack_bits_bb_0 = blocks.repack_bits_bb(8, payload_mod.bits_per_symbol(), length_tag_key, False, gr.GR_LSB_FIRST)
        self.blocks_null_source_0 = blocks.null_source(gr.sizeof_gr_complex*1)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(0.06)
        self.blocks_message_strobe_0 = blocks.message_strobe(pmt.cons(pmt.make_dict(), pmt.to_pmt(np.array(byte_sequence, dtype=np.uint8))), (int(1000*packet_freq)))


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_message_strobe_0, 'strobe'), (self.pdu_pdu_to_tagged_stream_0, 'pdus'))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_tagged_stream_mux_1, 1))
        self.connect((self.blocks_null_source_0, 0), (self.blocks_stream_to_tagged_stream_0, 0))
        self.connect((self.blocks_repack_bits_bb_0, 0), (self.digital_chunks_to_symbols_xx_0_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0, 0), (self.blocks_tagged_stream_mux_1, 0))
        self.connect((self.blocks_tagged_stream_mux_0, 0), (self.digital_ofdm_carrier_allocator_cvc_0, 0))
        self.connect((self.blocks_tagged_stream_mux_1, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.blocks_tagged_stream_mux_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0_0, 0), (self.blocks_tagged_stream_mux_0, 1))
        self.connect((self.digital_ofdm_carrier_allocator_cvc_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.digital_ofdm_cyclic_prefixer_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.digital_packet_headergenerator_bb_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.digital_ofdm_cyclic_prefixer_0, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_0, 0), (self.blocks_repack_bits_bb_0, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_0, 0), (self.digital_packet_headergenerator_bb_0, 0))


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
        self.blocks_stream_to_tagged_stream_0.set_packet_len(self.fft_len)
        self.blocks_stream_to_tagged_stream_0.set_packet_len_pmt(self.fft_len)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_sink_0.set_center_freq(self.freq, 0)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.uhd_usrp_sink_0.set_gain(self.gain, 0)

    def get_packet_freq(self):
        return self.packet_freq

    def set_packet_freq(self, packet_freq):
        self.packet_freq = packet_freq
        self.blocks_message_strobe_0.set_period((int(1000*self.packet_freq)))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)

    def get_occupied_carriers(self):
        return self.occupied_carriers

    def set_occupied_carriers(self, occupied_carriers):
        self.occupied_carriers = occupied_carriers
        self.set_byte_amount(self.frame_amount * len(self.occupied_carriers[0]) // (8//self.bps))
        self.set_header_formatter(digital.packet_header_ofdm(self.occupied_carriers, n_syms=1, len_tag_key=self.length_tag_key, frame_len_tag_key=self.length_tag_key, bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=payload_mod.bits_per_symbol(), scramble_header=True))

    def get_frame_amount(self):
        return self.frame_amount

    def set_frame_amount(self, frame_amount):
        self.frame_amount = frame_amount
        self.set_byte_amount(self.frame_amount * len(self.occupied_carriers[0]) // (8//self.bps))

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

    def get_pilot_symbols(self):
        return self.pilot_symbols

    def set_pilot_symbols(self, pilot_symbols):
        self.pilot_symbols = pilot_symbols

    def get_pilot_carriers(self):
        return self.pilot_carriers

    def set_pilot_carriers(self, pilot_carriers):
        self.pilot_carriers = pilot_carriers

    def get_header_formatter(self):
        return self.header_formatter

    def set_header_formatter(self, header_formatter):
        self.header_formatter = header_formatter

    def get_cp_len(self):
        return self.cp_len

    def set_cp_len(self, cp_len):
        self.cp_len = cp_len

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
        "-t", "--packet-freq", dest="packet_freq", type=eng_float, default=eng_notation.num_to_str(float(0.5)),
        help="Set Packet Wait Time [default=%(default)r]")
    parser.add_argument(
        "-r", "--samp-rate", dest="samp_rate", type=intx, default=1000000,
        help="Set Sample rate [default=%(default)r]")
    return parser


def main(top_block_cls=ofdm_tx_example, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(bps=options.bps, fft_len=options.fft_len, freq=options.freq, gain=options.gain, packet_freq=options.packet_freq, samp_rate=options.samp_rate)

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
