#!/usr/bin/env python
##################################################
# Gnuradio Python Flow Graph
# Title: Test Fft Web
# Generated: Wed Jul 30 16:42:11 2014
##################################################

from fft_web import fft_web
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import uhd
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import time

class test_fft_web(gr.top_block):

    def __init__(self, samp_rate=1e6, gain=0, fft_size=1024, frame_rate=10, freq=2.49e9):
        gr.top_block.__init__(self, "Test Fft Web")

        ##################################################
        # Parameters
        ##################################################
        self.samp_rate = samp_rate
        self.gain = gain
        self.fft_size = fft_size
        self.frame_rate = frame_rate
        self.freq = freq

        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_source_0 = uhd.usrp_source(
        	"addr=192.168.10.2",
        	uhd.stream_args(
        		cpu_format="fc32",
        		channels=range(1),
        	),
        )
        self.uhd_usrp_source_0.set_clock_source("internal", 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_center_freq(freq, 0)
        self.uhd_usrp_source_0.set_gain(gain, 0)
        self.fft_web_0 = fft_web(
            fft_size=fft_size,
            sample_rate=samp_rate,
            frame_rate=frame_rate,
            ip_address="srvwww.cortexlab.fr",
            port=6663,
        )

        ##################################################
        # Connections
        ##################################################
        self.connect((self.uhd_usrp_source_0, 0), (self.fft_web_0, 0))



    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.fft_web_0.set_sample_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.uhd_usrp_source_0.set_gain(self.gain, 0)

    def get_fft_size(self):
        return self.fft_size

    def set_fft_size(self, fft_size):
        self.fft_size = fft_size
        self.fft_web_0.set_fft_size(self.fft_size)

    def get_frame_rate(self):
        return self.frame_rate

    def set_frame_rate(self, frame_rate):
        self.frame_rate = frame_rate
        self.fft_web_0.set_frame_rate(self.frame_rate)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_source_0.set_center_freq(self.freq, 0)

if __name__ == '__main__':
    parser = OptionParser(option_class=eng_option, usage="%prog: [options]")
    parser.add_option("-R", "--samp-rate", dest="samp_rate", type="eng_float", default=eng_notation.num_to_str(1e6),
        help="Set samp_rate [default=%default]")
    parser.add_option("-G", "--gain", dest="gain", type="eng_float", default=eng_notation.num_to_str(0),
        help="Set gain [default=%default]")
    parser.add_option("-S", "--fft-size", dest="fft_size", type="intx", default=1024,
        help="Set fft_size [default=%default]")
    parser.add_option("-H", "--frame-rate", dest="frame_rate", type="intx", default=10,
        help="Set frame_rate [default=%default]")
    parser.add_option("-F", "--freq", dest="freq", type="eng_float", default=eng_notation.num_to_str(2.49e9),
        help="Set freq [default=%default]")
    (options, args) = parser.parse_args()
    tb = test_fft_web(samp_rate=options.samp_rate, gain=options.gain, fft_size=options.fft_size, frame_rate=options.frame_rate, freq=options.freq)
    tb.start()
    tb.wait()

