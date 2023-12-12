"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr
import pmt


class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block - 
    Compares a recieved packet message with a reference.
    Expects PDUs with byte samples
    """

    def __init__(self):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='Packet Comparator',   # will show up in GRC
            in_sig=[],
            out_sig=[]
        )

        self.logger = gr.logger(self.alias())
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).

        self.message_port_register_in(pmt.intern("Expected"))
        self.set_msg_handler(pmt.intern("Expected"), self.handle_expected)
        self.message_port_register_in(pmt.intern("RX in"))
        self.set_msg_handler(pmt.intern("RX in"), self.handle_rx_in)

        self.reference_vector = None

    def handle_rx_in(self, msg):
        if self.reference_vector is None:
            self.logger.error("Received a packet, but there is no reference to compare to")
            return
        
        py_msg = pmt.to_python(msg)
        data = py_msg[1].astype(np.uint8, copy=False)

        if len(data) != len(self.reference_vector):
            self.logger.warn("Packet with length different from reference, counted as erroneous")
            return
        if "packet_num" in py_msg[0]:
            packet_num = py_msg[0]['packet_num']
        else:
            packet_num = "Not Found"
        xored = np.bitwise_xor(data, self.reference_vector)
        bit_errors = np.unpackbits(xored)
        ber = np.mean(bit_errors)

        self.logger.info(f"Received packet num {packet_num} with a BER of {ber:.3g}")


    def handle_expected(self, msg):
        self.reference_vector = pmt.to_python(msg)[1].astype(np.uint8, copy=False)
        # self.logger.info("Recieved a reference message")

    def work(self, input_items, output_items):
        """example: multiply with constant"""
        output_items[0][:] = input_items[0] * self.example_param
        return len(output_items[0])
