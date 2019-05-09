import numpy as np

from references.polar.sequence import Sequence


# noinspection PyPep8Naming
class SCDecoder:
    """Successive Cancellation Decoder for polar codes.

    The implementation is based on the algorithm described in H. Vangala,
    E. Viterbo and Y. Hong, \"Permuted successive cancellation decoder for polar codes\"
    but it uses LLRs as an input to the decoder.
    """

    def __init__(self, N, K):
        """Initialize the decoder for the specific codeword and message size.

        The order of the decoder is used to calculate the codeword size and
        create the polar sequence.

        Args:
            N: Codeword size.
            K: Message length.

        Raises:
            ValueError: At least one argument is invalid.
        """
        if N > 1024 or N < 4:
            raise ValueError('Codeword size has to be in range [4, 1024]!')

        self._n = int(np.log2(N))
        self._N = N
        self._K = K

        if self._K > self._N:
            raise ValueError('Message size cannot exceed codeword size!')

        self._polar_sequence = Sequence(self._N, self._K)

    def decode(self, llrs_in):
        """Perform the decoding using given LLRs.

        The function uses the successive cancellation method to decode the message
        from given LLRs.

        Args:
            llrs_in: Log-Likehood Ratios for each codeword bit.

        Raises:
            ValueError: The argument is invalid
        """
        llrs_in = llrs_in

        if len(llrs_in) != self._N:
            raise ValueError('Number of LLRs has to be equal to ' + self._N + '!')

        bits = np.empty([self._N, self._n + 1])
        bits.fill(np.nan)

        llrs = np.empty([self._N, self._n + 1])
        llrs.fill(np.nan)

        llrs[:, self._n] = llrs_in

        def bitrev(n):
            return int(('{:0' + str(self._n) + 'b}').format(n)[::-1], 2)

        frozen_positions = self._polar_sequence.frozen_positions

        for bit in range(self._N):
            rbit = bitrev(bit)
            self._update_priori(llrs, bits, rbit, 0)
            if rbit in frozen_positions:
                bits[rbit][0] = 0
            else:
                bits[rbit][0] = 0 if llrs[rbit][0] >= 0 else 1
            self._update_posteriori(bits, rbit, 0)

        decoded_codeword = bits[:, 0]
        return np.int8(decoded_codeword[self._polar_sequence.unfrozen_positions])

    def _update_priori(self, llrs, bits, position, stage):
        block_size = 2 ** (self._n - stage)
        position_in_block = position % block_size
        offset = int(block_size / 2)

        if position_in_block < offset:
            if np.isnan(llrs[position][stage + 1]):
                self._update_priori(llrs, bits, position, stage + 1)
            if np.isnan(llrs[position + offset][stage + 1]):
                self._update_priori(llrs, bits, position + offset, stage + 1)
            llrs[position][stage] = np.sign(llrs[position][stage + 1]) * \
                np.sign(llrs[position + offset][stage + 1]) * \
                min(abs(llrs[position][stage + 1]), abs(llrs[position + offset][stage + 1]))
        else:
            llrs[position][stage] = llrs[position][stage + 1] + \
                                    ((-1) ** (bits[position - offset][stage] % 2)) * llrs[position - offset][stage + 1]

    def _update_posteriori(self, bits, position, stage):
        block_size = 2 ** (self._n - stage)
        position_in_block = position % block_size
        offset = int(block_size / 2)

        if position_in_block < offset or stage >= self._n:
            return

        bits[position - offset][stage + 1] = bits[position][stage] != \
            bits[position - offset][stage]
        bits[position][stage + 1] = bits[position][stage]
        self._update_posteriori(bits, position, stage + 1)
        self._update_posteriori(bits, position - offset, stage + 1)
