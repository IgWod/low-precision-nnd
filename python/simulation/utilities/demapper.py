def bpsk(y, stddev):
    """Demodulate noisy channel symbols into LLRs.

    The very simply demodulator for BPSK modulation in AWGN channel.

    Args:
         y: array of noisy channel symbols
         stddev: stddev of the AWGN channel

    Returns:
        Array of LLRs.
    """
    return 2 * y / (stddev ** 2)
