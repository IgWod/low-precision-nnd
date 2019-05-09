def bpsk(bits):
    """Modulate bits into real channel symbols using BPSK scheme.

    Args:
        bits: array of 0s and 1s.

    Returns:
        Array of real values where every input bit was mapped to the real number.
    """
    return 1 - 2 * bits
