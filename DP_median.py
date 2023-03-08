##############################################
#                                            #
# DP_median.py                               #
#                                            #
# Differentially private algorithm that      #
# returns the median of a list of numbers.   #
#                                            #
# Written by: Cheng-En Tsai                  #
#                                            #
##############################################

import scipy.stats

def sign(num: float) -> int:
    """ Returns the sign of num. """
    if num > 0:
        return 1
    if num == 0:
        return 0
    return -1

def q(x, y):
    """ q(x, y) is maximum when y is a median of x. """
    return -abs(sum(sign(y-entry) for entry in x))

def DP_median(
        x: list, epsilon: float, L: int, U: int, *, frac: int = 1
    ) -> float:
    """ Returns the noisy median of list x
        whose values are at least L and at most U.
    
        Implements Report-One-Sided-Noisy-Max algorithm
        that adds Exp(2/epsilon) noises.
        Searches values between L and U.
        When int is an integer greater than 1,
        shorten the intervals between candidate
        values down to 1/frac.
    """
    candidates = []
    for y in range((U-L)*frac):
        y = y / frac + L
        candidates.append(
            [q(x, y) + scipy.stats.expon.rvs(scale=2/epsilon), y]
        )
    candidates.append(
        [q(x, U) + scipy.stats.expon.rvs(scale=2/epsilon), U]
    )
    return max(candidates)[1]
