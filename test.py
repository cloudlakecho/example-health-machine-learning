
# test.py - various test with plot
# Cloud Cho, November 11, 2020


# Reference
#   Poisson: https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html
#   Beta: https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html
#   Date: https://stackoverflow.com/a/26819041/5595995

import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta

POISSON = False
BETA = False
DATE = True

if (POISSON):
    s = np.random.poisson(5, 100)
    count, bins, ignored = plt.hist(s, 2, density=True)
    plt.show()

if (BETA):
    s = np.random.beta(5, 20, 100)
    count, bins, ignored = plt.hist(s, 2, density=True)
    plt.show()

if (DATE):
    start = datetime.now()
    end = start + timedelta(days=1)
    start = "1994-12-28"
    end = "2012-07-20"
    start = datetime(1994, 12, 28)
    end = datetime(2012, 7, 20)
    random_date = start + (end - start) * random.random()
    print (start.strftime("%Y-%m-%d"))
    print (end.strftime("%Y-%m-%d"))
    print (random_date.strftime("%Y-%m-%d"))
