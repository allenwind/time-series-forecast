import numpy as np

def ma(series, window_size=5):
    # wiki: 
    # https://en.wikipedia.org/wiki/Moving-average_model
    
    values = []
    for i in range(window_size, len(series)):
        values.append(np.mean(series[i-window_size:i]))
    return np.array(values)

def cma(series):
    values = []
    cma0 = series[0]
    for i in range(len(series)-1):
        values.append(cma0)
        cma0 = (i*cma0 + values[i+1]) / (i+1)
    return np.array(values)

def wma(series, weights=None):
    if weights is None:
        weights = np.array([1, 2, 3, 4, 5])
        weights = weights / np.sum(weights)

    window_size = len(weights)
    values = []
    for i in range(window_size, len(series)):
        value = np.mean(weights * series[i-window_size:i])
        values.append(value)
    return np.array(values)

def ewma(series):
    pass

def dewma(series):
    pass
