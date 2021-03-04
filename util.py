import numpy as np
from scipy.signal import argrelextrema
from scipy import interpolate
import librosa

def local_max(Sm,lo_range,thres):
    local_max = []
    for i in range(len(Sm)):
        if Sm[i] > thres:
            if i == 0:
                if Sm[i] > max(Sm[i+1:i+lo_range+1]):
                    local_max.append(i)
            elif i == len(Sm)-1:
                if Sm[i] > max(Sm[i-lo_range:i]):
                    local_max.append(i)
            elif i > 0 and i < lo_range:
                if Sm[i] > max(Sm[:i]) and Sm[i] > max(Sm[i+1:i+lo_range+1]):
                    local_max.append(i)
            elif i < len(Sm)-1 and i > len(Sm) - lo_range:
                if Sm[i] > max(Sm[i-lo_range:i]) and Sm[i] > max(Sm[i+1:]):
                    local_max.append(i)
            else:
                if Sm[i] > max(Sm[i-lo_range:i]) and Sm[i] > max(Sm[i+1:i+lo_range+1]):
                    local_max.append(i)
    return np.asarray(local_max)

def amplitude(row,length,hop_len):
    x = np.arange(row.shape[0]+1)
    y = np.pad(row,(0,1),'constant',constant_values=row[-1])
    f = interpolate.interp1d(x, y)

    xnew = np.arange(0,row.shape[0],1/hop_len)[:length]
    ynew = f(xnew)
    
    return ynew

def adaptive_attack(o,x,thres,ptype='log'):
    '''
    Parameters
    ----------
    o: array like
        The oringal signal.
    x: array like
        The sysnthesised signal.
    thres: int
        Threshold(dB) of the attack preparation.
    type: str, optional
        The type of preparation {'log',linear'}, default='log'

    Returns
    -------
    x: array like
        Output signal.
    '''
    thres = 10**(thres/20)
    s = (abs(o)>0).argmax()
    e = (abs(o)>thres).argmax()
    print(s,e)
    att = np.zeros(e)
    if ptype is 'linear':
        att[s:e] = np.linspace(0,1.,e-s)
    elif ptype is 'log':
        att[s:e] = np.logspace(-5,0,e-s)
    else:
        raise NameError("Unknown type used.")
    x[:e] *= att
    return x