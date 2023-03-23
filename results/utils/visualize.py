"""
Author: Joel Nakatani
Overview:

Parameters:
"""

from neuron import h
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def loadFile(fName):
    if os.path.isfile(fName):
        return pd.read_csv(fName)
    else:
        return None

def func(dir):
    i = loadFile(os.path.join(dir,'iFile.dat'))
    v = loadFile(os.path.join(dir,'vFile.dat'))
    t = loadFile(os.path.join(dir,'tFile.dat'))
    plt.plot(t,i)
    plt.ylabel('nA')
    # plt.ylim(-10,0)
    plt.xlabel('ms')
    plt.savefig(os.path.join(dir,'results.pdf'))
    plt.cla()
    plt.clf()
    if v is not None:
        plt.plot(t,v)
        plt.ylabel('mV')
        plt.xlabel('ms')
        # plt.ylim(-90,0)
        plt.savefig(os.path.join(dir,'resultsV.pdf'))
        


if __name__ == "__main__":
    args = sys.argv
    func(args[1])
