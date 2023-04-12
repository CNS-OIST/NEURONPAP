
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
        return pd.read_csv(fName,header=None)
    else:
        return None

def func(dir,zoom=False,ext=False):
    i = loadFile(os.path.join(dir,'iFile.dat'))
    v = loadFile(os.path.join(dir,'vFile.dat'))
    mem = loadFile(os.path.join(dir,'iFileMem.dat'))
    t = loadFile(os.path.join(dir,'tFile.dat'))
    plt.plot(t,i)
    plt.ylabel('pA')
    if max(i.iloc[:, 0]) > 1:
        plt.ylim(0,25)
    plt.xlabel('ms')
    if zoom:
        plt.xlim(9.99,10.025)
    elif ext and max(t) >= 10000:
        plt.xlim(11,10000)
    plt.savefig(os.path.join(dir,'results.pdf'))
    plt.cla()
    plt.clf()
    if v is not None:
        plt.plot(t,v)
        plt.ylabel('mV')
        plt.xlabel('ms')
        print(max(v.iloc[:,0]))
        plt.ylim(-90,0)
        # if zoom or max(v.iloc[:, 0]) > -20: 
        #     plt.xlim(9.99,10.1)
        # elif ext and max(t) >= 10000:
        #     plt.xlim(11,10000)
            
        plt.savefig(os.path.join(dir,'resultsV.pdf'))
        plt.cla()
        plt.clf()
    if mem is not None:
        plt.plot(t,mem)
        plt.ylabel('pA')
        plt.xlabel('ms')
        # plt.ylim(-90,0)
        if zoom:
            plt.xlim(9.99,10.025)
        elif ext and max(t) >= 10000:
            plt.xlim(11,10000)

        plt.savefig(os.path.join(dir,'resultsIMem.pdf'))
        


if __name__ == "__main__":
    args = sys.argv
    func(args[1])
