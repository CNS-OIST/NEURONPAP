"""
Author: Joel Nakatani
Overview:

Parameters:
"""

from neuron import h
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class calibrateChannel:
    def __init__(self):
        h.load_file("stdgui.hoc")

        self.channelName = [
            'Kir',
            'TWIK',
            'NMDAR',
            # 'GluT'
        ]
        # self.expCurrName = {
        #     'Kir':,
        #     'TWIK':,
        #     'NMDAR':'Data/VClamp40.1Stim.dat',
        #     'GluT':
        # }
        self.expIVName = {
            'Kir':'Data/KirIV.csv',
            'TWIK':'Data/TWIKIV.csv',
            'NMDAR':'Data/NMDARIV.csv',
            'GluT':'Data/GluTrans.csv',
        }
        self.name2NEURON = {
            'Kir':('kir2','mm'),
            'TWIK':('twik','mm'),
            'NMDAR':('setNMDAs','pm'),
            'GluT':('setGluTs','pm')
        }
            

    def initModel(self,channel):
        h.xopen('./neuronHoc/simpleCheck.hoc')
        nrnName, chantype = self.name2NEURON[channel]
        # define record for channel
        setattr(self,channel,h.Vector())
        
        # Check channel type
        if chantype == 'mm':
            h.soma.insert(nrnName)
            getattr(self,channel).record(h.soma(0.5)._ref_ik)
        else:
            getattr(h,nrnName)(h.SectionList([h.soma]))
            if 'NMDA' in nrnName:
                listName = 'NMDAs'
                currName = '_ref_iNMDA'
            else:
                listName = 'GluTs'
                currName = '_ref_iGluT'
            sChannel = list(getattr(h,listName))[-1]
            # print(sChannel)
            getattr(self,channel).record(getattr(sChannel,currName))

        self.time = h.Vector()
        self.time.record(h._ref_t)

    def plotiCurve(self,channel,expSet,mdlSet):
        plt.cla()
        plt.clf()
        plt.scatter(*expSet,label='experiment')
        plt.plot(*mdlSet,label='model')
        plt.legend()
        plt.savefig(f'icurve{channel}.pdf')

    def getModeliCurve(self,channel,run=False,mV=None):
        if run:
            if mV != None:
                h.clampSwitch(0,mV)
                h.init()
                h.run()
                
        curr = np.array(getattr(self,channel))
        time = np.array(self.time)
        return time,curr

    def getExpiCurve(self,channel):
        fName = self.expIVName(channel)
        ext = fName.split('.')[-1]
        time = []
        curr = []
        if ext == 'dat':
            f1 = h.File()
            f1.ropen(fName)
            while not f1.eof():
                time.append(f1.scanvar()) 
                curr.append(f1.scanvar())
            f1.close()
        elif ext == 'csv':
            expData = pd.read_csv(fName)
            time = expData['time'].tolist()
            curr = expData['curr'].tolist()
        else:
            print(f'no file {fName}')

        curr = np.array(curr) / max(curr,key=abs)
        return np.array(time),curr
        
    def plotIVCurve(self,channel,expVolt,expCurr,mdlCurr):
        plt.cla()
        plt.clf()
        plt.scatter(expVolt,expCurr,label='Experiment')
        plt.plot(expVolt,mdlCurr,label='Model')
        plt.legend()
        plt.xlabel('Voltage (mV)')
        plt.ylabel('Normalized Current')
        plt.savefig(f'IVCurve{channel}.pdf')

    def getModelIVPoint(self,mV,channel):
        h.clampSwitch(0,mV)
        h.init()
        h.run()
        return max(np.array(getattr(self,channel)),key=abs)
    
    def getExpIVCurve(self,channel):
        fName = self.expIVName[channel]
        ext = fName.split('.')[-1]
        volt = []
        curr = []
        if ext == 'dat':
            f1 = h.File()
            f1.ropen(fName)
            while not f1.eof():
                volt.append(f1.scanvar())
                curr.append(f1.scanvar()) 
            f1.close()
        elif ext == 'csv':
            expData = pd.read_csv(fName)
            volt = expData['volt'].tolist()
            curr = expData['curr'].tolist()
        else:
            print(f'no file {fName}')

        curr = np.array(curr) / abs(curr[self.index])
        return np.array(volt),curr                

    def IVCurve(self,channel,vStep=None):
        expVolt,expCurr = self.getExpIVCurve(channel)
        mdlCurr = []
        for v in expVolt:
            self.initModel(channel)
            mdlCurr.append(self.getModelIVPoint(v,channel))
            if v == vStep:
                expSet = self.getExpiCurve(channel)
                mdlSet = self.getModeliCurve(channel)
                self.plotiCurve(channel,expSet,mdlSet)
        mdlCurr = np.array(mdlCurr) / abs(mdlCurr[self.index])
        return expVolt,expCurr,mdlCurr

if __name__ == "__main__":
    chans = calibrateChannel()
    for channel in chans.channelName:
        chans.index = 0
        resSet = chans.IVCurve(channel)
        chans.plotIVCurve(channel,*resSet)
    
