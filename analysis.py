# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import pickle
from classResults import ResultsPAPModel
with open('resultsParallel.pickle','rb') as handle:
    results = pickle.load(handle)
    
for i,res in enumerate(results):
    if (i+1) %6  == 0 or i == 0:
        plt.scatter(list(res[0].vPAP),list(res[0].vSoma),c=list(res[0].time))
        plt.colorbar(label="Time (ms)")
        plt.title(f'branch L:{res[0].bLen}')
        plt.xlabel('PAP membrane potential (mV)')
        plt.ylabel('Soma membrane potential (mV)')
        # plt.ylim((-88.97,-88.965))
        plt.savefig(f'result{i}-IsolatedPAP.pdf')
        plt.cla()
        plt.clf()
        plt.plot(range(len(list(res[0].KoPAP))),list(res[0].KoPAP))
        plt.savefig(f'result{i}-Ko.pdf')
        plt.cla()
        plt.clf()
        
