# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import pickle
from classResults import ResultsPAPModel
with open('resultsParallel.pickle','rb') as handle:
    results = pickle.load(handle)
    
for i,res in enumerate(results):
    if (i+1) %6  == 0:
        plt.scatter(list(res[0].vPAP),list(res[0].vSoma),c=np.array(range(len(list(res[0].vPAP))))/len((list(res[0].vPAP))))
        plt.colorbar(label="Time")
        plt.title(f'branch L:{res[0].bLen}')
        plt.xlabel('PAP membrane potential (mV)')
        plt.ylabel('Soma membrane potential (mV)')
        # plt.ylim((-89,-88))
        plt.savefig(f'result{i}-IsolatedPAP.pdf')
        plt.cla()
        plt.clf()
        plt.plot(range(len(list(res[0].KoPAP))),list(res[0].KoPAP))
        plt.savefig(f'result{i}-Ko.pdf')
        plt.cla()
        plt.clf()
        
