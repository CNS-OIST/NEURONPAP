"""
Author: Joel Nakatani
Overview:
Optimization using optuna
created for mpirun of steps.

Remove mpi4py dependence for single process run.
"""

import optuna
import logging
import sys
import argparse
import plotly

from neuron import h,load_mechanisms
import os
import pandas as pd


def setArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optiDB",
        "-o",
        dest="study_name",
        type=str,
        default="PAP_optimum",
        help="Used to specify db file for Optimization.",
    )  # sbmlFile arg
    parser.add_argument(
        "-s",
        "--saveDir",
        type=str,
        dest="saveDir",
        default=None,
        help="Used to specify directory to save data. Set directory SAVEDIR after option. Defaults to data_tripartite directory in tripytite.",
    )  # saveDir arg
    parser.add_argument(
        "--saveBest",
        action="store_true",
        dest="saveBest",
        default=False,
        help="Used to setup best value for optimization in dat file"
    )  # saveDir arg
    parser.add_argument(
        "--print",
        action="store_true",
        dest="printStudy",
        default=False,
        help="Used to print study"
    )  # saveDir arg

    

    return parser

def saveSynWeight(w,file='../results/optimize/optW.dat'):
    fName = open(file,'w')
    fName.write(f'{w}\n')
    fName.close()
    fName = open(file,'r')
    return fName.read()

def saveDelta(d,file='../results/optimize/optDelta.dat'):
    fName = open(file,'w')
    fName.write(f'{d}\n')
    fName.close()
    fName = open(file,'r')
    return fName.read()


def saveTau(t,A,B,i,dName='../results/optimize/'):
    fName = os.path.join(
        dName,
        f'optT{i}.dat'
    )
    sFile = open(fName,'w')

    if i > 1:
        sFile.write(f'{t}\n')
        sFile.write(f'{A}\n')
        sFile.write(f'{B}\n')
    else:
        sFile.write(f'{t}\n')
        sFile.write(f'{A}\n')
    sFile.close()
    sFile = open(fName,'r')
    return sFile.readlines()

def relLikelihood(expData='../src/Data/VClamp40.1Stim.dat',scale=1):
    # Read simulated data
    iCurve = pd.read_csv('iFile.dat',header=None,names=['current'])
    tSample = pd.read_csv('tFile.dat',header=None,names=['t'])

    # Read answer data
    fName = open(expData,'r')
    lines = fName.readlines()
    tList = []
    vList = []
    for line in lines[2:]:
        t,v = line.strip().split()
        tList.append(float(t))
        vList.append(float(v)*scale)
    # Get RSS
    # itResult =[(t,i) for i,t in zip(iCurve['current'],tSample['t']) if t in tList]
    # vList = [v for v in vList]
    # rss = 0
    # for i,current in enumerate(itResult):
    #     t,c = current
    #     rss += (c - vList[i]) ** 2
    rss = abs(max(iCurve['current']) * 1000 - 2)
    return rss 

# def resLikelihood(expData='./Data/VClamp40.1Stim.dat'):
#     scale = 1 # Surface area of patch membrane 2um to PAP area
#     iCurve = pd.read_csv('iFile.dat',header=None,names=['current'])
#     tSample = pd.read_csv('tFile.dat',header=None,names=['t'])

#     fName = open(expData,'r')
#     lines = fName.readlines()
#     tList = []
#     vList = []
#     for line in lines[2:]:
#         t,v = line.strip().split()
#         tList.append(float(t))
#         vList.append(float(v))
#     itResult = [(t,i/scale) for i,t in zip(iCurve['current'],tSample['t']) if t in tList]
#     vList = [v for v in vList]
#     rss = 0
#     for i,current in enumerate(itResult):
#         t,c = current
#         rss += (c - vList[i]) ** 2
#     return rss

def objective(trial):
    """
    objective function used to run optimization.
    Input:
     Should be a trial instance created by optuna.
     (i.e. just use trial variable)

    Output:
     Score of function, model, etc.

    Depending on the .suggest_* method of the trial instance optuna will select a input.
    (Documentation:https://optuna.readthedocs.io/en/stable/reference/multi_objective/generated/optuna.multi_objective.trial.MultiObjectiveTrial.html?highlight=suggest)

    Use that input to run your simulation.
    Make sure that you save the end result of the simulation and also generate a score rating the reactions.

    Return:
     Return the evaluated score of the simulation.

    """
    # Save Synaptic weight
    SynWeight = trial.suggest_float("SynWeight", 0, 5)
    saveSynWeight(SynWeight)

    # # suggest Tau1
    # tau1 = trial.suggest_float("tau1", 0, 10)
    # wTau2 = trial.suggest_float("wTau2", 0, 0.1)
    # saveTau(tau1,wTau2,None,1)
    
    # suggest Tau2
    # tau2 = 3.97
    # A2 = trial.suggest_float("A2", 0, 1000)
    # B2 = trial.suggest_float("B2", 0, 1)
    # saveTau(tau2,A2,B2,2)
    
    # suggest Tau3
    # tau3 = 41.62
    # A3 = trial.suggest_float("A3", 0, 1000)
    # B3 = trial.suggest_float("B3", 0, 1)
    # saveTau(tau3,A3,B3,3)

    
    h.xopen("./Exp08-NaSpike-ExpSyn.hoc")
    
    

    score = relLikelihood() #Score based on shape also
    return score


def loadStudy():
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_name,
    )
    return study


def printStudy(study):
    df = study.trials_dataframe(attrs=("number", "value", "params"))
    pd.set_option('display.max_rows', len(df))
    print(df)

def showStudy(study):
    fig = optuna.visualization.plot_optimization_history(study)
    plotly.offline.plot(fig, filename='studyVisualize.html')
    
def initStudy(storage_name, study_name):
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
    )
    return study

def setBest(study):
    parmDict = study.best_params
    saveSynWeight(parmDict['SynWeight'])

    saveDelta(parmDict['DELTA'])

    # saveTau(parmDict['tau1'],
    #     parmDict['wTau2'],
    #     None,
    #     1
    # )

    saveTau(parmDict['tau2'],
            parmDict['A2'],
            parmDict['B2'],
            2
            )
    saveTau(parmDict['tau3'],
            parmDict['A3'],
            parmDict['B3'],
            3
            )


if __name__ == "__main__":
    args = setArgParser().parse_args()
    if ".db" in args.study_name:
        study_name = args.study_name.replace(".db", "")
    else:
        study_name = args.study_name
    storagePath = os.path.join(os.path.abspath(args.saveDir), study_name)
    storage_name = f"sqlite:///{storagePath}.db"
    study = initStudy(storage_name, study_name)

    if args.saveBest:
        if os.path.isfile(f'{storagePath}.db'):
            print(f'Found database{storage_name}')
            printStudy(study)
            setBest(study)
    elif args.printStudy:
        printStudy(study)
        showStudy(study)
    else:
        print(f"Saving to databse{storage_name}")
        study.optimize(
            objective,
            n_trials=1,
        )
        printStudy(study)
