"""
Library of conveninet utils such as parallizeFor
"""

import math
from mpi4py import MPI
import tqdm
import numpy as np
from textSDIO import *
import copy
import random


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def MPIReadlines(fName):
    # if simultaneous access creates errors update this function
    # fails in parallel must think of a better way
    f = open(fName, "r")
    lines = f.readlines()
    f.close()

    lines = [float(line.strip()) for line in lines]
    return lines


def eq(x, a, b):
    return a * x + b


def loadFile(fName):
    if os.path.isfile(fName):
        return pd.read_csv(fName, header=None)
    else:
        return None


def plot(dir, zoom=False, ext=False):
    i = loadFile(os.path.join(dir, "iFile.dat"))
    v = loadFile(os.path.join(dir, "vFile.dat"))
    mem = loadFile(os.path.join(dir, "iFileMem.dat"))
    t = loadFile(os.path.join(dir, "tFile.dat"))
    plt.plot(t, i)
    plt.ylabel("pA")
    if max(i.iloc[:, 0]) > 1:
        plt.ylim(0, 25)
    plt.xlabel("ms")
    if zoom:
        plt.xlim(9.99, 10.025)
    elif ext and max(t) >= 10000:
        plt.xlim(11, 10000)
    plt.savefig(os.path.join(dir, "results.pdf"))
    plt.cla()
    plt.clf()
    if v is not None:
        plt.plot(t, v)
        plt.ylabel("mV")
        plt.xlabel("ms")
        print(max(v.iloc[:, 0]))
        plt.ylim(-90, 0)
        # if zoom or max(v.iloc[:, 0]) > -20:
        #     plt.xlim(9.99,10.1)
        # elif ext and max(t) >= 10000:
        #     plt.xlim(11,10000)

        plt.savefig(os.path.join(dir, "resultsV.pdf"))
        plt.cla()
        plt.clf()
    if mem is not None:
        plt.plot(t, mem)
        plt.ylabel("pA")
        plt.xlabel("ms")
        # plt.ylim(-90,0)
        if zoom:
            plt.xlim(9.99, 10.025)
        elif ext and max(t) >= 10000:
            plt.xlim(11, 10000)

        plt.savefig(os.path.join(dir, "resultsIMem.pdf"))
    return max(v.iloc[:, 0])


def get_iter(parmA, parmASteps, parmB, parmBSteps, starta=0, startb=0):
    # Update to get dynamic loops
    iterations = []
    if type(parmA) is int and type(parmASteps) is int:
        parmA_iters = range(starta, parmA + 1, parmASteps)
    else:
        parmA_iters = np.arange(starta, parmA + parmASteps / 2, parmASteps)
    for i in parmA_iters:
        if type(parmB) is int and type(parmBSteps) is int:
            for j in range(startb, parmB + 1, parmBSteps):
                iterations.append((i, j))
        else:
            for j in np.arange(startb, parmB + parmBSteps / 2, parmBSteps):
                iterations.append((i, j))

    return iterations


def parallizeFor(
    iterations,
    functions,
    functionArgs,
    functionParms,
    callmethods,
    methodArgs,
    mode="InitArgs",
    randomize=True,
):
    # ranodmize in case of consectuive iteration pairs that take time
    if randomize:
        iterations = comm.bcast(random.sample(iterations, len(iterations)), root=0)

    # Calculate the number of iterations each process will handle
    iterations_per_process = len(iterations) // size

    # Adjust the range for the last process
    if len(iterations) % size == 0:
        remaining_iterations = 0
        minimum = rank * iterations_per_process
        maximum = (rank + 1) * iterations_per_process
    elif rank >= size - len(iterations) % size:
        remaining_iterations = 1
        minimum = rank * (iterations_per_process + remaining_iterations) - (
            size - len(iterations) % size
        )
        maximum = (rank + 1) * (iterations_per_process + remaining_iterations) - (
            size - len(iterations) % size
        )
    else:
        remaining_iterations = 0
        minimum = rank * iterations_per_process
        maximum = (rank + 1) * iterations_per_process

    # # results list for each rank
    # results = [ [] for i in range(len(iterations)) ]
    results = []

    comm.Barrier()

    # Set up tqdm
    pbar = tqdm.tqdm(
        total=iterations_per_process + remaining_iterations,
        desc=f"rank{rank:02}",
        position=rank,
    )

    comm.Barrier()
    if mode == "InitArgs":
        # print(f"Thread {rank} will perform sets {iterations[minimum:maximum]}")
        for index in range(minimum, maximum):
            parmSet = iterations[index]
            # print(f"Thread {rank} is performing set {parmSet}")
            results.append([])
            for k, func in enumerate(functions):
                if len(functionParms) > 1:
                    for l, parameterName in enumerate(functionParms):
                        functionArgs[k][parameterName] = parmSet[l]
                else:
                    # print(parmSet)
                    functionArgs[k][functionParms[0]] = parmSet
                tmpInstance = functions[k](**functionArgs[k])

                for l, method in enumerate(callmethods[k]):
                    getattr(tmpInstance, method)(**methodArgs[k][l])

                results[-1].append(
                    tmpInstance.copyAttr()
                )  # default workaround for mpi section pickle bug

            # update tqdm
            pbar.update(1)
    elif mode == "MethodArgs":
        for index in range(minimum, maximum):
            # print(index)
            parmSet = iterations[index]
            methodArgsTmp = copy.deepcopy(methodArgs)
            # print(methodArgs)
            # print(f'Thread {rank} is performing set {parmSet}')
            sys.stdout.flush()
            results.append([])
            for k, func in enumerate(functions):
                tmpInstance = functions[k](**functionArgs[k])

                for l, method in enumerate(callmethods[k]):
                    if method in functionParms:
                        keyList = [
                            key
                            for key, v in methodArgs[k][l].items()
                            if type(v) == str and "parallelItem" in v
                        ]
                        # accepts only two variations in method args
                        if len(keyList) == 2:
                            if (
                                methodArgsTmp[k][l][keyList[0]]
                                < methodArgsTmp[k][l][keyList[1]]
                            ):
                                methodArgsTmp[k][l][keyList[0]] = parmSet[0]
                                methodArgsTmp[k][l][keyList[1]] = parmSet[1]
                            else:
                                methodArgsTmp[k][l][keyList[1]] = parmSet[0]
                                methodArgsTmp[k][l][keyList[0]] = parmSet[1]
                        # else:
                        #     print(f'Not two {keyList=}')
                    # print(methodArgsTmp[k][l])
                    # print(f'{parmSet=}')
                    getattr(tmpInstance, method)(**methodArgsTmp[k][l])

                results[-1].append(
                    tmpInstance.copyAttr()
                )  # default workaround for mpi section pickle bug
                # print(tmpInstance.SpikeNum,tmpInstance.SpikeFreq)

            # update tqdm
            pbar.update(1)

    comm.Barrier()
    results = comm.gather(results, root=0)
    if rank == 0:
        tmpResults = []
        for res in results:
            tmpResults += res
            # print(res[0][0].SpikeNum,res[0][0].SpikeFreq)
        return tmpResults


def find_nan_inf_index(lst):
    for i, value in enumerate(lst):
        if math.isnan(value) or math.isinf(value):
            return i
    return "stop"  # Return -1 if no NaN or inf value is found


def remove_nan_values(lst, lst2):
    index = find_nan_inf_index(lst)
    while index != "stop":
        del lst[index]
        lst2 = np.delete(lst2, index)
        index = find_nan_inf_index(lst)
    if len(lst) == len(lst2):
        return lst, lst2
    else:
        eMessage("wrong list length from removing")
