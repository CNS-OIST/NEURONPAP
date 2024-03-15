import math
from mpi4py import MPI
import tqdm
import numpy as np
from textSDIO import *


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


def get_iter(parmA, parmASteps, parmB, parmBSteps):
    # Update to get dynamic loops
    iterations = []
    for i in range(0, parmA + 1, parmASteps):
        for j in range(0, parmB + 1, parmBSteps):
            iterations.append((i, j))

    return iterations


def parallizeFor(
    iterations, functions, functionArgs, functionParms, callmethods, methodArgs
):
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

    for index in range(minimum, maximum):
        parmSet = iterations[index]
        # print(f'Thread {rank} is performing set {parmSet}')
        results.append([])
        for k, func in enumerate(functions):
            for l, parameterName in enumerate(functionParms):
                functionArgs[k][parameterName] = parmSet[l]
            tmpInstance = functions[k](**functionArgs[k])

            for l, method in enumerate(callmethods[k]):
                getattr(tmpInstance, method)(**methodArgs[k][l])

            results[-1].append(
                tmpInstance.copyAttr()
            )  # default workaround for mpi section pickle bug

        # update tqdm
        pbar.update(1)

    comm.Barrier()
    results = comm.gather(results, root=0)
    if rank == 0:
        tmpResults = []
        for res in results:
            tmpResults += res
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
