"""
Library of conveninet utils such as parallizeFor
"""

import math
import os
import pickle
from mpi4py import MPI
import tqdm
import numpy as np
from textSDIO import *
import copy
import random
import sys
from contextlib import contextmanager
import tracemalloc


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


@contextmanager
def global_function_override_runtime(
    function_name, new_function, module_name="__main__"
):
    """Temporarily override a function in the __main__ namespace."""
    main_module = sys.modules[module_name]
    original_function = getattr(main_module, function_name, None)

    # Rebind the name in the __main__ module to the new function upon entry
    setattr(main_module, function_name, new_function)

    try:
        yield
    finally:
        # Crucial: Rebind the name back to the original function upon exit
        setattr(main_module, function_name, original_function)


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
            # tracemalloc.start()
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
            # snapshot = tracemalloc.take_snapshot()
            # top_stats = snapshot.statistics("lineno")
            # print("[ Top 10 ]")
            # for stat in top_stats[:10]:
            #    print(stat)

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


def sizeof(obj):
    seen = set()

    def inner(o):
        if id(o) in seen:
            return 0
        seen.add(id(o))
        size = sys.getsizeof(o)
        if isinstance(o, dict):
            size += sum(inner(k) + inner(v) for k, v in o.items())
        elif isinstance(o, (list, tuple, set)):
            size += sum(inner(i) for i in o)
        return size

    return inner(obj)  # ← fix


def print_pickle_objs():
    if len(sys.argv) < 2:
        print("Usage: python script.py <pickle_file>")
        sys.exit(1)

    filename = sys.argv[1]

    with open(filename, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        sorted_items = sorted(data.items(), key=lambda x: sizeof(x[1]), reverse=True)
        for k, v in sorted_items:
            print(f"{k}: {sizeof(v)} bytes")

    elif isinstance(data, (list, tuple)):
        sorted_items = sorted(enumerate(data), key=lambda x: sizeof(x[1]), reverse=True)
        for idx, item in sorted_items:
            print(f"index {idx}: {sizeof(item)} bytes")

    else:
        # fallback: try to print attribute names if object has __dict__
        if hasattr(data, "__dict__"):
            items = vars(data).items()
            sorted_items = sorted(items, key=lambda x: sizeof(x[1]), reverse=True)
            for name, val in sorted_items:
                print(f"{name}: {sizeof(val)} bytes")
        else:
            print(f"value: {sizeof(data)} bytes")
        if len(sys.argv) < 2:
            print("Usage: python script.py <pickle_file>")
            sys.exit(1)

        filename = sys.argv[1]

        with open(filename, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            sorted_items = sorted(
                data.items(), key=lambda x: sizeof(x[1]), reverse=True
            )
            for k, v in sorted_items:
                print(k, sizeof(v))

        elif isinstance(data, (list, tuple)):
            sorted_items = sorted(data, key=lambda x: sizeof(x), reverse=True)
            for item in sorted_items:
                print(sizeof(item))

        else:
            raise TypeError("Unsupported pickle structure")


class LazySharedObject:
    def __init__(self, shared_array, win):
        self._shared_array = shared_array
        self._cache = {}
        self._win = win

    def _load(self):
        d = object.__getattribute__(self, "__dict__")

        # if cached bytes exist → use them
        if "_local_bytes" in d:
            return pickle.loads(d["_local_bytes"])

        # otherwise read from shared memory
        shared_array = object.__getattribute__(self, "_shared_array")

        data_bytes = bytes(memoryview(shared_array))

        # cache locally
        d["_local_bytes"] = data_bytes

        return pickle.loads(data_bytes)

    def _write_back(self, obj):
        data_bytes = pickle.dumps(obj)
        new_size = len(data_bytes)

        if new_size > self._shared_array.nbytes:
            self._resize(new_size)

        self._shared_array[:new_size] = np.frombuffer(data_bytes, dtype=np.uint8)

        self._shared_array[new_size:] = 0

    def _resize(self, required_size):
        if rank != 0:
            return
        new_nbytes = int(required_size * 1.5)

        self._win.Free()

        itemsize = 1
        self._win = MPI.Win.Allocate_shared(new_nbytes, itemsize, comm=comm)

        buf, _ = self._win.Shared_query(0)
        self._shared_array = np.ndarray(buffer=buf, dtype=np.uint8, shape=(new_nbytes,))

    def __iter__(self):
        obj = pickle.loads(memoryview(self._shared_array))
        try:
            for item in obj:
                yield item
        finally:
            # automatically runs when loop finishes or breaks
            del obj

    def __getitem__(self, key):
        obj = pickle.loads(memoryview(self._shared_array))
        try:
            return obj[key]
        finally:
            del obj

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        d = object.__getattribute__(self, "__dict__")
        cache = d.setdefault("_cache", {})

        if name in cache:
            return cache[name]

        # Get _load WITHOUT triggering __getattr__
        load = object.__getattribute__(self, "_load")
        obj = load()

        try:
            if hasattr(obj, name):
                value = getattr(obj, name)
            elif isinstance(obj, dict) and name in obj:
                value = obj[name]
            else:
                raise AttributeError(name)
        finally:
            del obj

        cache[name] = value
        return value

    def __iadd__(self, other):
        obj = self._load()

        try:
            obj += other  # works for list
            self._write_back(obj)
        finally:
            del obj
        return self

    def append(self, value):
        obj = self._load()
        try:
            obj.append(value)
            self._write_back(obj)
        finally:
            del obj

    def extend(self, values):
        obj = self._load()
        try:
            obj.extend(values)
            self._write_back(obj)
        finally:
            del obj

    def __len__(self):
        obj = self._load()
        try:
            return len(obj)
        finally:
            del obj

    def dump(self, file):
        obj = self._load()
        try:
            pickle.dump(obj, file)
        finally:
            del obj

    def dump_to_file(self, filename):
        with open(filename, "wb") as f:
            self.dump(f)

    def load(self):
        return self._load()


def load_interm_data(pickle_obj, root=0):
    if isinstance(pickle_obj, LazySharedObject):
        wMessage("Expcted raw pickled object no lazy")
        return
    if rank == root:
        data_bytes = pickle.dumps(pickle_obj, protocol=pickle.HIGHEST_PROTOCOL)
        del pickle_obj
        nbytes = len(data_bytes)
    else:
        nbytes = 0
    nbytes = comm.bcast(nbytes, root=root)
    win = MPI.Win.Allocate_shared(nbytes if rank == 0 else 0, 1, comm=comm)
    buf, _ = win.Shared_query(0)
    shared_buf = np.ndarray(buffer=buf, dtype=np.uint8, shape=(nbytes,))
    if rank == root:
        shared_buf[:] = np.frombuffer(data_bytes, dtype=np.uint8)
    comm.Barrier()

    return LazySharedObject(shared_buf, win), win


def release_pickle(shared_array, win):
    import gc

    if comm is not None:
        comm.Barrier()

    if shared_array is not None:
        try:
            del shared_array._shared_array
        except AttributeError:
            pass
        del shared_array

    if comm is not None:
        comm.Barrier()

    if win is not None:
        win.Free()

    # Force garbage collection
    gc.collect()


def cylindrical_shell_volume(diameter, shell_depth, length):
    R = diameter / 2
    r = R + shell_depth
    return np.pi * length * (r**2 - R**2)


def column_in_array(col, arr):
    if arr.ndim == 1:
        return False
    col = np.asarray(col).reshape(-1)
    for k in range(arr.shape[1]):
        if np.array_equal(col, arr[:, k]):
            return True
    return False


if __name__ == "__main__":
    print_pickle_objs()
