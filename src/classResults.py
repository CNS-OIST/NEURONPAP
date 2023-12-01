import copy

class ResultsPAPModel():
    # class to copy data of simulations
    # necessary for MPI copy as NEURON components cannot be copied

    dont_copy = [
        "PAP",
        "soma",
        "branches",
        "branch",
        "NMDAs",
        "NCs"
        ]

    def copyAttr(self):
        newInstance = ResultsPAPModel()
        newInstance.__dict__ = {
            attr: copy.deepcopy(self.__dict__[attr])
            # print(attr)
            for attr in self.__dict__
            if attr not in self.dont_copy
        }
        return newInstance

    def getRMP(self):
        RMP = list(self.vSoma)[-1]
        return RMP
