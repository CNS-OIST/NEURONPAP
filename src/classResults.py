import copy

class ResultsPAPModel():
    # class to copy data of simulations
    # necessary for MPI copy as NEURON components cannot be copied
    to_be_copied = [
        "vPAP",
        "vSoma",
        "iNMDA",
        "iMem",
        "iKSoma",
        "time",
        "bLen",
        "somaSize",
        "PAPWid",
        "bWid",
        "initialKo",
        "KoPAP",
        "KoSoma",
        "KiPAP",
        "KiSoma",
    ]

    def copyAttr(self):
        newInstance = ResultsPAPModel()
        newInstance.__dict__ = {
            attr: copy.deepcopy(self.__dict__[attr])
            for attr in self.__dict__
            if attr in self.to_be_copied
        }
        return newInstance

    def getRMP(self):
        RMP = list(self.vSoma)[-1]
        return RMP
