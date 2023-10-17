class ResultsPAPModel():
    # vPAP = list()
    # vSoma = list()
    # iNMDA = list()
    # iMem = list()
    # iKSoma = list()
    # time = list()
    to_be_copied = [
        'vPAP',
        'vSoma',
        'iNMDA',
        'iMem',
        'iKSoma',
        'time',
        'bLen',
        'somaSize',
        'papWid',
        'bWid',
        'initialKo',
        'KoPAP',
        'KoSoma'
    ]

    def copyAttr(self):
        newInstance = ResultsPAPModel()
        newInstance.__dict__ = {attr:copy.deepcopy(self.__dict__[attr])  for attr in self.__dict__ if attr in self.to_be_copied}
        return newInstance
