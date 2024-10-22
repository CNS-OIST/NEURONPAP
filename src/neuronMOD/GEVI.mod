TITLE input Resistance Range Var
COMMENT

ENDCOMMENT

NEURON {
	SUFFIX GEVI
        RANGE kON,kOFF
	THREADSAFE
    }
    
    UNITS {
        (mV) = (millivolt)
    }
    ASSIGNED {
        v (mV)
        v_initial (mV)
        fluor (mV/ms)
    }
    PARAMETER {
        tON = 55 (ms)
        tOFF = 48 (ms)
    }
    INITIAL {
        dF = 0
        v_initial = v
    }
    STATE {
        dF (mV)        
    }
    BREAKPOINT {
        SOLVE state METHOD derivimplicit
    }
    DERIVATIVE state {
        if ((v - v_initial) - dF > 0 ){
            fluor = ((v - v_initial) - dF) / tON
        } else {
            fluor = (dF - (v - v_initial)) / tOFF
        }
        dF' = fluor
    }

