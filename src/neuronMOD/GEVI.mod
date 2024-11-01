TITLE GEVI fluor
COMMENT

ENDCOMMENT

NEURON {
	SUFFIX GEVI
        RANGE tON,tOFF
	THREADSAFE
    }
    
    UNITS {
        (mV) = (millivolt)
    }
    ASSIGNED {
        v (mV)
        fluor (mV/ms)
    }
    PARAMETER {
        tON = 58 (ms)
        tOFF = 44 (ms)
    }
    INITIAL {
        dF = v
    }
    STATE {
        dF (mV)        
    }
    BREAKPOINT {
        calcFluor()
        SOLVE state METHOD derivimplicit
    }
    DERIVATIVE state {
        : printf("%f\n",fluor)
        dF' = fluor
    }
    PROCEDURE calcFluor (){
        if ((v - dF) > 0 ){
            fluor = (v - dF) / tON
        } else {
            fluor = (v - dF) / tOFF
        }
        
    }

