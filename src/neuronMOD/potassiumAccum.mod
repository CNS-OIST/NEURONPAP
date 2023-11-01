TITLE Potassium ion accumulation
: Intracellular potassium ion accumulation 

NEURON {
	SUFFIX k_acc
	USEION k READ ko, ik WRITE ko
        RANGE tauk, flag,kout,ko0
	THREADSAFE
}

UNITS {
	(um) = (micron)	
	(mV) = (millivolt)
	(mM) = (milli/liter)
	(mA) = (milliamp)
	F = (faraday) (coulombs)
	PI = (pi) (1)
        
}

PARAMETER {    
    ik 	(mA/cm2)
    ko0 = 2.5 (mM)
    tauk = 50 (ms) : Halnes chapter 9 the NEURON book
}

ASSIGNED {
    area (cm2)
    kout (mM)
    flag (1) : switch between step and pulse
}

STATE {
    ko (mM)
}

INITIAL {
	: VERBATIM

	: ki = _ion_ki;
	
	: ENDVERBATIM
        : figure a way to set ko to global default ko
        ko = ko0
        flag = 0
        kClear(ko)        
    }
    
    BREAKPOINT {
        SOLVE state METHOD derivimplicit
        : if (ko > 2.6) {
        :     printf("ko: %g\n",ko)
    : }
}

DERIVATIVE state {
    kClear(ko)
    ko' =  kout
        : printf("ko0: %g\n",ko0)
    }
    
    PROCEDURE kClear(ko (mM)) {
        if (flag == 1){
            kout = ko0 - ko
        } else {
            kout = ik / (F * area) - (ko-ko0)/tauk            
        }
        : printf("kout: %g\n",kout)
    }
