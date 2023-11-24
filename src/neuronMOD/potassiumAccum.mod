TITLE Potassium ion accumulation
: Intracellular potassium ion accumulation 

NEURON {
	SUFFIX k_acc
	USEION k READ ko, ik WRITE ko
        RANGE tauk, ko0
	THREADSAFE
}

UNITS {
	(um) = (micron)	
	(mV) = (millivolt)
	(mM) = (milli/liter)
	(mA) = (milliamp)
	F = (faraday) (coulombs)
}

PARAMETER {    
    ko0 = 2.5 (mM)
    fhspace = 300 (angstrom) : effective thickness
    tauk = 50 (ms) : Halnes chapter 9 the NEURON book    
}

ASSIGNED {
    ik 	(mA/cm2)
}

STATE {
    ko (mM)
}

INITIAL {
        ko = ko0
    }
    
    BREAKPOINT {
        SOLVE state METHOD derivimplicit
}

DERIVATIVE state {
    ko' =  (1e8)*ik /(fhspace*F) + (ko0-ko)/tauk
    }
    
