TITLE Potassium ion accumulation
: Intracellular potassium ion accumulation 

NEURON {
	SUFFIX K_acc
	USEION k READ ko, ik WRITE ko
        RANGE ko0
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
    diam (um)
}

STATE {
        : ki (mM)
    ko (mM)
}

 INITIAL {
	: VERBATIM

	: ki = _ion_ki;
	
	: ENDVERBATIM
        : figure a way to set ko to global default ko
        : ko = ko0
        ko = ko0
}

BREAKPOINT {
    SOLVE state METHOD derivimplicit
    }
    
    DERIVATIVE state {
        ko' = ik / (F * area) - (ko - ko0)/tauk
    }
 

