TITLE Potassium ion accumulation
: Intracellular potassium ion accumulation 

NEURON {
	SUFFIX K_acc
	USEION k READ ko, ik WRITE ko
        RANGE ko0
}

UNITS {
	
	(mV) = (millivolt)
	(mM) = (milli/liter)
	(mA) = (milliamp)
	F = (faraday) (coulombs)
}

PARAMETER {
    
    ik 	(mA/cm2)
    ko0 = 2.5 (mM)
    tauk = 100 (ms)
    
}

ASSIGNED {
        area (cm2)
}

STATE {
	ko (mM)
	
}

 INITIAL {
	VERBATIM

	ko = _ion_ko;
	
	ENDVERBATIM
        : figure a way to set ko to global default ko
        : ko = ko0
}

BREAKPOINT {
	SOLVE state METHOD derivimplicit
    }
    
    DERIVATIVE state {
        ko' = ik / (F * area) - (ko - ko0)/tauk
}