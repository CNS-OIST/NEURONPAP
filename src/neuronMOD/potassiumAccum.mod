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
	PI = (pi) (1)
        
}

PARAMETER {
    
    ik 	(mA/cm2)
    ko0 = 2.5 (mM)
    tauk = 50 (ms) : Halnes chapter 9 the NEURON book
    Dk = 1.96e-9 (m2/s)

    
}

ASSIGNED {
        area (cm2)
	diam (um)
}

STATE {
        : ki (mM)
    ko (mM)
    : ka (mM)
}

:  INITIAL {
: 	: VERBATIM

: 	: ki = _ion_ki;
	
: 	: ENDVERBATIM
:         : figure a way to set ko to global default ko
:         : ko = ko0
:         ka = ki
: }

BREAKPOINT {
	SOLVE state METHOD derivimplicit
   	: SOLVE conc METHOD sparse
    }
    
    DERIVATIVE state {
        ko' = ik / (F * area) - (ko - ko0)/tauk
        : ki' = -ik / (F * area)
    }
 

: KINETIC conc {
: 	COMPARTMENT PI*diam*diam/4 {ka}
: 	LONGITUDINAL_DIFFUSION Dk*PI*diam*diam/4 {ka}
: 	: LONGITUDINAL_DIFFUSION Dk {ka}
:         ~ ka << (-(ik)/F*PI*diam)
:         ki = ka
: }