TITLE Sodium ion accumulation

NEURON {
	SUFFIX nai_acc
	USEION na READ nai,ina WRITE nai
        RANGE tauna, nao0, flag
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
    nao0 = 140 (mM)
    fhspace = 100 (angstrom) : effective thickness 
    tauna = 3 (ms) : Halnes chapter 9 the NEURON book  of Halnes
    flag  = 0 (1)
    nai0 = 15 (mM)
}

ASSIGNED {
    ina 	(mA/cm2)
    dt (ms)
    d (um)
}

STATE {
    nao (mM)
    nai (mM)
}

INITIAL {
    nao = nao0

    }
    
    BREAKPOINT {
        SOLVE state METHOD derivimplicit
    }
    
    DERIVATIVE state {
        : if (ko0 > 2.5){
        :     printf("%g\n",ko0)
    : }
    nai' = -(ina)/(d/2*F) * (1e4) + (nai0 - nai)/tauna
}

    
