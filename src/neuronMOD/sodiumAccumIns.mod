TITLE Sodium ion accumulation

NEURON {
    SUFFIX nai_acc
    USEION na READ nai,ina WRITE nai
    RANGE nai0
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
    nai0 = 15 (mM)
}

ASSIGNED {
    ina 	(mA/cm2)
    dt (ms)
    diam (um)
}

STATE {
    nai (mM)
}

INITIAL {

    nai = nai0
}

BREAKPOINT {
    SOLVE state METHOD sparse
}

KINETIC state {
    COMPARTMENT PI*diam*diam/4 {nai}
    ~ nai << (-1*(ina)*PI*diam/F * (1e4) )
}


