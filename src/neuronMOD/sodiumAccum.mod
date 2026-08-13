TITLE Sodium ion accumulation
: Intracellular potassium ion accumulation
COMMENT
To Do
Check with Halnes chapter 9 parameters if they are sufficient,
Yamada Methods in Neuronal Modeling may be more suitable.

Look Up Armstrong units to double check.


ENDCOMMENT

NEURON {
    SUFFIX na_acc
    USEION na READ nao, ina WRITE nao
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
    fhspace = 200 (angstrom) : effective thickness 
    tauna = 3 (ms) : Halnes chapter 9 the NEURON book  of Halnes
    flag  = 0 (1)
}

ASSIGNED {
    ina 	(mA/cm2)
    dt (ms)
}

STATE {
    nao (mM)
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
    nao' = (1e8)*ina /(fhspace*F) - (nao - nao0)/tauna
}


