TITLE Potassium ion accumulation
: Intracellular potassium ion accumulation
COMMENT
To Do
Check with Halnes chapter 9 parameters if they are sufficient,
Yamada Methods in Neuronal Modeling may be more suitable.

Look Up Armstrong units to double check.


ENDCOMMENT

NEURON {
	SUFFIX k_acc
	USEION k READ ko, ik WRITE ko
        RANGE tauk, ko0, flag, kbath, flux
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
    fhspace = 100 (angstrom) : effective thickness 
    tauk = 3 (ms) : Halnes chapter 9 the NEURON book  of Halnes
    flag  = 0 (1)
}

ASSIGNED {
    ik 	(mA/cm2)
    kbath (mM/ms)
    flux (mM/ms)
    dt (ms)
}

STATE {
    ko (mM)
}

INITIAL {
    ko = ko0
    kbathRate()
    
}

BREAKPOINT {
        kbathRate()
        SOLVE state METHOD derivimplicit
    }
    
    DERIVATIVE state {
        : if (ko0 > 2.5){
        :     printf("%g\n",ko0)
    : }
    ko' = flux + kbath
    : printf("%g, %g, %g, %g\n",flag,ik,kbath,(1e8)*ik /(fhspace*F)/kbath)
}
PROCEDURE kbathRate(){
    UNITSOFF
    if (flag > 0){
        flux = 0
        kbath = 0
        ko = ko0
        : printf("%g\n",kbath)
    } else {
        flux = (1e8)*ik /(fhspace*F) 
        kbath =  (ko0 - ko)/tauk
    }
    
    if (flag == 1) {
        : instantaneous free bath mode for one step
        flag = 0   
    }
    UNITSON
}
    
