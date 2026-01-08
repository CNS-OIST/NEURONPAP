TITLE Potassium ion accumulation

NEURON {
	SUFFIX k_acc
	USEION k READ ko, ik WRITE ko
        RANGE tauk_0, ko0, flag, kbath, flux, slowing
        RANGE fhspace
	THREADSAFE
}

UNITS {
	(um) = (micron)	
	(mV) = (millivolt)
	(mM) = (milli/liter)
	(mA) = (milliamp)
        (nA) = (nanoamp)
	F = (faraday) (coulombs)
}

PARAMETER {    
    ko0 = 2.5 (mM)
    fhspace = 400 (angstrom) : effective thickness 
    tauk_0 = 4.0 (ms) :Ransom C.B. (2000) Journal of Physiology
    flag  = 0 (1)
    slowing = 1 (1)
    startFlag = 0 (1)
    tstart = 0 (1)
}

ASSIGNED {
    tauk (ms)
    ik 	(mA/cm2)
    kbath (mM/ms)
    flux (mM/ms)
    dt (ms)
    iNMDA (mA/cm2)
    iGluT (mA/cm2)
}

STATE {
    ko (mM)
}

INITIAL {
    ko = ko0
    tauk = tauk_0
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
        tauk = slowing * tauk_0: slowing 
        kbath =  (ko0 - ko)/tauk
    }
    
    if (flag == 1) {
        : instantaneous free bath mode for one step
        flag = 0   
    }
    UNITSON
}
    
