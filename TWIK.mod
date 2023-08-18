TITLE potassium outward leak channel TWIK-1

COMMENT
This is kinetics of Kir4.1 channels used in the Janic et al. 2022 paper.
MAYBE needs electrode current statement


ENDCOMMENT
NEURON {
    SUFFIX twik
    USEION k READ ki,ko WRITE ik
    RANGE  powk,vs,taukp,PBkp,kob,vzerokp,Skp
}

UNITS {
	(molar) = (1/liter)
	(nA) = (nanoamp)
        (mV) = (millivolt)
	(mM) =	(millimolar)
	(J)  = (joules)
        
    }
    
CONSTANT {
	T = 273.16	(degC)
	F = 9.6485e4	(coul)	: Faraday's constant (coulombs/mol)
	R = 8.314	(J/degC): universal gas constant (joules/mol/K)
	z = 1		()		: valency of K+
}
    

PARAMETER {
    powk = 2 
    vs = 25.7 (mV)
    : ki = 130     (mM)
    taukp = 3.0 (ms)
    PBkp = 1.24e-08 (cm3/s)
    kob = 2.5 (mM) : 5.0 (mM)
    vzerokp = -20.5 (mV)
    Skp = 1.7 
}

ASSIGNED {
    v 		(mV)
    ik      (nA)
    
    : Pkp (cm3/s)
    : vkp (mV)
    nkp  (1)
    
    ko      (mM)
    ki      (mM)    
}


STATE {
    n
}

INITIAL {
    rates(v,ko,ki)
    n = 0
}


BREAKPOINT {
    SOLVE state METHOD derivimplicit
    
    ik = pow(n,powk) * Pkp(ko) * pow(F,2) * pow(z,2) * v * (ki - ko*exp(-z*v/vs)) / (R * T * (1 - exp(-z*v/vs))) / 1000
    printf("Pkp:%g v:%g nkp:%g ko:%g vkp:%g n:%g\n", Pkp(ko), v, nkp,ko,vkp(ko),n)
    printf("exp:%g\n",exp(-z*v/vs))
    : ionMove()
        : ik = (0.001)*gkir * ( v - ek*NormK - va1) *sqrt(((ko)/(1 (mM)))/(1+exp((v-ek*NormK-va2)/va3)))		: calculate ik 
	: printf("v: %g, ko: %g, va2: %g\n", v, ko, va2)
        : consider different channel dynamics
    }
:    PROCEDURE ionMove() {
:     ko = ko + ik / (F * area)
:     : ki = ki - ik / (F * area)

: }
DERIVATIVE state {
    rates(v,ko,ki)
    n' = (nkp - n) / taukp
    }

PROCEDURE rates(v (mv),ko (mM),ki (mM)) {
    : Pkp = PBkp * (1 + 0.85 * log10(ko/kob))
    : vkp = vzerokp - Skp * vs * log(ko/kob)
    nkp = (1 - ko/ki)/ (1 + exp(-z * (v - vkp(ko))/vs))
}

FUNCTION Pkp(ko (mM)) (cm3/s){
    Pkp = PBkp * (1 + 0.85 * log10(ko/kob))
}

FUNCTION vkp(ko (mM)) (mV) {
    vkp = vzerokp - Skp * vs * log(ko/kob)    
}
