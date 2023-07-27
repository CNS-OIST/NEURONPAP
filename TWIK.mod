TITLE potassium outward leak channel TWIK-1

COMMENT
This is kinetics of Kir4.1 channels used in the Janic et al. 2022 paper.


ENDCOMMENT
NEURON {
    SUFFIX twik
    USEION k READ ko WRITE ik	
    RANGE  ik, gkir, NormK, Pkir, v, dz, vs, delta, DeltazB, vkir, ki
        
}

UNITS {
	(molar) = (1/liter)
	(nA) = (nanoamp)
        (mV) = (millivolt)
        (mS)  = (millisiemens)
	(mM) =	(millimolar)
	(J)  = (joules)
        
    }
    
CONSTANT {
	T = 273.16	(degC)
	F = 9.648e4	(coul)	: Faraday's constant (coulombs/mol)
	R = 8.315	(J/degC): universal gas constant (joules/mol/K)
	z = 1		(1)		: valency of K+
}
    

PARAMETER {
    v 		(mV)
    k = 2 
    vs = 25.7 (mV)
    ki = 130     (mM)
    taukp = 3.0 (ms)
    PBkp = 1.24e-08 (cm3/s)
    kob = 2.5 (mM) : 5.0 (mM)
    vzerokp = -20.5 (mV)
    Skp = 1.7
}

ASSIGNED {
    ik      (nA)
    
    ko      (mM)
}

INITIAL {
    vkp(ko)
    nkp(v)
    Pkp(ko)
    
    n = 0
    }


BREAKPOINT {
    SOLVE n METHOD runge
    ik = pow(n,k) * Pkp(ko) * pow(F,2) * pow(z,2) * v * (ki - ko*exp(-z*v/vs)) / (R * T * (1 - exp(-z*v/vs)))
        : ik = (0.001)*gkir * ( v - ek*NormK - va1) *sqrt(((ko)/(1 (mM)))/(1+exp((v-ek*NormK-va2)/va3)))		: calculate ik 
	: printf("v: %g, ko: %g, va2: %g\n", v, ko, va2)
        : consider different channel dynamics
    }
    
    DERIVATIVE n {
        n' = (nkp(v) - n) / taukp
    }
    FUNCTION Pkp(ko(mM)){
        Pkp = PBkp * (1 + 0.85 * log(ko/kob,10))
    }
    FUNCTION nkp(v(mV)) {
        nkp = (1 - ko/ki)/ (1 + exp(-z * F *(v - vkp(ko))/(R * T)))
    }
    FUNCTION vkp(ko (mM)){
        vkp = vzerokp - Skp * vs * log(ko/kob)
    }
    

