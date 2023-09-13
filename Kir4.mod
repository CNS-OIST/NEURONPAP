TITLE inward rectifier potassium (Kir4) channel

COMMENT
This is kinetics of Kir4.1 channels used in the Janic et al. 2022 paper.
MAYBE needs electrode current statement


ENDCOMMENT
NEURON {
    SUFFIX kir4 			
    USEION k READ ki,ko WRITE ik
    RANGE Pkir, dz, vs, delta, DeltazB, vkir, tauKir, nkir
        
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
    Pkir = 9.485e-08 (cm3/s)
    dz = 1.0 : >1.0
    vs = 25.7 (mV)
    delta = 0.5 : 0.4 : 0.5
    DeltazB = 0.45 : -3.1 : 0.45
    vkir = -55 (mV) : -45 (mV)
    :    tauKir = 600 (ms) :Sibille J. 2015
    :nkir = 1 : guessed for now
    kl = 10.89       (mV)    		: Stegen et al. 2012
    vhalft=67.0828	 (mV)    		: fitted #100 \muM sens curr 350a,  Stegen et al. 2012
    at=0.00610779	 (/ms)   		: Stegen et al. 2012
    bt=0.0817741	 (/ms)	 		: Note: typo in Stegen et al. 2012    
    celsius         (degC)  		: unused if q10 == 1.    
    q10 = 1.                              	: temperature scaling
    : va1 = -14.83 (mV) 	
    : va2 = -105.82 (mV) : 34 (mV)
    : va3 = 19.23 (mV)
    : gkir = 1.44e-02  (mS/cm2) 
    : ek = -70 (mV)
    : NormK = 0.81 
	
}

ASSIGNED {
    v 		(mV)
    ik      (nA)
    
    ki      (mM)
    ko      (mM)
    tauKir (ms)
    nkir (1)
}

STATE { n }

INITIAL { n = 0 }


BREAKPOINT {
    SOLVE state METHOD derivimplicit
    ik = n *PKir* z*F*(ki - ko * exp(-dz*v/vs))/(exp(-delta*(DeltazB+dz)*(v-vkir)/vs) + 1)
        : ik = (0.001)*gkir * ( v - ek*NormK - va1) *sqrt(((ko)/(1 (mM)))/(1+exp((v-ek*NormK-va2)/va3)))		: calculate ik 
	: printf("v: %g, ko: %g, va2: %g\n", v, ko, va2)
        : consider different channel dynamics
        : gating particle added to equation 
    }
    
    DERIVATIVE state {
        rate(v)
        n' = (nKir - n) / tauKir
    }
    PROCEDURE rate(v (mV)) {
        LOCAL qt
        nkir = 1/(1 + exp((v-vkir)/kl))			: l_steadystate
	qt=q10^((celsius-33)/10) 	
 	tauKir = 1/(qt *(at*exp(-v/vhalft) + bt*exp(v/vhalft) ))	
   
    }
