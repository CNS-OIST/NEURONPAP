TITLE potassium leak model

COMMENT
This is kinetics of Kir4.1 channels used in the Janic et al. 2022 paper.
MAYBE needs electrode current statement


ENDCOMMENT
NEURON {
    SUFFIX kleak
    USEION k READ ek WRITE ik
    RANGE  gleak
        
}

UNITS {
	(molar) = (1/liter)
	(mA) = (milliamp)
        (mV) = (millivolt)
        (uS)  = (microsiemens)
	(mM) =	(millimolar)
	(J)  = (joules)
        (um) = (micron)
        
    }
    
    

PARAMETER {
    gleak = 0.001 (uS)
}

ASSIGNED {
    v 		(mV)
    ik      (mA/cm2)
    ek      (mV)
}



BREAKPOINT {
    ik = (100) *gleak * (v - ek) / (4e5 (um2))
        : divided by estimated surface area Radulescu A. et al (2022)
        : ik = (0.001)*gkir * ( v - ek*NormK - va1) *sqrt(((ko)/(1 (mM)))/(1+exp((v-ek*NormK-va2)/va3)))		: calculate ik 
	: printf("v: %g, ko: %g, va2: %g\n", v, ko, va2)
        : consider different channel dynamics
    }
    