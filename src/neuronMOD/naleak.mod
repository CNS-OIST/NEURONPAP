TITLE sodium leak model

COMMENT

ENDCOMMENT
NEURON {
    SUFFIX naleak
    USEION na READ nao,nai,ena WRITE ina
    RANGE  gleak

}

UNITS {
    (molar) = (1/liter)
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S)  = (siemens)
    (mM) =	(millimolar)
    (J)  = (joules)
    (um) = (micron)
    (mS)  = (millisiemens)
    (uM) = (micromolar)
    F  = (faraday) (coulombs)

}

CONSTANT {
    T = 300	(degC)
    R        = 8.3145   (J/degC) 	
    z = 1
}


PARAMETER {
    gleak = 1 (S/cm2) : ratio from Kalia et al. (2021) * Janic et al K leak
}

ASSIGNED {
    area (um2)
    v 		(mV)
    ina      (mA/cm2)
    ena     (mV)
    nao (mM)
    nai (mM)
}



BREAKPOINT {
    ina = gleak * (v - ena) 
    : divided by estimated surface area Radulescu A. et al (2022)
    : ik = (0.001)*gkir * ( v - ek*NormK - va1) *sqrt(((ko)/(1 (mM)))/(1+exp((v-ek*NormK-va2)/va3)))		: calculate ik 
    : printf("v: %g, ko: %g, va2: %g\n", v, ko, va2)
    : consider different channel dynamics
}
:    PROCEDURE calcENA(){
:        ena = NERNST(nao, nai, z)
:    }
:    FUNCTION NERNST (co (mM), ci (mM), zion (1)) (mV) {
:        NERNST = (1e3) * R*T/F/zion * log(co/ci)
:}
