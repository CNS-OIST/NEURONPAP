TITLE chloride leak model

COMMENT
This is kinetics of Kir4.1 channels used in the Janic et al. 2022 paper.
MAYBE needs electrode current statement


ENDCOMMENT
NEURON {
    SUFFIX clleak
    USEION cl WRITE icl,ecl,clo VALENCE -1
    USEION k READ ko
    USEION na READ nao
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
        (uM) = (micromolar)
        F  = (faraday) (coulombs)
    }
    CONSTANT {
	T = 300	(degC)
 	R        = 8.3145   (J/degC) 	
        z = -1
    }
    
    INITIAL {
        calcECL()
        }
    

PARAMETER {
    gleak = 5.57e-6 (uS) : ratio from Kalia et al. (2021) * Janic et al K leak
    :    cli = 7.6 (mM) : Thapaliya P et al 2023
    cli_0 = 30 (mM): Verkhratsky A. review Adv Exp Med Biol 2019
    clo_0 = 130 (mM): Untiet V. Nat Comm. 14, Article number: 1871 (2023) 
}

ASSIGNED {
    clo (mM)
    ko (mM)
    nao (mM)
    v 		(mV)
    icl      (mA/cm2)
    ecl     (mV)
}



BREAKPOINT {
    calcECL()
    icl = (100) * gleak * (v - ecl) / (4e5 (um2)) 
        : divided by estimated surface area Radulescu A. et al (2022)
        : ik = (0.001)*gkir * ( v - ek*NormK - va1) *sqrt(((ko)/(1 (mM)))/(1+exp((v-ek*NormK-va2)/va3)))		: calculate ik 
	: printf("v: %g, ko: %g, va2: %g\n", v, ko, va2)
        : consider different channel dynamics
    }
    
    PROCEDURE calcECL(){
        ecl = NERNST(clo_0, cli_0, z)
    }
    FUNCTION NERNST (co (mM), ci (mM), zion (1)) (mV) {
        NERNST = (1e3) * R*T/F/zion * log(co/ci)
}