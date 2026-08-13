TITLE potassium leak model

COMMENT

ENDCOMMENT
NEURON {
    SUFFIX kleak
    USEION k READ ek WRITE ik
    RANGE  gleak,ik_leak

}

UNITS {
    (molar) = (1/liter)
    (mA) = (milliamp)
    (mV) = (millivolt)
    (mM) =	(millimolar)
    (J)  = (joules)
    (um) = (micron)
    (S) = (siemens)

}



PARAMETER {
    gleak = 1 (S/cm2) 
}

ASSIGNED {
    v 		(mV)
    ik      (mA/cm2)
    ik_leak      (mA/cm2)
    ek      (mV)
}



BREAKPOINT {
    ik_leak = gleak * (v - ek) 
    ik = ik_leak
}

