TITLE Natrium-Kalium Pump

COMMENT
    Somjen 2008 et al.
  
ENDCOMMENT


NEURON {
    SUFFIX nakpump
    USEION k READ ko WRITE ik
    USEION na READ nai WRITE ina
    RANGE ik_pump, ina_pump, km_k, km_na, totalpump :qna, qk
}

UNITS {
    (mV)	= (millivolt)
    (molar) = (1/liter)
    (mM)	= (millimolar)
    (um)	= (micron)
    (mA)	= (milliamp)
    (mol)	= (1)
    :FARADAY	= (faraday) (coulomb)
    FARADAY		= 96485.309 (coul)
    PI	= (pi)		(1)
    R 	= (k-mole)	(joule/degC)
}

PARAMETER {
    celsius		(degC)
    km_k = 2		(mM) 
    km_na = 10		(mM)
    totalpump = 1	(mA/cm2)  
    : set to 0 in hoc if this pump not wanted
}

STATE { qna qk }

ASSIGNED {
    ik		(mA/cm2)
    ina		(mA/cm2)
    ik_pump	(mA/cm2)
    ina_pump		(mA/cm2)
    ko		(mM)
    nai		(mM)
    diam		(um)
    L		(um)
}

BREAKPOINT {
    ik_pump = -2*totalpump*stroom(nai,ko)
    ina_pump = ik_pump * -3/2
    ik = ik_pump
    ina = ina_pump
}

INITIAL {
    qna=0
    qk=0
    ik_pump = -2*totalpump*stroom(nai,ko)
    ina_pump = ik_pump * -3/2
    ik = ik_pump
    ina = ina_pump
}

FUNCTION stroom(na (mM),k (mM)) {
    stroom =   ( 1 / ((1+km_k/k)*(1+km_k/k)) ) * ( 1 / ((1+km_na/na)*(1+km_na/na)*(1+km_na/na)) )
}
