TITLE sodium calcium exchange
: taken from Courtemanche et al Am J Physiol 1998 275:H301

NEURON {
    SUFFIX ncx
    THREADSAFE
    USEION ca READ cao, cai WRITE ica
    USEION na READ nao, nai WRITE ina
    RANGE ica, ina , itot, imax, rate
    GLOBAL kna, kca, gamma, cont_factor, ksat, naf, is_fixed
    NONSPECIFIC_CURRENT incx
}

UNITS {
    (mM) = (milli/liter)

    (mA) = (milliamp)
    (mV) = (millivolt)
    F = (faraday) (coulombs)
    R 	= (k-mole)	(joule/degC)
}

PARAMETER {
    imax	= 1.2       (mA/cm2) 
    kna	=  87.5     (mM)
    kca	=  1.38     (mM)
    gamma	= .35	(1)	: voltage dependence factor
    cont_factor = -1    (1)
    allost = 152 (mM)
    ksat = 0.1          (1)
    naf = 15.0      (mM)
    is_fixed = 1        (1)
}

ASSIGNED {
    celsius	(degC)
    v	(mV)
    ica	(mA/cm2)
    ina	(mA/cm2)
    itot	(mA/cm2)
    incx   (mA/cm2)
    rate    (mA/cm2)
    cao	(mM)
    cai	(mM)
    nao	(mM)
    nai	(mM)
}
INITIAL {
    rate = 0
}

BREAKPOINT {
    :LOCAL rate
    if (is_fixed==0) {
        rate = pumprate(v,nai,nao,cai,cao)
    }
    if (is_fixed==1) {
        rate = pumprate(v,naf,nao,cai,cao)
    }
    ina =  3*rate
    ica = -2*rate
    incx=itot*cont_factor
}

FUNCTION pumprate(v(mV),nai(mM),nao(mM),cai(mM),cao(mM)) (mA/cm2){
    LOCAL q10, Kqa, KB, k,allo
    k = (1e-3)*F/(R*(celsius + 273.14))
    q10 = 3^((celsius - 37)/10 (degC))
    Kqa = exp(gamma*v*k)
    KB = exp( (gamma - 1)*v*k)
    allo = 1/(1+((allost/cai)^2))
    pumprate = q10*imax*allo*(Kqa*nai*nai*nai*cao-KB*nao*nao*nao*cai)/((kna*kna*kna + nao*nao*nao)*(kca + cao)*(1 + ksat*KB))
}
