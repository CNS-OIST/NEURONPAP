TITLE Potassium ion accumulation

NEURON {
	SUFFIX k_acc
	USEION k READ ko, ik WRITE ko
        RANGE tauk_0, ko0, flag, kbath,kbath_change, flux, flux_change,slowing
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
  PI = (pi) (1)
}

PARAMETER {    
    ko0 = 2.5 (mM)
    fhspace = 400 (angstrom) : effective thickness 
    tauk_0 = 4 (ms) :Ransom C.B. (2000) Journal of Physiology
    flag  = 0 (1)
    slowing = 1 (1)
    startFlag = 0 (1)
    tstart = 0 (1)
    : not considering torosity as ECS is nanoscopic
    Dk = 19.2e-6 (cm2/s)
    : arbitrary permeability ratio compared to free for ECS outside
    perm = 7.5e-5 (cm/s)
    memL = 80 (angstrom)
    :partition = 70 (1) :2/140
    openMem = 1(1)

}

ASSIGNED {
    tauk (ms)
    ik 	(mA/cm2)
    kbath (1)
    kbath_change(mM/ms)
    flux (1)
    flux_change(mM/ms)
    dt (ms)
    iNMDA (mA/cm2)
    iGluT (mA/cm2)
    diam (micron)
    L (micron)  
    d_eff (micron2/ms)
}

STATE {
    ko (mM)
}

INITIAL {
    ko = ko0
    d_eff = (100000) * Dk /1.6/1.6: Hrabe (2019) biophysj 
    :perm = (10) * partition*Dk/memL
    :perm_outside =0.5 * diam/6.5 : lineraly scale permeability
    tauk_0 = 1/(d_eff * 2*openMem/log(2*(fhspace*(1e-4)+diam/2)/diam)/(fhspace*fhspace*(1e-8)+diam*fhspace*(1e-4)))
    if (tauk_0 < 0.01) {
      : dont simulate very fast dissipation
      flag = 2
      }else{
        :printf("%g\n",tauk)
        flag = 0
        }
    tauk = tauk_0
    kbathRate()
        :printf("%g\n",tauk)
}

BREAKPOINT {
        kbathRate()
        SOLVE state METHOD derivimplicit
        if (ko <= 0){
            ko = 0
        }
    }
    
    DERIVATIVE state {
        kbathRate()
        : if (ko0 > 2.5){
        :     printf("%g\n",ko0)
        : }
        ko' = flux * (1e8)*ik /(fhspace*F) + kbath * (ko0-ko)/tauk
        flux_change =(1e8)*ik /(fhspace*F) 
        kbath_change =(ko0-ko)/tauk
 
        : printf("%g, %g, %g, %g\n",flag,ik,kbath,(1e8)*ik /(fhspace*F)/kbath)
    }
PROCEDURE kbathRate(){
    if (flag > 0){
        flux = 0
        tauk = dt 
        kbath = 1
        :printf("%g\n",ko)
    } else {
        flux =  1
        tauk = slowing * tauk_0: slowing 
        kbath =  1
        :printf("%g, ",kbath)
    }
    
    if (flag == 1) {
        : instantaneous free bath mode for one step
        flag = 0   
    }
}
    
