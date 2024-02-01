COMMENT

The model of Glutamate  transporter.
is based on two papers, 

from the paper 

1. Zhang Z1, Tao Z, Gameiro A, Barcelona S, Braams S, Rauen T, Grewer C. 
Transport direction determines the kinetics of substrate transport by the glutamate transporter EAAC1.
Proc Natl Acad Sci U S A. 2007 Nov 13;104(46):18025-30. Epub 2007 Nov 8.

we determine the basic kinetic scheme for glutamate transporters, 

from the  paper
 
2. Bergles, D.E. & Jahr, C.E. 
Synaptic activation of glutamate transporters in hippocampal astrocytes. Neuron 19, 1297-1308 (1997).

we corrected the numerical values of the kinetic constants corresponding to the dynamics of glutamate transporters in astrocytes




ENDCOMMENT

NEURON {
    POINT_PROCESS  GluTrans
    USEION k READ ki,ko
    USEION na READ nao,nai
    RANGE part, C1, C2, C3, C4, C5, C6
    RANGE  iGluT, Gluout, density, itransLog,multiple
    NONSPECIFIC_CURRENT iGluT


}

UNITS {
    (l) = (liter)
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (mA) = (milliamp)
    (pS) = (picosiemens)
    (umho) = (micromho)
    (mM) = (milli/liter)
    (uM) = (micro/liter)
    F = (faraday) (coulombs)
    PI      = (pi)       (1)
    (um) = (micrometer)
}

PARAMETER {	
    : Rates

    k12 = 20           (l /mM /ms)
    k21 = 0.1          (/ms)
    k23 = 0.015       (l /mM /ms)
    k32 = 0.5          (/ms)
    k34 = 0.2          (/ms)
    k43 = 0.6          (/ms)
    k45 = 4            (/ms)
    k54 = 10           (l /mM /ms)
    k56 = 1            (/ms) 
    k65 = 0.1          (l /mM /ms) 
    k16 = 0.0016          (l /mM /ms)
    k61 =  2e-4        (l /mM /ms)
    
    Gluin = 0.3      (mM/l)
    Gluout_0 = 20e-6	(mM/l)

    density =1e12  : (/cm2) : 10000 per um2
    charge = 1.6e-19 (coulombs)
    synCleftSpace = 1.41e-15 (liter)
    : PSD 200 nm Cleft Height 20 nm
    tau1 = 0.61 (ms) : Rise of Glutamate in cleft (From astrocyte POV) Diamnon J.S. 2005 J Neurosci
    tau2 = 5.8 (ms) : Fall of Glutamate in cleft (From astrocyte POV) Diamnon J.S. 2005 J Neurosci
    multiple = 1 : Count of GluT
}

ASSIGNED {
    v	   (mV)		:  voltage
    iGluT (nA)            : 
    surf   (cm2)
    volin  (liter)
    volout (liter)
    itransLog
    ki (mM)
    ko (mM)
    nai (mM)
    nao (mM)
    Kout (mM/liter)
    Kin (mM/liter)
    Naout (mM/liter)
    Nain (mM/liter)
    area (um2)
    Gluout (mM/liter)
    tSyn (ms)
    flag
    maxGlu (mM/liter)
}

STATE {
    : Transporter  states (all fractions)
            : 
    C1	(/cm2)	:  
    C2	(/cm2)	:  
    C3	(/cm2)	: 
    C4	(/cm2)	: 
    C5	(/cm2)	: 
    C6  (/cm2)
}

INITIAL {
    C1= 0.9074    
    C2= 0.0199    
    C3= 0.0435    
    C4= 0.0103    
    C5= 0.0142    
    C6= 0.0047
    volin = 1
    volout = 1
    surf = 1
    koi(ki,ko)
    naoi(nai,nao)
    Gluout = Gluout_0
    tSyn = 0
    maxGlu = 0 
}
NET_RECEIVE(weight) {
    tSyn = t
    maxGlu = weight * 1 (mM) / 1 (liter)
    : printf("Glu:%g\n",Gluout)
}


BREAKPOINT {
    koi(ki,ko)
    naoi(nai,nao)
    SOLVE kstates METHOD sparse
    : printf("%g,%g\n",C1,C2)
    gluDiff(maxGlu,tSyn)
        
    iGluT=-charge*(1e+004)*(0.6*(C1*k16*Kout*u(v,0.6)-C6*k61*Kin) -0.1*(C1*k12*Gluout*u(v,-0.1)-C2*k21)+0.5*(C2*k23*Naout*u(v,0.5)-C3*k32)+0.4*( C3*k34*u(v,0.4)-C4*k43)+0.6*(C5*k56*u(v,0.6)-C6*k65*Nain) ) * multiple * area
    : if (Gluout > Gluout_0){
    :     printf("%g:%g\n",Gluout,iGluT)
        
        
    : }
    : itransLog=log(-iGluT*(1e+006))
    
    :iGluT=-charge*density*(1e+006)*(0.6*(C1*k16*Kout*u(v,0.6)-C6*k61*Kin) +0.4*( C3*k34-C4*k43)+0.6*(C5*k56*u(v,0.6)-C6*k65*Nain) )
}

KINETIC kstates {
            COMPARTMENT volin { Nain Kin Gluin}
            COMPARTMENT volout { Naout Kout Gluout}
            : COMPARTMENT surf { C1 C2 C3 C4 C5 C6}
        : surf=1 : !!!!!!!
        ~ C1   <-> C2      (Gluout*k12*u(v,-0.1), k21)
        ~ C2  <-> C3       (Naout*k23*u(v,0.5),k32)
        ~ C3 <-> C4	       (k34*u(v,0.4),k43)
        ~ C4 <-> C5 	   (k45,k54*Gluin)
        ~ C5 <-> C6	       (k56*u(v,0.6),k65*Nain)
        ~ C6  <-> C1       (Kin*k61, k16*u(v,0.6)*Kout)
        
    CONSERVE C1+C2+C3+C4+C5+C6= 1
}


PROCEDURE gluDiff(maxGlu (mM/liter),tSyn(ms)){
    Gluout = Gluout_0 + maxGlu*(tau2/(tau2-tau1)*(-exp(-(t-tSyn)/tau1) + exp(-(t- tSyn)/tau2)))
    : if (maxGlu > 0){
    :     printf("%g,%g\n",maxGlu,Gluout)
    : }
}

FUNCTION u(x(mV), th) {
    u = exp(th*x/(2*(26.7 (mV))))
}


PROCEDURE koi(ki(mM),ko(mM)){
    Kin = ki/1 (liter)
    Kout = ko/1(liter)
}

PROCEDURE naoi(nai(mM),nao(mM)){
    Nain = nai/1 (liter)
    Naout = nao/1(liter)
}