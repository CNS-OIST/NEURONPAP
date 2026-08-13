COMMENT
Model from CA1 pyramidal neuron: nonlinear a5-GABAAR controls synaptic NMDAR activation (Schulz et al 2018)
Based on the mod file written by Mark Cembrowski, 2015
Extended by a voltage-dependent outward rectification

This is an extension of the Exp2Syn class to incorporate tracking of the
specific features of different inhibitory synapses.  Specifically, this includes
whether a synapse:
	is Vgat+ (vgat)
	is Sst+ (sst)
	is Npy+ (npy)
These features are implement in order to track synapses and turn them on/off
as the simulated genotype demands.

This class also extended to have an isOn attribute, which acts as a switch
on whether the synapse is on (if = 0, conductance is always = 0; if = 1,
synapse behaves as normal).

As background, the Exp2Syn features are described as:

Two state kinetic scheme synapse described by rise time tau1,
and decay time constant tau2. The normalized peak condunductance is 1.
Decay time MUST be greater than rise time.

The solution of A->G->bath with rate constants 1/tau1 and 1/tau2 is
 A = a*exp(-t/tau1) and
 G = a*tau2/(tau2-tau1)*(-exp(-t/tau1) + exp(-t/tau2))
	where tau1 < tau2

If tau2-tau1 -> 0 then we have a alphasynapse.
and if tau1 -> 0 then we have just single exponential decay.

The factor is evaluated in the
initial block such that an event of weight 1 generates a
peak conductance of 1.

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method.


ENDCOMMENT

NEURON {
    USEION cl READ ecl
    POINT_PROCESS inhSyn
    NONSPECIFIC_CURRENT iGaba
    RANGE tau1, tau2, e, i
    RANGE g
    RANGE vgat,sst,npy,pv,xEff,V50,slope_factor
    RANGE isOn,multiple
    THREADSAFE
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
    (uM) = (micro/liter)
    (mM) = (milli/liter)
    (pS) = (picosiemens)
}

PARAMETER {
    hilln = 1.5  (1)
    gabaEC = 7 (uM) :gingrich1995 J physiology 
    tau1=.1 (ms) <1e-9,1e9>
    tau2 = 10 (ms) <1e-9,1e9>
    e=-70	(mV)
    g_max = 28 (pS)
    vgat=0
    sst=0
    npy=0
    pv=0
    xEff=-1
    isOn=1
    multiple=0
    V50=-52 (mV)
    slope_factor=3 (mV)
}
CONSTANT {
    z = -1
}

ASSIGNED {
    v (mV)
    iGaba (nA)
    g (pS)
    ecl (mV)
    factor
}

STATE {
    A 
    B 
}

INITIAL {
    LOCAL tp
    if (tau1/tau2 > .99) {
        tau1 = .99*tau2
    }
    A = 0
    B = 0
    tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
    factor = -exp(-tp/tau1) + exp(-tp/tau2)
    factor = 1/factor
}

BREAKPOINT {
    SOLVE state METHOD derivimplicit
    g = g_max*rect(v)*(B - A)*isOn*multiple
    : printf("%g\n",ecl)
    iGaba = (1e-06)*g*(v - ecl)
}

DERIVATIVE state {
    A' = -A/tau1
    B' = -B/tau2
}

NET_RECEIVE(weight) {
    A = A + hillGaba(weight * 1 (mM))*factor
    B = B + hillGaba(weight * 1 (mM))*factor
}


FUNCTION rect (v(mV))( ){
    rect= 1+(0.25-1)/ ( 1. + exp (( v - V50 )/slope_factor) ) 
}

FUNCTION hillGaba(gabaConc (mM)){

    hillGaba = 1/(1 + pow((1e-3)*gabaEC/gabaConc,hilln))
}
