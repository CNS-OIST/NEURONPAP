TITLE inward rectifier potassium (Kir) channel

COMMENT

Mod File by A. Hanuschkin <AH, 2011> for:
Yim MY, Hanuschkin A, Wolfart J (2015) Hippocampus 25:297-308.
http://onlinelibrary.wiley.com/doi/10.1002/hipo.22373/abstract

Channel description and parameters from:
Stegen M, Kirchheim F, Hanuschkin A, Staszewski O, Veh R, and Wolfart J. Cerebral Cortex, 22:9, 2087-2101, 2012.

Mod File history:
- tau(V), linf(V) fitted to experimental values of human dentate gyrus granual cells
- ModelDB file adapted from 
  Wolf JA, Moyer JT, Lazarewicz MT, Contreras D, Benoit-Marand M, O'Donnell P, Finkel LH (2005) J Neurosci 25:9080-95
  https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=112834&file=/nacb_msp/kir.mod
- file modified to uses nomoclature of 
  Li X, Ascoli GA (2006) J of Comput Neurosci 21(2):191-209 
  Li X, Ascoli GA (2008) Neural Comput 20:1717-31

A. Hanuschkin(c) 2011,2012

ENDCOMMENT


UNITS {
    (mA) = (milliamp)
    (fA) = (femtoamp)
    (molar) = (1/liter)
    (mM) =	(millimolar)
    (um) = (micron)

    (mV) = (millivolt)
    (S)  = (siemens)
    (pS) = (picosiemens)
}

PARAMETER {
	v 		(mV)
	gkbar  = 50	(pS) 	: 50 pS single channel conductance Yang et al 2000 

	: Boltzman steady state curve	
        vhalfl = -98.92  (mV)    		: fitted to patch data, Stegen et al. 2012
        kl = 10.89       (mV)    		: Stegen et al. 2012

	: tau_infty 
        vhalft=-59.6170749	 (mV)    	: refitted with VC from Olsen 2012 Methods Mol Biol
        at=1.98752141	 (/ms)   		: 
	bt=0.0143908141	 (/ms)	 		: 

	: Temperature dependence
        : celsius          (degC)  		: unused if q10 == 1.
        : q10 = 1.                              	: temperature scaling
        A = 0.09534626                          : fit to sqrt rule
}



NEURON {
	SUFFIX kir2 			
	USEION k READ ek,ko WRITE ik	
        RANGE  gkbar, vhalfl, kl, vhalft, at, bt, q10 
        GLOBAL linf,taul
        
        THREADSAFE
}


STATE {
        l
}

ASSIGNED {
        ik                              (mA/cm2)
        gk                              (S/cm2)
        ek                              (mV)
        linf      (1)
        taul (ms)
        ko                              (mM)
        area (um2)
}


INITIAL {
	rate(v)
	l=linf
}


BREAKPOINT {
	SOLVE states METHOD cnexp	: solve differential equations in states with method 'cnexp'
	gk = (0.0001) * gkbar*(A*sqrt(ko/1 (mM)))/area
	: use state l to calulate gk
        : calculate gkbar from fitting single channel recording
        ik = gk *l* ( v - ek )		: calculate ik 
}



DERIVATIVE states {     
        rate(v)
        l' =  (linf - l)/taul		: differential equation 
}

PROCEDURE rate(v (mV)) { :callable from hoc
        LOCAL qt
	: qt=q10^((celsius-33)/10)
        qt = 1
        linf = 1/(1 + exp((v-vhalfl)/kl))			: l_steadystate fit janiac data
 	taul = 1/(qt *(at*exp(-v/vhalft) + bt*exp(v/vhalft) ))
}
