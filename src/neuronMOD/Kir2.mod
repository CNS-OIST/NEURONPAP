TITLE inward rectifier potassium (Kir) channel

COMMENT

Based on the Mod File by A. Hanuschkin <AH, 2011> for:
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
  
  - sqrt effect of external potassium added following 
  - model maximum conductance reduced to single channel resistance measured by Yang et al. 2000
  
  R. J. Nakatani(c) 2023
  - 

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
        q10 = 1.                              	: temperature scaling
        A = 0.09534626                          : fit to sqrt rule and match single channel conductance 50 pS at Yang condition
multiple = 0 (1)
density = 370e8 (/cm2)
density_std = 1e8 (/cm2)
}



NEURON {
	SUFFIX kir2 			
	USEION k READ ek,ko WRITE ik	
        RANGE  gkbar, vhalfl, kl, vhalft, at, bt, q10, multiple,count,count_std
        RANGE ik_kir,gk
        GLOBAL linf,taul
        
        THREADSAFE
}


STATE {
        l
}

ASSIGNED {
        ik                              (mA/cm2)
        ik_kir                          (mA/cm2)
        gk                              (pS/cm2)
        ek                              (mV)
        linf      (1)
        taul (ms)
        ko                              (mM)
        area (um2)
        celsius (degC)
        count (1)
        count_std(1)
}


INITIAL {
	rate(v)
	l=linf
  count = (1e-08) * area * density
  count_std = (1e-08) * area * density_std
  gk = (1e8) * gkbar*(A*sqrt(ko/1 (mM))) * (count + multiple * count_std)/area
}



BREAKPOINT {
  LOCAL updatedCount
	SOLVE states METHOD derivimplicit	: solve differential equations in states with method 'cnexp'
    updatedCount = (count + multiple * count_std)
  if (updatedCount < 0){
    updatedCount = 0
  }
  if (ko <0){
      gk = 0
  } else {
    gk = (1e8) * gkbar*(A*sqrt(ko/1 (mM))) * updatedCount /area
  }


        : printf("%g\n",area)
        : printf("%g\n",gk*area*(1e-8))
	: use state l to calulate gk
        : area will be multiplied per section resulting in single channel conductance per segment.
        : calculate gkbar from fitting single channel recording
        ik_kir = (1e-12)*gk *l* ( v - ek )		: calculate ik 
        ik = ik_kir
}



DERIVATIVE states {     
        rate(v)
        l' =  (linf - l)/taul		: differential equation 
}

PROCEDURE rate(v (mV)) { :callable from hoc
    LOCAL qt 
    qt=q10^((celsius-33)/10(degC))
        linf = 1/(1 + exp((v-vhalfl)/kl))			: l_steadystate fit janiac data
 	taul = 1/(qt *(at*exp(-v/vhalft) + bt*exp(v/vhalft) ))
}
