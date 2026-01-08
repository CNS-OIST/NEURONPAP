COMMENT
Longitudinal diffusion of potassium (no buffering)
Savtchenko et al., 2018
ENDCOMMENT

NEURON {
	SUFFIX kdifl
	USEION k READ ik, ki WRITE ki
	RANGE Dk, ki0, iextra, no_clamp, gap
}

PARAMETER {
    no_clamp = 1 (1) : flag for turning off ki changes
    gap = 1 (1) :flag for mediating gapjunction
    tau_k = 1 (ms)
    ki0 = 110 (mM)
	Dk = 0.6 (micron2/ms)
	iextra = 0 (milliamp/cm2)
	  }

UNITS {
	
	(mM) = (milli/liter)
	(um) = (micron)
	FARADAY = (faraday) (coulomb)
	PI = (pi) (1)
	
}

INITIAL {
	ki = ki0
	ka = ki
  tau_k = diam * diam / 4 /Dk

}

ASSIGNED {
	ik (milliamp/cm2)
	diam (um)
	ki       (mM)
}

STATE {
	ka (mM)
}



BREAKPOINT {
  if (gap == no_clamp){
    ki = ki0
  } else {
    SOLVE diff METHOD sparse
  }
  :if (gap == 1) {
  :  printf("%g\n",tau_k)
  :}
}

KINETIC diff {
	COMPARTMENT PI*diam*diam/4 {ka}
	LONGITUDINAL_DIFFUSION Dk*diam*diam {ka}
	: LONGITUDINAL_DIFFUSION Dk {ka}
	~ ka << (-(ik-iextra)/(FARADAY)*PI*diam/2*(1e4)*no_clamp)
  ~ ka <<  ((ki0 - ka)/tau_k*PI*diam*diam/4*gap)
  if (no_clamp == 0 && gap == 0) {
    printf("%g\n",ka)
  }
	ki = ka 
  }

