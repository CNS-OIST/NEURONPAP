: Voltage gap junstion
NEURON {
    POINT_PROCESS Gap
    NONSPECIFIC_CURRENT i_gap
    :USEION k READ ki WRITE ki
    
    RANGE r, i, VoltageGap,multiple
}
UNITS {
    (molar) = (1/liter)
  
    (mM) =	(millimolar)
    (pS) = (picosiemens)
  }

PARAMETER {
    r = 1(megohm)
    multiple = 1000 (1)
    uS = 56 (pS)
    VoltageGap = -85 (millivolt)
    tau_k = 1 (ms): instantaneous
    ki_0 = 120 (mM)
}

INITIAL {
      r = (1e6)/(uS * multiple)
  }

ASSIGNED {
    v (millivolt)
    vgap (millivolt)
    i_gap (nanoamp)
}

BREAKPOINT {
    i_gap = (v-VoltageGap)/r
}

