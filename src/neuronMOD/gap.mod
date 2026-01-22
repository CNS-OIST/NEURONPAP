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
    multiple = 1 (1)
    uS = 5.6 (pS) :from 56 unitary conductance split by average number of gap junctions between cells
    VoltageGap = -85 (millivolt)
    tau_k = 1 (ms): instantaneous
    ki_0 = 120 (mM)
}

INITIAL {
  if (multiple < 0) {
      multiple = 0
    }
      r = (1e6)/(uS * multiple)
  }

ASSIGNED {
    v (millivolt)
    vgap (millivolt)
    i_gap (nanoamp)
}

BREAKPOINT {
    r = (1e6)/(uS * multiple)
    i_gap = (v-VoltageGap)/r
}

