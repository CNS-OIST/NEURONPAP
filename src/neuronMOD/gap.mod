: Voltage gap junstion
NEURON {
    POINT_PROCESS Gap
    NONSPECIFIC_CURRENT i_gap
    :USEION k READ ki WRITE ki
    
    RANGE r, i, VoltageGap
}
UNITS {
    (molar) = (1/liter)
  
    (mM) =	(millimolar)
  }

PARAMETER {
    r = 5(megohm)
    VoltageGap = -85 (millivolt)
    tau_k = 1 (ms): instantaneous
    ki_0 = 120 (mM)
}

:INITIAL {
:    ki = ki_0
:  }

:STATE {
:    ki (mM)
:  }

ASSIGNED {
    v (millivolt)
    vgap (millivolt)
    i_gap (nanoamp)
}

BREAKPOINT {
  :SOLVE state METHOD derivimplicit
    i_gap = (v-VoltageGap)/r
}

:DERIVATIVE state{
:  ki' = (ki_0 - ki)/tau_k
:  }
