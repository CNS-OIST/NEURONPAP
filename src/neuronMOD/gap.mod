: Voltage gap junstion
NEURON {
    POINT_PROCESS Gap
    NONSPECIFIC_CURRENT i_gap
    :USEION k READ ki WRITE ki
    
    RANGE  i, VoltageGap,multiple,g
}
UNITS {
    (molar) = (1/liter)
  
    (mM) =	(millimolar)
    (pS) = (picosiemens)
  }

PARAMETER {
    multiple = 50 (1)
    g_max = 56 (pS) :from 56 unitary conductance measured in culture
    VoltageGap = -85 (millivolt)
    tau_k = 1 (ms): instantaneous
    ki_0 = 120 (mM)
}
ASSIGNED {
  g (pS)
}

INITIAL {
  if (multiple < 0) {
      multiple = 0
    }
      g = (g_max * multiple)
  }

ASSIGNED {
    v (millivolt)
    vgap (millivolt)
    i_gap (nanoamp)
}

BREAKPOINT {
    g = (g_max * multiple)
    i_gap = (1e-06)*g*(v-VoltageGap)
}

