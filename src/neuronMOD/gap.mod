: Voltage gap junstion
NEURON {
    POINT_PROCESS Gap
    NONSPECIFIC_CURRENT i
    RANGE r, i, VoltageGap
}

PARAMETER {
    r = 100000(megohm)
    VoltageGap = -85 (millivolt)
}

ASSIGNED {
    v (millivolt)
    vgap (millivolt)
    i (nanoamp)
}

BREAKPOINT {
    i = (v-VoltageGap)/r
}
