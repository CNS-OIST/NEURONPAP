NEURON {
    SUFFIX segcolor
    RANGE cval,mode
}

PARAMETER {
    mode = 0 : 1 is max,-1 is min,
    cval  = 0(mV)
}

ASSIGNED {
  v (mV)
  }
INITIAL {
    cval = v
  }

BREAKPOINT{
    if (mode == 1) {
      if (cval < v) {
        cval = v
      }
    } else if ( mode == -1) {
      if (cval > v) {
      cval = v
      }
    } else {
        cval = v
      }
  }

