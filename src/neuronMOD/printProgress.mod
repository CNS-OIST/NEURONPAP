TITLE Print Progress
COMMENT
prints time progress

ENDCOMMENT

NEURON {
    SUFFIX prnt_prog
    RANGE completion
    THREADSAFE
}
PARAMETER {
    completion  = 100 (ms)
}

BREAKPOINT {
    print_progress()
}
PROCEDURE print_progress(){
    printf("Completion:%2f\r",t/completion*100) 
}


