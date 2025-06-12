TITLE Triple-exp model of NMDAR has (HH-type gating) (temp. sensitivity) (voltage-dependent time constants) (desensitization)

COMMENT
This is a Triple-exponential model of an NMDAR 
that has a slow voltage-dependent gating component in its conductance
time constants are voltage-dependent and temperature sensitive

Mg++ voltage dependency from Spruston95 -> Woodhull, 1973 

Desensitization is introduced in this model. Actually, this model has 5 differential equations
becasue desensitization is solved numerically. 
It can be reduced to 3 by solving its A state analitically.
For more info read the original paper. 

Keivan Moradi 2012

TODO
[ ] check maximum VD change

ENDCOMMENT

NEURON {
    POINT_PROCESS Exp5NMDA
    NONSPECIFIC_CURRENT iNMDA
	RANGE tau1, tau1_0, tau2_0, a2, b2, wtau2, tau3_0, a3, b3, tauV, e, i, gVI, gVDst, gVDv0, Mg, K0, delta, tp, wf, tau_D1, d1,multiple, shift
	THREADSAFE
}

UNITS {
	(nA) = (nanoamp)
	(fA) = (femtoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
	(mM) = (milli/liter)
	(uM) = (micro/liter)
	(pS) = (picosiemens)
	(um) = (micron)
	(J)  = (joules)
}

PARAMETER {
: Parameters Control Neurotransmitter and Voltage-dependent gating of NMDAR
tau1_0 = 1.69		(ms)	<1e-9,1e9>	: Spruston95 CA1 dend [Mg=0 v=-80 celcius=18] be careful: Mg can change these values
a1 = 0.09 (ms)
b1 = 0.03 (1/mV)
: parameters control exponential rise to a maximum of tau2
: tau2_0 = 3.97	(ms)
tau2_0 = 19 (ms)
	a2 = 0.70		(ms)
	b2 = 0.0243		(1/mV)
	wtau2= 0.95		<1e-9,1> : Hestrin90 0.65
	
: parameters control exponential rise to a maximum of tau3
	tau3_0 = 41.62	(ms)
	a3 = 34.69		(ms)
	b3 = 0.01		(1/mV)
	: Hestrin90 CA1 soma  [Mg=1 v=-40 celcius=30-32] the decay of the NMDA component of the EPSC recorded at temperatures above 30 degC 
	: the fast phase of decay, which accounted for 65%-+12% of the decay, had a time constant of 23.5-+3.8 ms, 
	: whereas the slow component had a time constant of 123-+83 ms.
	: wtau2= 0.78 Spruston95 CA1 dend [Mg=0 v=-80 celcius=18] percentage of contribution of tau2 in deactivation of NMDAR
	Q10_tau1 = 2.2			: Hestrin90
	Q10_tau2 = 3.68			: Hestrin90 -> 3.5-+0.9, Korinek10 -> NR1/2B -> 3.68
	Q10_tau3 = 2.65			: Korinek10
	T0_tau	 = 35	(degC)	: reference temperature 
	: Hestrin90 CA1 soma  [Mg=1 v=-40 celcius=31.5->25] The average Q10 for the rising phase was 2.2-+0.5, 
	: and that for the major fast decaying phase was 3.5-+0.9
	tp = 30			(ms)	: time of the peack -> when C + B - A reaches the maximum value or simply when NMDA has the peack current
							: tp should be recalculated when tau1 or tau2 or tau3 changes
: Parameters control desensitization of the channel
	: these values are from Fig.3 in Varela et al. 1997
	: the (1) is needed for the range limits to be effective
	: d1 = 0.2 	  	(1)		< 0, 1 >     : fast depression
	: tau_D1 = 2500 	(ms)	< 1e-9, 1e9 >
	d1 = 1	  	(1)		< 0, 1 >     : fast depression
	tau_D1 = 2500 	(ms)	< 1e-9, 1e9 >
: Parameters Control voltage-dependent gating of NMDAR
	tauV = 7		(ms)	<1e-9,1e9>	: Kim11 
							: at 26 degC & [Mg]o = 1 mM, 
							: [Mg]o = 0 reduces value of this parameter
							: Because TauV at room temperature (20) & [Mg]o = 1 mM is 9.12 Clarke08 & Kim11 
							: and because Q10 at 26 degC is 1.52
							: then tauV at 26 degC should be 7 
	gVDst = 0.007	(1/mV)	: steepness of the gVD-V graph from Clarke08 -> 2 units / 285 mv
	gVDv0 = -100	(mV)	: Membrane potential at which there is no voltage dependent current, from Clarke08 -> -90 or -100
	gVI = 33			(pS)	: Maximum Conductance of Voltage Independent component, This value is used to calculate gVD
        :additional change to fit -60 mV 33 pS = gVI no gVD for astrocyte NMDAR from Lalo Curve
	Q10 = 1.52				: Kim11
	T0 = 26			(degC)	: reference temperature 
	celsius 		(degC)	: actual temperature for simulation, defined in Neuron
: Parameters Control Mg block of NMDAR
	Mg = 1			(mM)	: external magnesium concentration from Spruston95
	: K0 = 4.1		(mM)	: IC50 at 0 mV from Spruston95
        K0 = 500               (mM)
        : shift = 0             (1) : theoretical shift from 0 mV Mg block
        shift = -71.5 (mV)
        delta = 10 (1)
	: delta = 0.01 	(1)		: the electrical distance of t        he Mg2+ binding site from the outside of the membrane from Spruston95
        : The Parameter Controls Ohm haw in NMDAR
        e = -3.3		(mV)	: in CA1-CA3 region = -0.7 from Spruston Lalo et al. 2006 from Verkhratsky lab
        multiple = 1 (1)
        flag = 0 (1)
        : glu = 1 (mM) 
        gluEC = 4.3 (uM) : From Nahum-Levy et al. 2001 Biophysical Journal
        hilln = 1.2 (1) : From Nahum-Levy et al. 2001 Biophysical Journal
        synWeight = 0.56 : From Moradi
}

CONSTANT {
	T = 273.16	(degC)
	F = 9.648e4	(coul)	: Faraday's constant (coulombs/mol)
	R = 8.315	(J/degC): universal gas constant (joules/mol/K)
	z = 2		(1)		: valency of Mg2+
}

ASSIGNED {
	v		(mV)
	dt		(ms)
        maxI (nA)
	iNMDA		(nA)
        prvI (nA)
	g		(pS)
	factor (1)
	wf (1)
        q10_tau1
	q10_tau2
	q10_tau3
	inf		(pS)
	tau		(ms)
        tau1 (ms)
	tau2	(ms)
	tau3	(ms)
	wtau3 (1)
        prvW (1)
        prvA (1)
        prvB (1)
        prvC (1)
        tPeak (ms)
        area (um2)
}

STATE {
	A		: Gating in response to release of Glutamate
	B		: Gating in response to release of Glutamate
	C		: Gating in response to release of Glutamate
	gVD (pS): Voltage dependent gating
    }
    
    INITIAL {
        prvW = 0
        prvI = 0
	Mgblock(v)
	: temperature-sensitivity of the of NMDARs
	q10_tau1 = Q10_tau1^((31.5 - celsius)/10(degC))
	q10_tau2 = Q10_tau2^((T0_tau - celsius)/10(degC))
	q10_tau3 = Q10_tau3^((T0_tau - celsius)/10(degC))
	: temperature-sensitivity of the slow unblock of NMDARs
	tau  = tauV * Q10^((T0 - celsius)/10(degC))
        
        rates(v)
        : factor should mirror at 0 mV
	wtau3 = 1 - wtau2
	: if tau3 >> tau2 and wtau3 << wtau2 -> Maximum conductance is determined by tau1 and tau2
	tp = tau1*tau2*log(tau2/(wtau2*tau1))/(tau2 - tau1)
	factor = (-exp(-tp/tau1) + wtau2*exp(-tp/tau2) + wtau3*exp(-tp/tau3))
	factor = 1/factor	
        : printf("tau:%g,%g,%g\n",tau1,tau2,tau3)
	: printf("factor:%g\n",factor)
	: printf("tp:%g\n",tp)
	: printf("T0:%g\n",T0_tau)
	A = 0
	B = 0
	C = 0
	gVD = 0
	wf = 1
        flag = 0
        maxI = 0
    }
    
    BREAKPOINT {

	SOLVE state METHOD derivimplicit : 
	: we found acceptable results with "runge" integration method
	: However, M. Hines encouraged us to use "derivimplicit" method instead - which is slightly slower than runge - 
	: to avoid probable unstability problems
        : numerical error accumalation compensation
        g = gVI * (wtau3*C + wtau2*B - A) * multiple
        : if (g<0){
        :     printf("%g\n",g)
        :     printf("\t%g,%g\n",A,B)
        :     : g = 0
        :     : A = 0
        :     : B = 0
        :     printf("\t%g,%g\n",tau1,tau2)
        : }
	iNMDA = (1e-6) * g * Mgblock(v) * (v - e)
        
    }
    
    DERIVATIVE state {
        rates(v)
    A' = -A/tau1
    B' = -B/tau2
    C' = -C/tau3
	: Voltage Dapaendent Gating of NMDA needs prior binding to Glutamate Kim11
	: gVD' = ((wtau3*C + wtau2*B)/wf)*(inf-gVD)/tau
	: gVD' = (inf-gVD)/tau Regular HH-type
 }

NET_RECEIVE(weight, D1, tsyn (ms)) {
	INITIAL {
	: these are in NET_RECEIVE to be per-stream
	: this header will appear once per stream
		D1 = 1
		tsyn = t
	}

	D1 = 1 - (1-D1)*exp(-(t - tsyn)/tau_D1)
	tsyn = t
        
	wf = synWeight*factor*D1*hillGluc(weight*1 (mM)) 
        : printf("%g,%g,%g,%g,%g\n",weight,factor,D1,wf,hillGluc(glu))
        : printf("%g\n",weight)
        : printf("%g",tsyn)
        
	A = A + wf
	B = B + wf
	C = C + wf
        : printf("%g,%g,%g\n",A,B,C)
       
	D1 = D1 * d1
        flag = 0
        wf = 1
        : printf("%g,%g,%g\n",A,B,C)
   }
    

FUNCTION Mgblock(v(mV)) {
	: from Spruston95
	Mgblock = 1 / (1 + (Mg/K0)*exp((0.001)*(z)*delta*F*(-v+shift)/R/(T+celsius)))
    }
    
    PROCEDURE rates(v (mV)) {
        : Follows mirroing aspect before and after 0 mV of Astrocyte NMDAR
	: inf = (fabs(v) - gVDv0) * gVDst * gVI 
        
	tau1 = (tau1_0 + a1*(exp(-b1*v)))*q10_tau1
        : printf("%g\n",tau1)
	tau2 = (tau2_0 + a2*(exp(b2*v)))*q10_tau2
        
        :update when enough data to get voltage dependence
	tau3 = (tau3_0 + a3*(exp(b3*v)))*q10_tau3
	if (tau1/tau2 > .9999) {
		tau1 = tau2 * .9999
	}
	if (tau2/tau3 > .9999) {
		tau2 = tau3*.9999
	    }
        }
        
    
    FUNCTION hillGluc(gluConc (mM)){
        hillGluc = 1/(1 + pow((1e-3)*gluEC/gluConc,hilln))
    }
