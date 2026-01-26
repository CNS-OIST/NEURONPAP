import string


class gl:
    unit_mV = "(mV)"
    unit_s = "(s)"
    unit_ms = "(ms)"
    unit_mM = "(mM)"
    unit_pA = "(pA)"
    unit_micron_raw = "$\mu$m"
    unit_micron = f"({unit_micron_raw})"
    # label
    ms = "Time " + unit_ms
    s = "Time" + unit_s
    volt = "Voltage " + unit_mV
    curr = "Current " + unit_pA
    d_volt = "Membrane potential change " + unit_mV
    ek_raw = "E$_\mathrm{K}$"
    ek = ek_raw + " " + unit_mV
    vm = "V$_\mathrm{m}$"
    durstim = "Stim. duration " + unit_ms
    pap_affect = "Affected PAP length " + unit_micron
    seed_num = "Seed number"
    pap_len = "PAP length " + unit_micron
    fluor = "$\Delta F/F_0$ (%)"
    volt_atten = f"{vm}/V$_0$"
    abs_distance = f"Distance {unit_micron}"
    clim_volt = (0, 20)
    lim_Vmemb = (-90, -50)
    lim_d_volt = (0, 70)
    lim_ko = (0, 30)
    lim_ek = (-100, 20)
    lim_curr = (-500, 500)
    lim_min_amp = (-1000, 100)

    @staticmethod
    def lim_zoom(initStep, dt, time_frame=20, cvode=None):
        if cvode:
            # get index of initTstop
            return (cvode, cvode + time_frame)
        else:
            return (initStep * dt, initStep * dt + time_frame)

    @staticmethod
    def current_ion(ion):
        return "I$_\mathrm{" + str(ion) + "}$"

    @staticmethod
    def ion_o(ion, short=False):
        if short:
            return f"{ion}" + "$_\mathrm{o}$ "
        else:
            return f"Extracellular [{ion}] " + gl.unit_mM

    @staticmethod
    def delta_ion_o(ion):
        return "$\Delta$" + gl.ion_o(ion)

    @staticmethod
    def chan_num(chan):
        return f"# of {chan} Channels"

    @staticmethod
    def free(label):
        # TODO: add rules to implement like sentence case
        #
        gl.correct_sentence_case(label)
        return label

    @staticmethod
    def correct_sentence_case(label):
        s = label
        if not s:
            return s

        first_alpha_index = -1
        for i, char in enumerate(s):
            # Check if the character is an ASCII letter (a-z, A-Z)
            if char in string.ascii_letters:
                first_alpha_index = i
                break

        if first_alpha_index == -1:
            return s

        prefix = s[:first_alpha_index]
        first_letter_upper = s[first_alpha_index].upper()
        suffix_lower = s[first_alpha_index + 1 :].lower()

        return prefix + first_letter_upper + suffix_lower
