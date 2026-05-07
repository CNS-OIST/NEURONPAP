import string


class gl:
    unit_mV = "(mV)"
    unit_s = "(s)"
    unit_ms = "(ms)"
    unit_liter = "(L)"
    mM_raw = "mM"
    unit_mM = f"({mM_raw})"
    unit_pA = "(pA)"
    unit_micron_raw = "$\mu$m"
    unit_micron = f"({unit_micron_raw})"
    unit_micron_bold = r'($\mathbf{\mu}$m)'
    unit_curr_density_raw = "mA/cm$^2$"
    unit_curr_density = f"({unit_curr_density_raw})"
    unit_hz_raw = "Hz"
    unit_hz = f"({unit_hz_raw})"
    unit_um_cubed_raw = f"{unit_micron_raw}$^3$"
    unit_um_cubed = f"({unit_um_cubed_raw})"
    unit_um_squared_raw = f"{unit_micron_raw}$^2$"
    unit_um_squared = f"({unit_um_squared_raw})"
    hz = "Frequency " + unit_hz
    ri = r"Input Resistance (M$\Omega$)"
    # label
    ms = "Time " + unit_ms
    s = "Time" + unit_s
    volt = "Voltage " + unit_mV
    curr = "Current " + unit_pA
    d_volt = "Membrane potential change " + unit_mV
    ek_raw = "E$_\mathrm{K}$"
    ek = ek_raw + " " + unit_mV
    vm = "V$_\mathrm{m}$" 
    d_volt_short = "$\Delta$" + vm + " " +unit_mV
    durstim = "Stim. duration " + unit_ms
    pap_affect = "Affected PAP length " + unit_micron
    seed_num = "Seed number"
    pap_len = "PAP length " + unit_micron
    fluor = "$\Delta F/F_0$ (%)"
    volt_atten = f"{vm}/V$_0$"
    lim_cvk_volt = (-90, -40)
    lim_cvk_ko = (0, 10)
    abs_distance = f"Distance {unit_micron}"
    max_ko = 80
    lim_log_ko = (3, max_ko)
    clim_volt = (0, 20)
    lim_Vmemb = (-90, -50)
    lim_VmembSoma = (-90, -80)
    lim_d_volt = (0, 80)
    lim_ko = (0, 30)
    lim_ek = (-100, 20)
    lim_ek_zoom = (-100, -25)
    lim_curr = (-500, 500)
    lim_curr_inset = (-0.11, 0.21)
    lim_min_amp = (-50, 1)
    lim_min_amp_bs = (-100, 100)
    shell_num = "shell number"
    figsize_full = (8, 11.5)
    figsize_halfw = figsize_full[0] / 2, figsize_full[1]
    figsize_halfh = figsize_full[0], figsize_full[1] / 2
    figsize_panel = figsize_full[0] / 2, figsize_full[1] / 3
    figsize_panel_long = figsize_full[0], figsize_full[1] / 3
    figsize_ikPlots = figsize_full[0] / 3, figsize_full[1] / 3
    figsize_distCurr = figsize_full[0] * 2 / 3, figsize_full[1] * 2 / 3
    figsize_distCurr_panel = figsize_full[0] / 3, figsize_full[1] / 3
    clampI = "$I_{clamp}$"
    sigma_glt = "$\Sigma\ I_{GLT}$"
    font = {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.labelsize": 10,
        "text.latex.preamble": r"\usepackage{xcolor}",
    }

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
    def ion_o(ion, short=True,unit=True):
        if ion == "K":
            valence = "$^+$"
        else:
            valence = ""
        if short:
            if unit:
                return f"[{ion}{valence}]" + "$_\mathrm{o}$" + " " + gl.unit_mM
            else:
                return f"[{ion}{valence}]" + "$_\mathrm{o}$"
        else:
            return f"Extracellular [{ion}{valence}] " + gl.unit_mM

    @staticmethod
    def delta_ion_o(ion, short=False):
        return "$\Delta$" + gl.ion_o(ion, short=short)

    @staticmethod
    def chan_num(chan):
        return f"# of {chan} Channels"

    @staticmethod
    def density_num(chan):
        return f"Current density of {chan} Channels {gl.unit_curr_density}"

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
