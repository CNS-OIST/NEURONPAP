from scipy.special import expi
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from global_labels import gl
import os

plt.rcParams.update(gl.font)
saveDir = os.path.abspath("../morphResults")
plt.rcParams["savefig.directory"] = saveDir
plt.ioff()


def inverse_exponential_integral_equation(x, target_y):
    """
    Function to find the root for Ei(x) - target_y = 0.
    """
    return expi(x) - target_y


def halt_at_zero(t, y):
    return y[0]


def test_get_inverse_expi(target_y_value):
    # Example usage:
    initial_guess = 1.0  # An initial guess for x

    # Find the root (the inverse value of x)
    inverse_x = fsolve(
        inverse_exponential_integral_equation, initial_guess, args=(target_y_value,)
    )

    print(f"For Ei(x) = {target_y_value}, the approximate inverse x is: {inverse_x[0]}")
    print(f"Checking: Ei({inverse_x[0]}) = {expi(inverse_x[0])}")


class analytical_ko:
    V = 1e-15  # um3 in L
    VClamp = -80  # mV
    g = 50e-12  # S
    z = 1
    F = 96485  # C/m2
    R = 8.314  # J/K/mol
    T = 307  # K
    NA = 6.02e23
    ki = 140e-3  # mM
    ko0 = 3  # mM
    vhalf = -98.92  # mV
    kl = 10.89  # mV
    q10 = 1
    g_scale = 0.09534626

    def __init__(self, **kwargs):
        for key in kwargs.keys():
            if hasattr(self, key):
                setattr(self, key, kwargs[key])

        self.g *= kwargs.pop("multiple", 1) * self.g_scale
        self.g *= self.lV(self.VClamp)
        print(f"VC is {self.VClamp} mV")

    def lV(self, V):
        return 1 / (1 + np.exp((V - self.vhalf) / self.kl))

    def init_Ko(self, mM):
        self.get_ALPH_vals()
        return expi(
            0.5 * (np.log(self.V * mM * 1e-3 / self.gamma) - self.alpha / self.beta)
        )

    def calc_analytical(self, inverse_x):
        self.get_ALPH_vals()
        return self.gamma * np.exp(2 * inverse_x + self.alpha / self.beta)

    def get_ALPH_vals(self):
        alph = self.g / self.z / self.F / np.sqrt(self.V)
        self.A = self.VClamp * 1e-3 * alph
        self.B = alph * self.R * self.T / self.z / self.F
        self.C = self.V
        self.D = self.ki
        self.alpha = self.A
        self.beta = self.B
        self.gamma = self.C * self.D

    def calcNernst(self, ko):
        return self.R * self.T / self.z / self.F * np.log(ko / self.V / self.ki)

    def calcInvNernst(self, E):
        return np.exp(self.F * self.z / self.R / self.T * E * 1e-3) * self.V * self.ki

    def phase_plot(self, *xrange):
        x = np.linspace(*xrange, 100)
        plt.figure(5)
        plt.plot(x, self.dxdt(x), label=f"{self.V*1e15:.1f} um3", color="black")
        plt.figure(6)
        plt.plot(
            x / self.V * 1e3,
            self.dxdt(x) / self.V * 1e3,
            label=f"{self.V*1e15:.1f} um3",
            color="black",
        )

    def dxdt(self, x):
        self.get_ALPH_vals()
        try:
            tmp = np.sqrt(x) * (self.A - self.B * np.log(x / (self.C * self.D)))
        except RuntimeWarning:
            return 0

        return tmp

    def ode_system(self, t, y):
        x = y[0]
        return [self.dxdt(x)]

    def NI_Ko(self, endt, dt):
        self.get_ALPH_vals()
        x0 = [self.ko0 * self.V * 1e-3]
        t_span = (0, endt)
        t_eval = np.linspace(t_span[0], t_span[1], int(endt / dt))

        halt_at_zero.terminal = True
        halt_at_zero.direction = -1
        sol = solve_ivp(
            fun=self.ode_system,
            t_span=t_span,
            y0=x0,
            t_eval=t_eval,
            events=halt_at_zero,
        )

        if sol.success:
            plt.figure(3)
            plt.plot(
                sol.t,
                sol.y[0] / self.V * 1e3,
                label=f"{self.V * 1e15:.1f} {gl.unit_um_cubed_raw}",
            )
            plt.figure(4)
            plt.plot(
                sol.t,
                sol.y[0] * self.NA,
                label=f"{self.V * 1e15:.1f} {gl.unit_um_cubed_raw}",
            )
        else:
            print("ODEの解に失敗しました:", sol.message)

    def Ko(self, t):
        self.get_ALPH_vals()
        initial_guess = 1e-17
        target_y_value = self.init_Ko(self.ko0) - self.beta * t / (
            np.sqrt(self.gamma) * np.exp(self.alpha / 2 / self.beta)
        )
        if type(t) is int or type(t) is float:
            inverse_x = fsolve(
                inverse_exponential_integral_equation,
                initial_guess,
                args=(target_y_value,),
            )
            print(
                f"E_k = {self.calcNernst(self.calc_analytical(inverse_x[0])) * 1e3} mV"
            )

            return self.calc_analytical(inverse_x[0])
        else:
            tmp = []
            for time_point in target_y_value:
                # test_get_inverse_expi(time_point)
                inverse_x = fsolve(
                    inverse_exponential_integral_equation,
                    initial_guess,
                    args=(time_point,),
                )
                # print(
                #    f"E_k = {self.calcNernst(self.calc_analytical(inverse_x[0])) * 1e3} mV"
                # )

                tmp.append(self.calc_analytical(inverse_x[0]))

            return tmp


def t_half(clamp, V_ECS):
    analytical_trace = analytical_ko(ko0=20, multiple=5e2, VClamp=clamp)
    g = analytical_trace.g
    F = analytical_trace.F
    R = analytical_trace.R
    T = analytical_trace.T
    ki = analytical_trace.ki
    ko0 = analytical_trace.ko0
    VClamp = analytical_trace.VClamp
    a = g / F / np.sqrt(V_ECS)
    alpha = VClamp * 1e-3 * a
    beta = a * R * T / F
    gamma = V_ECS * ki
    C = analytical_trace.init_Ko(ko0)
    return (
        -np.sqrt(gamma)
        * np.exp(alpha / 2 / beta)
        / beta
        * (expi(0.5 * np.log(V_ECS * ko0 * 1e-3 / 2 / gamma) - alpha / 2 / beta) - C)
    )


def plot_All(VClamp):
    analytical_trace = analytical_ko(ko0=20, multiple=5e2, VClamp=VClamp)
    print(f"{analytical_trace.Ko(0) / analytical_trace.V * 1e3} mM for t0")
    endt = 5  # s
    dt = 1e-4

    order = -3
    VCSList = np.logspace(-1, order, 10)
    VCSList *= 1e-15  # um3 to L
    exp_V = 1.53e-18
    VCSList = np.flip(np.sort(np.append(VCSList, exp_V)))

    time_to_0 = []
    for V in VCSList:
        t = np.linspace(0, endt, int(endt / dt))
        analytical_trace.V = V
        trace = analytical_trace.Ko(t)
        t *= 1e3
        plt.figure(0, figsize=gl.figsize_panel)
        plt.plot(
            t,
            trace / analytical_trace.V * 1e3,
            label=f"{analytical_trace.V*1e15:.1e} {gl.unit_um_cubed_raw}",
            color=gl.sim_others if analytical_trace.V != exp_V else gl.sim_main,
            lw=0.5 if analytical_trace.V != exp_V else 2,
        )
        plt.figure(1, figsize=gl.figsize_panel)
        plt.plot(
            t,
            np.array(trace) * analytical_trace.NA,
            label=f"{analytical_trace.V*1e15:.3f} {gl.unit_um_cubed_raw}",
        )
        trace_len = [t for t in trace if t > 0]
        time_to_0.append(len(trace_len) * dt)
        analytical_trace.NI_Ko(endt, dt)
        analytical_trace.phase_plot(
            0, analytical_trace.calcInvNernst(analytical_trace.VClamp) * 2
        )

    plt.figure(0)
    plt.axhline(
        y=analytical_trace.calcInvNernst(analytical_trace.VClamp)
        / analytical_trace.V
        * 1e3,
        color="red",
        linestyle="--",
    )
    plt.xlabel(gl.ms)
    plt.ylabel(gl.ion_o("K"))
    plt.legend()
    plt.xlim((0, 500))
    plt.ylim(bottom=5)
    plt.savefig(os.path.join("../morphResults", f"trace_mol{VClamp}.pdf"))

    plt.figure(1)
    plt.legend()
    plt.xlabel(gl.ms)
    plt.ylabel(gl.ion_o("K"))
    plt.ylim(bottom=0)
    plt.savefig(os.path.join("../morphResults", f"ode{VClamp}.pdf"))

    plt.figure(2, figsize=gl.figsize_panel)
    plt.scatter(VCSList, np.array(time_to_0) * 1e3)
    plt.xlabel("Volume (L)")
    plt.ylabel("Time to 0 (ms)")
    # plt.savefig(f"time_to_0_{VClamp}.pdf")
    plt.figure(3, figsize=gl.figsize_panel)
    plt.xlabel("Time (t)")
    plt.ylabel("x(t) (mM)")
    plt.legend()
    plt.axhline(
        y=analytical_trace.calcInvNernst(analytical_trace.VClamp)
        / analytical_trace.V
        * 1e3,
        color="red",
    )
    plt.ylim(bottom=0)
    # plt.savefig(f"ode{VClamp}.pdf")
    plt.figure(4, figsize=gl.figsize_panel)
    plt.xlabel("Time (t)")
    plt.ylabel("x(t)")
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig(os.path.join("../morphResults", f"ode_count{VClamp}.pdf"))

    plt.figure(5, figsize=gl.figsize_panel)
    plt.legend()
    plt.xlabel("x (mol)")
    plt.ylabel("dxdt")
    # plt.savefig(f"phase_plot{VClamp}.pdf")
    plt.figure(6, figsize=gl.figsize_panel)
    plt.legend()
    plt.xlabel("x (mM)")
    plt.ylabel("dxdt")
    plt.savefig(os.path.join("../morphResults", f"phase_plot_mM_{VClamp}.pdf"))
    plt.figure(7, figsize=gl.figsize_panel)
    x = np.linspace(10**order, 5 * 10**order, 100)
    x *= 1e-15
    plt.plot(x, t_half(VClamp, x) * 1e3, color=gl.sim_others)
    plt.scatter(
        exp_V, t_half(VClamp, exp_V) * 1e3, label="model", color=gl.sim_main, zorder=3
    )
    plt.xlabel(r"V$_{ECS}$ " + gl.unit_liter)
    plt.ylabel(r"t$_{1/2}$ " + gl.unit_ms)
    plt.ylim((0, 25))
    plt.legend()
    plt.savefig(os.path.join("../morphResults", f"t_half_{VClamp}.pdf"))


if __name__ == "__main__":
    for v in [-80]:
        plt.cla()
        plt.clf()
        plot_All(v)
