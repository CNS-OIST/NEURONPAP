import numpy as np
import matplotlib.pyplot as plt
from neuron import h
from matplotlib.animation import FuncAnimation
from mpi4py import MPI
import types

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from global_labels import gl
import os


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def plot_distance_shells(ax, origin, step_num=5, alpha=0.05, color="k", resolution=30):
    center = (h.x3d(0, sec=origin), h.y3d(0, sec=origin), h.z3d(0, sec=origin))
    xlist, ylist, zlist = get_all_coords()
    all_x = np.concatenate(xlist)
    all_y = np.concatenate(ylist)
    all_z = np.concatenate(zlist)

    coords = np.column_stack((all_x, all_y, all_z))
    dists = np.linalg.norm(coords - np.array(center), axis=1)
    max_radius = np.ceil(dists.max())
    step = max_radius / step_num
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    for r in np.arange(step, max_radius + step, step):
        x = center[0] + r * np.outer(np.cos(u), np.sin(v))
        y = center[1] + r * np.outer(np.sin(u), np.sin(v))
        z = center[2] + r * np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=False)


def get_all_coords():
    xlist = []
    ylist = []
    zlist = []

    # Extract morphology + RANGEVAR
    for sec in h.allsec():
        n3d = int(h.n3d(sec=sec))
        xs = [h.x3d(i, sec=sec) for i in range(n3d)]
        ys = [h.y3d(i, sec=sec) for i in range(n3d)]
        zs = [h.z3d(i, sec=sec) for i in range(n3d)]

        xlist.append(np.array(xs))
        ylist.append(np.array(ys))
        zlist.append(np.array(zs))

    return xlist, ylist, zlist


def animate_morphology(
    tstop=100,
    dt=1,
    rangevar="v",
    colormap_name="magma",
    outfile="morphology_animation.mp4",
    frame_num=None,
    zoom=None,
    clim=None,
    no_advance=False,
):
    if rank != 0:
        return
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    h.tstop = tstop

    # override calc frames when frame num set
    if frame_num:
        dt = tstop / frame_num

    frames = int(tstop / dt)
    if dt < h.dt:
        h.dt = dt

    if zoom and "zoom" not in outfile:
        fPath = outfile.split(".mp4")[0]
        fPath += "_zoom.mp4"
        outfile = fPath

    plot_3d_morphology(
        rangevar=rangevar,
        colormap_name=colormap_name,
        fig=fig,
        ax=ax,
        add_colorbar=True,
        zoom=zoom,
        clim=clim,
    )

    def update(frame_t):
        ax.cla()  # fully clear axes
        print(f"rendering {frame_t}/{frames}")

        # draw without creating new colorbars
        plot_3d_morphology(
            rangevar=rangevar,
            colormap_name=colormap_name,
            fig=fig,
            ax=ax,
            add_colorbar=False,
            zoom=zoom,
            clim=clim,
        )

        if not no_advance:
            h.continuerun(h.t + dt)

    anim = FuncAnimation(fig, update, frames=frames, interval=1, repeat=False)
    real_fps = 2 / dt  # 2 ms in simulation per 1 second video

    anim.save(outfile, fps=real_fps)
    print(f"Saved animation → {outfile}")


def plot_3d_morphology(
    rangevar="v",
    colormap_name="magma",
    fig=None,
    ax=None,
    add_colorbar=True,
    add_shell=None,
    add_null=False,
    show=False,
    zoom=None,
    clim=None,
):
    """
    Plot NEURON morphology in 3D with diameter scaling and a RANGE variable as color.
    """
    if rank != 0:
        return

    # Get colormap
    if type(colormap_name) is list:
        cmap = mcolors.ListedColormap(colormap_name)
    else:
        cmap = plt.colormaps[colormap_name]

    # Prepare new fig/ax if none provided
    if fig is None or ax is None:
        plt.cla()
        plt.clf()
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d", computed_zorder=not add_shell)

    if add_shell:
        plot_distance_shells(ax, h.soma, step_num=add_shell)

    dlist = []
    varlist = []

    # Extract morphology + RANGEVAR
    for sec in h.allsec():
        n3d = int(h.n3d(sec=sec))
        ds = [h.diam3d(i, sec=sec) for i in range(n3d)]

        # color uses section rangevar (segment midpoint)
        if type(rangevar) == types.FunctionType:
            rv = rangevar(sec(0.5))
        else:
            try:
                rv = getattr(sec(0.5), rangevar)
            except:
                rv = np.nan

        dlist.append(np.array(ds))
        varlist.append(rv)

    if type(colormap_name) is list:
        if clim is None:
            min_clim, max_clim = min(varlist), max(varlist)
        else:
            min_clim, max_clim = clim
        norm = mcolors.Normalize(vmin=min_clim, vmax=max_clim)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap)

    if clim:
        vmin, vmax = clim
    else:
        # Normalize RANGEVAR to colormap
        vmin = min(varlist)
        vmax = max(varlist)
    sm.set_clim(vmin=vmin, vmax=vmax)

    if zoom:
        size = 1
        # get average center of PAP
        count = 0
        x = 0
        y = 0
        z = 0
        for sec in zoom:
            x += h.x3d(0.5, sec=sec)
            y += h.y3d(0.5, sec=sec)
            z += h.z3d(0.5, sec=sec)
            count += 1

        mid_x = x / count
        mid_y = y / count
        mid_z = z / count
        max_range = size
    else:
        xlist, ylist, zlist = get_all_coords()
        all_x = np.concatenate(xlist)
        all_y = np.concatenate(ylist)
        all_z = np.concatenate(zlist)

        xmin, xmax = all_x.min(), all_x.max()
        ymin, ymax = all_y.min(), all_y.max()
        zmin, zmax = all_z.min(), all_z.max()

        max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
        mid_x = (xmax + xmin) / 2
        mid_y = (ymax + ymin) / 2
        mid_z = (zmax + zmin) / 2

    if add_shell:
        max_range *= 1.2

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    # Plot each section with diameter scaling and color mapping
    for xs, ys, zs, ds, rv in zip(xlist, ylist, zlist, dlist, varlist):
        color = sm.to_rgba(rv)
        if add_null:
            color = color[:-1] + (1,)
        ax.plot(xs, ys, zs, color=color, linewidth=np.mean(ds) / 2)

    # Integer ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.zaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # === Colorbar (optional) ===
    if add_colorbar:
        cbar = plt.colorbar(sm, ax=ax)
        if rangevar == "v":
            cbar.set_label(gl.volt)
        elif type(rangevar) == types.FunctionType:
            cbar.set_label(rangevar.__name__)

    ax.set_xlabel(gl.free("x ") + gl.unit_micron)
    ax.set_ylabel(gl.free("y ") + gl.unit_micron)
    ax.set_zlabel(gl.free("z ") + gl.unit_micron)
    if zoom:
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    plt.tight_layout()
    if show:
        plt.show()


def plot_combined(
    rangevar,
    origin,
    paths_away,
    path_toward,
    fName=None,
    precomputed_toward=None,
    precomputed_away=None,
):
    if not fName:
        fName = f"combined_{rangevar}_{origin=}"
    plt.cla()
    plt.clf()
    plt.figure(figsize=(9, 5))
    xmin = np.inf
    xmax = -np.inf
    if not precomputed_toward and not precomputed_away:
        precomputed_toward = []
        precomputed_away = []
        total_dict = {}
        tmp_dict = convert_list_section_to_python(paths_away)

        total_dict.update(tmp_dict)
        tmp_dict = convert_list_section_to_python(path_toward)
        total_dict.update(tmp_dict)
        for key, sec_list in total_dict.items():
            dist, var = convert_sec_list_to_var_distance(rangevar, origin, sec_list)
            if "to_soma" in key:
                precomputed_toward.append((dist, var))
            else:
                precomputed_away.append((dist, var))
    for dist, var in precomputed_toward:
        plt.plot(dist, var, color="black")
    for dist, var in precomputed_away:
        plt.plot(-1 * np.array(dist), var, color="black")
        if len(dist) > 0:
            if xmin > np.min(-1 * np.array(dist)):
                xmin = np.min(-1 * np.array(dist))
            if xmax < np.max(-1 * np.array(dist)):
                xmax = np.max(-1 * np.array(dist))

    plt.axvline(x=0, ymin=0, ymax=1, color="lightgrey", linestyle="--")
    if rangevar == "v":
        plt.axhline(
            y=1 / np.e, xmin=0, xmax=1, label="1/e", color="grey", linestyle="--"
        )

    plt.xlabel(gl.free("Distance ") + gl.unit_micron)
    plt.ylabel(gl.volt)
    plt.xlim((1.1 * xmin, 1.1 * xmax))
    plt.legend()
    plt.savefig(os.path.join("../morphResults", f"{fName}.pdf"))
    return precomputed_toward, precomputed_away


def plot_paths(rangevar, origin, list_section, fname="", precomputed=None):
    plt.cla()
    plt.clf()
    plt.figure(figsize=(9, 5))
    if not precomputed:
        precomputed = []
        section_dict = convert_list_section_to_python(list_section)
        for key, sec_list in section_dict.items():
            print(key)
            dist, var = convert_sec_list_to_var_distance(rangevar, origin, sec_list)
            precomputed.append((dist, var))
    for dist, var in precomputed:
        plt.plot(dist, var, color="black")

    # plt.axvline(x=0, ymin=0, ymax=1, color="lightgrey", linestyle="--")
    plt.axvline(x=3.6, ymin=0, ymax=1, color="deepskyblue", linestyle="--")
    if rangevar == "v":
        plt.axhline(
            y=1 / np.e, xmin=0, xmax=1, label="1/e", color="grey", linestyle="--"
        )

    plt.legend()
    plt.ylabel(gl.volt_atten)
    plt.xlabel(gl.abs_distance)

    plt.savefig(os.path.join("../morphResults", f"{fname}_{rangevar}.pdf"))
    return precomputed


def convert_sec_list_to_var_distance(var, origin, sec_list, normalize=True):
    varList = []
    distanceList = []
    for i, sec in enumerate(sec_list):
        if i == 0:
            if normalize:
                norm = getattr(origin(0.5), var)
            else:
                norm = 1
            h.distance(sec=h.soma)
        distanceList.append(h.distance(0.5, sec=sec) - h.distance(0.5, sec=origin))
        try:
            rv = (getattr(sec(0.5), var) + 85) / (norm + 85)
        except:
            rv = 0.0
        varList.append(rv)

    return distanceList, varList


def convert_list_section_to_python(list_section):
    tmp_section = {}
    tmp_section["to_soma"] = []
    for i, obj in enumerate(list_section):
        try:
            iter(obj)
        except TypeError:
            tmp_section["to_soma"].append(obj.sec)
        else:
            tmp_section[f"to_leaf{i}"] = []
            for sr in obj:
                tmp_section[f"to_leaf{i}"].append(sr.sec)

    return tmp_section


if __name__ == "__main__":
    # Load NEURON run-time and your geometry file
    h.load_file("stdrun.hoc")
    h.load_file("GeometryAstrocyteCA1.hoc")

    # === USAGE EXAMPLE ===
    plot_3d_morphology()
