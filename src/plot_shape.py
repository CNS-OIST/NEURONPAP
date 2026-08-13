import numpy as np
import matplotlib.pyplot as plt
from neuron import h
from matplotlib.animation import FuncAnimation
from mpi4py import MPI
import types

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from global_labels import gl
import os

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def plot_distance_shells(ax, origin, step_num=5, alpha=0.05, color="k", resolution=30):
    """
    Draw concentric semi-transparent spherical shells centered on origin, spaced
    evenly out to the farthest morphology point, to visualize distance bins.
    """
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


def split_into_n_equal_parts(lst, n=5):
    """
    Split a list into n parts of near-equal size, with each part (except the
    last) sharing its final point with the first point of the next part.
    """
    k, m = divmod(len(lst), n)
    parts = [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]
    for i in range(len(parts) - 1):
        if parts[i + 1]:
            parts[i].append(parts[i + 1][0])

    return parts


def get_all_coords(rangesec_name=None, rangesec_step=None):
    """
    Collect 3D pt3d coordinates for every section in the model. If
    rangesec_name/rangesec_step are given, that section's points are instead
    subsampled at rangesec_step intervals and split into equal sub-segments,
    each appended as its own entry. Returns per-section x/y/z coordinate arrays.
    """
    xlist = []
    ylist = []
    zlist = []

    for sec in h.allsec():
        n3d = int(h.n3d(sec=sec))
        if (
            rangesec_name is not None
            and str(sec) == rangesec_name
            and rangesec_step is not None
        ):
            xs = [h.x3d(i, sec=sec) for i in np.arange(0, n3d, rangesec_step * n3d)]
            ys = [h.y3d(i, sec=sec) for i in np.arange(0, n3d, rangesec_step * n3d)]
            zs = [h.z3d(i, sec=sec) for i in np.arange(0, n3d, rangesec_step * n3d)]

            # adjust endpoint
            xs[-1] = h.x3d(n3d - 1, sec=sec)
            ys[-1] = h.y3d(n3d - 1, sec=sec)
            zs[-1] = h.z3d(n3d - 1, sec=sec)

            num_secs = int(1 / rangesec_step)

            for l, coord in [(xlist, xs), (ylist, ys), (zlist, zs)]:
                for tmp in split_into_n_equal_parts(coord, num_secs):
                    l.append(np.array(tmp))

        else:
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
    """
    Run and save an MP4 animation of the 3D morphology plot over the course of
    a NEURON simulation, redrawing the shape at each timestep and advancing the
    simulation (unless no_advance is set). No-op on any rank other than 0.
    """
    if rank != 0:
        return
    fig = plt.figure(figsize=gl.figsize_distCurr_panel)
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
        """
        Clear the axes, redraw the 3D morphology for the current frame (without
        adding a new colorbar), and advance the simulation by dt unless
        no_advance is set.
        """
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
    color_names=None,
    rangesec=None,
    norm=None,
):
    """
    Plot NEURON morphology in 3D with diameter scaling and a RANGE variable as color.
    """
    if rank != 0:
        return
    if rangesec is not None:
        rangesec_name, rangesec_step = rangesec
    else:
        rangesec_name = None
        rangesec_step = None

    # Get colormap
    if type(colormap_name) is list:
        cmap = mcolors.ListedColormap(colormap_name)
    else:
        cmap = plt.colormaps[colormap_name]

    # Prepare new fig/ax if none provided
    if fig is None and ax is None:
        plt.cla()
        plt.clf()
        plt.rcParams.update(gl.font)
        fig = plt.figure(figsize=gl.figsize_distCurr_panel, constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d", computed_zorder=not add_shell)

    if add_shell:
        plot_distance_shells(ax, h.soma, step_num=add_shell)

    dlist = []
    varlist = []

    # Extract morphology + RANGEVAR
    for sec in h.allsec():
        n3d = int(h.n3d(sec=sec))
        if n3d == 0:
            continue
        ds = [h.diam3d(i, sec=sec) for i in range(n3d)]

        # color uses section rangevar (segment midpoint)
        if rangesec_name is not None and str(sec) == rangesec_name:
            ds = [h.diam3d(i, sec=sec) for i in np.arange(0, n3d, rangesec_step * n3d)]

        if type(rangevar) == types.FunctionType:
            rv = rangevar(sec(0.5))
        else:
            try:
                rv = getattr(sec(0.5), rangevar)
            except AttributeError:
                if color_names is not None:
                    if type(color_names) is list and str(sec) in color_names:
                        if "soma" in str(sec):
                            rv = 2
                        elif "dendrite" in str(sec):
                            rv = 1
                        else:
                            rv = 3
                    elif type(color_names) is dict:
                        if rangesec_name == str(sec):
                            totalL = sec.L
                            rv = [
                                totalL * ratio
                                for ratio in np.arange(
                                    rangesec_step, 1 + rangesec_step / 2, rangesec_step
                                )
                            ]
                        elif str(sec) in color_names.keys():
                            rv = color_names[str(sec)]
                        else:
                            rv = np.nan
                    else:
                        rv = 0
                else:
                    rv = np.nan

        if type(rv) is list:
            varlist += rv
            print(varlist)
        else:
            varlist.append(rv)

        if rangesec_name is not None and rangesec_name == str(sec):
            num_secs = int(1 / rangesec_step)

            for d in split_into_n_equal_parts(ds, num_secs):
                dlist.append(np.array(d))

        else:
            dlist.append(np.array(ds))

    if type(colormap_name) is list:
        if norm is None:
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
        xlist, ylist, zlist = get_all_coords(
            rangesec_name=rangesec_name, rangesec_step=rangesec_step
        )
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
        if norm is not None:
            color = cmap(norm(rv))
        else:
            color = sm.to_rgba(rv)
        if color_names is not None and type(color_names) is not dict:
            if rv > 0.5:
                if rv > 2.5:
                    ds = 5
                zorder = 3
            else:
                zorder = 0

        elif type(color_names) is dict:
            if not np.isnan(rv):
                zorder = 3
            else:
                zorder = None
        else:
            zorder = None

        if add_null and np.isnan(rv):
            color = (0.8, 0.8, 0.8, 1.0)

        ax.plot(
            xs,
            ys,
            zs,
            axlim_clip=True,
            color=color,
            linewidth=np.mean(ds) / 2,
            zorder=zorder,
        )

    # Integer ticks
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # ax.zaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis="x", pad=-5)
    ax.tick_params(axis="y", pad=-5)
    ax.tick_params(axis="z", pad=-2)
    ax.set_xlabel(gl.free("x ") + gl.unit_micron, labelpad=-5)
    ax.set_ylabel(gl.free("y ") + gl.unit_micron, labelpad=-5)
    ax.set_zlabel(gl.free("z ") + gl.unit_micron, labelpad=-7)
    if zoom:
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    plt.tight_layout()
    fig.subplots_adjust(left=0, right=0.85, bottom=0.1, top=1)
    # === Colorbar (optional) ===
    if add_colorbar:
        cbaxes = inset_axes(
            ax,
            width="60%",  # Width: 60% of the parent axes
            height="5%",  # Height: 5% of the parent axes
            loc="lower center",  # Center the new axes at the bottom
            bbox_to_anchor=(0.0, -0.15, 1, 1),  # Position relative to the parent axes
            bbox_transform=ax.transAxes,  # Use axes coordinates for bbox_to_anchor
            borderpad=0,
        )
        cbar = plt.colorbar(sm, cax=cbaxes, orientation="horizontal")
        if rangevar == "v":
            cbar.set_label(gl.volt)
        elif rangevar == "num_shell":
            cbar.set_label(gl.shell_num, labelpad=-2)
        elif type(rangevar) == types.FunctionType:
            cbar.set_label(rangevar.__name__)
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
    """
    Plot a range variable against distance along paths toward the soma and
    paths away from origin on one axis (away-paths use negated distance so
    both directions meet at origin). Computes distance/variable data via
    convert_sec_list_to_var_distance if not already precomputed, saves the
    figure as a PDF, and returns the computed data for reuse.
    """
    if not fName:
        fName = f"combined_{rangevar}_{origin=}"
    plt.cla()
    plt.clf()
    plt.figure(figsize=gl.figsize_panel)
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
    """
    Plot a range variable against distance from origin for each path in
    list_section as thin, semi-transparent lines. Computes the distance/
    variable data via convert_sec_list_to_var_distance if not already
    precomputed, saves the figure as a PDF, and returns the computed data.
    """
    plt.cla()
    plt.clf()
    plt.figure(figsize=gl.figsize_panel)
    if not precomputed:
        precomputed = []
        section_dict = convert_list_section_to_python(list_section)
        for key, sec_list in section_dict.items():
            dist, var = convert_sec_list_to_var_distance(
                rangevar, origin, sec_list, RMP=None
            )
            precomputed.append((dist, var))
    for dist, var in precomputed:
        plt.plot(dist, var, color="black", lw=0.3, alpha=0.5)

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


def convert_sec_list_to_var_distance(var, origin, sec_list, normalize=True, RMP=-85):
    """
    Compute, for each section in sec_list, its distance from origin and its
    range variable value normalized between RMP and the origin's own value
    (or unnormalized if normalize is False). If RMP is None, it is instead
    taken as the minimum membrane voltage found across sec_list.
    """
    varList = []
    distanceList = []
    min = 0
    if RMP is None:
        for i, sec in enumerate(sec_list):
            if getattr(sec(0.5), "v") < min:
                min = getattr(sec(0.5), "v")
        RMP = min
        print(RMP)

    for i, sec in enumerate(sec_list):
        if i == 0:
            if normalize:
                norm = getattr(origin(0.5), var)
            else:
                norm = 1
            h.distance(sec=h.soma)
        distanceList.append(h.distance(0.5, sec=sec) - h.distance(0.5, sec=origin))
        try:
            rv = (getattr(sec(0.5), var) - RMP) / (norm - RMP)
        except:
            rv = 0.0
        varList.append(rv)

    return distanceList, varList


def convert_list_section_to_python(list_section):
    """
    Convert a list of section refs / iterables of section refs into a plain
    dict of Python section lists: non-iterable entries are grouped under
    "to_soma", while iterable entries become their own "to_leafN" list.
    """
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


class NeuronMorphologyVisualizer:
    def __init__(self, saveDir="."):
        """
        Initialize empty section/children maps and create saveDir for output plots.
        """
        self.sections = {}
        self.children = {}
        self.saveDir = saveDir
        os.makedirs(self.saveDir, exist_ok=True)

    def load_morphology(self, filepath):
        """
        Load a NEURON .hoc morphology file and extract its section topology.
        """
        if filepath.endswith(".hoc"):
            h.load_file(filepath)
        else:
            raise ValueError("Only .hoc files supported")

        self._extract_topology()

    def _extract_topology(self):
        """
        Build self.sections (name -> length, average diameter, parent name)
        and self.children (parent name -> list of child names) by iterating
        over every loaded NEURON section.
        """
        self.sections = {}
        self.children = {}

        for sec in h.allsec():
            name = sec.name()
            L = sec.L
            diam = np.mean([seg.diam for seg in sec])

            sr = h.SectionRef(sec=sec)
            parent = sr.parent.name() if sr.has_parent() else None

            self.sections[name] = {"L": L, "diam": diam, "parent": parent}

            if parent:
                self.children.setdefault(parent, []).append(name)

    def _cylinder(self, start, direction, length, radius, n=30):
        """
        Build the 3D surface mesh (lateral wall plus bottom and top end caps)
        of a cylinder of given length and radius, starting at start and
        extending along direction. Returns the (X,Y,Z) mesh tuples for the
        wall, bottom cap, and top cap, plus the computed end point.
        """
        direction = direction / np.linalg.norm(direction)

        not_v = np.array([1, 0, 0]) if abs(direction[0]) < 0.9 else np.array([0, 1, 0])
        n1 = np.cross(direction, not_v)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(direction, n1)

        t = np.linspace(0, length, 2)
        theta = np.linspace(0, 2 * np.pi, n)
        t, theta = np.meshgrid(t, theta)

        X = (
            start[0]
            + direction[0] * t
            + radius * np.cos(theta) * n1[0]
            + radius * np.sin(theta) * n2[0]
        )
        Y = (
            start[1]
            + direction[1] * t
            + radius * np.cos(theta) * n1[1]
            + radius * np.sin(theta) * n2[1]
        )
        Z = (
            start[2]
            + direction[2] * t
            + radius * np.cos(theta) * n1[2]
            + radius * np.sin(theta) * n2[2]
        )

        end = start + direction * length

        r = np.linspace(0, radius, n)
        theta2 = np.linspace(0, 2 * np.pi, n)
        r, theta2 = np.meshgrid(r, theta2)

        Xb = start[0] + r * np.cos(theta2) * n1[0] + r * np.sin(theta2) * n2[0]
        Yb = start[1] + r * np.cos(theta2) * n1[1] + r * np.sin(theta2) * n2[1]
        Zb = start[2] + r * np.cos(theta2) * n1[2] + r * np.sin(theta2) * n2[2]

        Xt = end[0] + r * np.cos(theta2) * n1[0] + r * np.sin(theta2) * n2[0]
        Yt = end[1] + r * np.cos(theta2) * n1[1] + r * np.sin(theta2) * n2[1]
        Zt = end[2] + r * np.cos(theta2) * n1[2] + r * np.sin(theta2) * n2[2]

        return (X, Y, Z), (Xb, Yb, Zb), (Xt, Yt, Zt), end

    def plot_local(self, section_name, tilt_angle=np.pi / 6):
        """
        Plot a schematic 3D view of one section together with its parent and
        child sections as cylinders, each scaled relative to section_name's
        own length/diameter. The section is drawn vertically, its parent
        extends straight below it, and children fan out at tilt_angle around
        the soma (or stack vertically otherwise); the figure is saved as a PDF.
        """
        if section_name not in self.sections:
            raise ValueError(f"{section_name} not found")

        plt.cla()
        plt.clf()
        plt.close("all")

        L0 = self.sections[section_name]["L"]
        d0 = self.sections[section_name]["diam"]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        main_color = "black"
        other_color = "lightgrey"

        def draw_cyl(start, direction, L, r, color):
            """
            Build a cylinder mesh via self._cylinder and plot its wall and end
            caps on the current axis, returning the cylinder's end point.
            """
            (X, Y, Z), (Xb, Yb, Zb), (Xt, Yt, Zt), end = self._cylinder(
                start, direction, L, r
            )
            ax.plot_surface(X, Y, Z, color=color)
            ax.plot_surface(Xb, Yb, Zb, color=color)
            ax.plot_surface(Xt, Yt, Zt, color=color)
            return end

        # main section always vertical
        start0 = np.array([0.0, 0.0, 0.0])
        end0 = draw_cyl(start0, np.array([0, 0, 1]), 1.0, 0.5, main_color)

        # parent (vertical)
        parent = self.sections[section_name]["parent"]
        if parent:
            Lp = self.sections[parent]["L"] / L0
            dp = self.sections[parent]["diam"] / d0
            draw_cyl(start0, np.array([0, 0, -1]), Lp, dp / 2, other_color)

        childs = self.children.get(section_name, [])
        n_child = len(childs)

        for i, c in enumerate(childs):
            Lc = self.sections[c]["L"] / L0
            dc = self.sections[c]["diam"] / d0

            if "soma" in section_name.lower():
                # angled only for soma
                phi = 2 * np.pi * i / max(1, n_child)
                direction = np.array(
                    [
                        np.cos(phi) * np.sin(tilt_angle),
                        np.sin(phi) * np.sin(tilt_angle),
                        np.cos(tilt_angle),
                    ]
                )
                start_c = end0
            else:
                # vertical stacking (no angle)
                offset = np.array(
                    [(i - (n_child - 1) / 2) * 0.2, (i - (n_child - 1) / 2) * 0.1, 0.0]
                )
                direction = np.array([0, 0, 1])
                start_c = end0 + offset

            draw_cyl(start_c, direction, Lc, dc / 2, other_color)

        ax.set_box_aspect([1, 1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        save_path = os.path.join(self.saveDir, f"localMorph_{section_name}.pdf")
        plt.savefig(save_path)


if __name__ == "__main__":
    # Load NEURON run-time and your geometry file
    h.load_file("stdrun.hoc")
    h.load_file("GeometryAstrocyteCA1.hoc")

    # === USAGE EXAMPLE ===
    plot_3d_morphology()
