import numpy as np
import matplotlib.pyplot as plt
from neuron import h
from matplotlib.animation import FuncAnimation
from mpi4py import MPI

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def animate_morphology(
    tstop=100,
    dt=1,
    rangevar="v",
    colormap_name="viridis",
    outfile="morphology_animation.mp4",
    zoom=None,
):
    if rank != 0:
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    h.tstop = tstop

    if dt < h.dt:
        h.dt = dt

    frames = int(tstop / dt)

    plot_3d_morphology(
        rangevar=rangevar,
        colormap_name=colormap_name,
        fig=fig,
        ax=ax,
        add_colorbar=True,
        zoom=zoom,
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
        )

        for i in range(int(dt / h.dt)):
            h.fadvance()

    anim = FuncAnimation(fig, update, frames=frames, interval=1, repeat=False)

    anim.save(outfile, fps=30)
    print(f"Saved animation → {outfile}")


def plot_3d_morphology(
    rangevar="v",
    colormap_name="magma",
    fig=None,
    ax=None,
    add_colorbar=True,
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
    cmap = plt.colormaps[colormap_name]

    # Prepare new fig/ax if none provided
    if fig is None or ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    xlist = []
    ylist = []
    zlist = []
    dlist = []
    varlist = []

    # Extract morphology + RANGEVAR
    for sec in h.allsec():
        n3d = int(h.n3d(sec=sec))
        xs = [h.x3d(i, sec=sec) for i in range(n3d)]
        ys = [h.y3d(i, sec=sec) for i in range(n3d)]
        zs = [h.z3d(i, sec=sec) for i in range(n3d)]
        ds = [h.diam3d(i, sec=sec) for i in range(n3d)]

        # color uses section rangevar (segment midpoint)
        try:
            rv = getattr(sec(0.5), rangevar)
        except:
            rv = 0.0

        xlist.append(np.array(xs))
        ylist.append(np.array(ys))
        zlist.append(np.array(zs))
        dlist.append(np.array(ds))
        varlist.append(rv)
    sm = plt.cm.ScalarMappable(cmap=cmap)

    if clim:
        vmin, vmax = clim
    else:
        # Normalize RANGEVAR to colormap
        vmin = min(varlist)
        vmax = max(varlist)
    sm.set_clim(vmin=vmin, vmax=vmax)

    if zoom:
        size = 10
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
        # === SCALING CODE ===
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

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    # === END SCALING ===

    # Plot each section with diameter scaling and color mapping
    for xs, ys, zs, ds, rv in zip(xlist, ylist, zlist, dlist, varlist):
        color = sm.to_rgba(rv)
        ax.plot(xs, ys, zs, color=color, linewidth=np.mean(ds) / 2)

    # Integer ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.zaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # === Colorbar (optional) ===
    if add_colorbar:
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(rangevar)

    ax.set_xlabel("x ($\mu$m)")
    ax.set_ylabel("y ($\mu$m)")
    ax.set_zlabel("z ($\mu$m)")

    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    # Load NEURON run-time and your geometry file
    h.load_file("stdrun.hoc")
    h.load_file("GeometryAstrocyteCA1.hoc")

    # === USAGE EXAMPLE ===
    plot_3d_morphology()
