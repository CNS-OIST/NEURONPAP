import numpy as np
import matplotlib.pyplot as plt
from neuron import h


def plot_3d_morphology(
    rangevar="v", colormap_name="viridis", fig=None, ax=None, show=False
):
    """
    Plot NEURON morphology in 3D with diameter scaling and a RANGE variable as color.
    """

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

    # Normalize RANGEVAR to colormap
    vmin = min(varlist)
    vmax = max(varlist)
    normed = [(v - vmin) / (vmax - vmin + 1e-12) for v in varlist]

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
    for xs, ys, zs, ds, rv_norm in zip(xlist, ylist, zlist, dlist, normed):
        color = cmap(rv_norm)
        ax.plot(xs, ys, zs, color=color, linewidth=np.mean(ds) / 2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([vmin, vmax])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(rangevar)

    # Integer ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.zaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"3D Morphology Colored by {rangevar}")

    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    # Load NEURON run-time and your geometry file
    h.load_file("stdrun.hoc")
    h.load_file("GeometryAstrocyteCA1.hoc")

    # === USAGE EXAMPLE ===
    plot_3d_morphology()
