import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core import Lattice

lattice = Lattice.hexagonal(3.50, 5.55)

miller = [
    (1, 1, 0), (2, 1, 0), (2, 1, 2), (2, -1, 2), (2, 2, 1), (2, 1, 1),
    (0, 0, 1), (1, 0, 1), (1, 0, 0), (2, 0, 1), (1, 0, 2), (1, 1, 1),
]

energies = [
    1.1080023897174445, 1.2261302042994882, 1.1977770104961567,
    1.1819005784109515, 1.1788328529840504, 1.1694816282074936,
    1.126368472184947, 1.0896930924493664, 1.0499401567597217,
    1.080316119664485, 1.1821109740145999, 1.2219761029482277,
]

wulff = WulffShape(lattice, miller, energies)
ax = wulff.get_plot()

ax.set_title("Forma di Wulff del Lutezio (mp-145), generata con mymatgen, dati estratti dal file JSON", fontsize=10, pad=20)

# --- legenda costruita a mano, indipendente dai colori interni di pymatgen ---
# miller_energy_dict contiene SOLO le facce che sopravvivono nella forma finale
energy_dict = wulff.miller_energy_dict
hkls_sorted = sorted(energy_dict, key=lambda h: energy_dict[h])  # gamma crescente

cmap = plt.get_cmap("Blues")
gmin, gmax = min(energy_dict.values()), max(energy_dict.values())

def gamma_to_color(g):
    if gmax == gmin:
        return cmap(0.6)
    t = 0.25 + 0.65 * (g - gmin) / (gmax - gmin)
    return cmap(t)

legend_handles = [
    Patch(facecolor=gamma_to_color(energy_dict[hkl]), edgecolor="k",
          label=f"{hkl}  γ={energy_dict[hkl]:.3f}")
    for hkl in hkls_sorted
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
          title="Surface energy (J/m²)")
ax.set_axis_on()
ax.xaxis.set_visible(True)
ax.yaxis.set_visible(True)
ax.zaxis.set_visible(True)
ax.grid(True)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_box_aspect((1, 1, 1))

plt.show()