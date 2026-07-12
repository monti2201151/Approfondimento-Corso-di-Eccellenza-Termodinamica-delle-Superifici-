"""
wulff_simulator.py
==================

Simulatore da zero della costruzione di Wulff (fasi 1-8), in un unico
script autosufficiente, organizzato per sezioni corrispondenti alle
fasi del progetto:

  1. rappresentazione delle famiglie di piani (Miller)
  2. conversione Miller -> vettore normale
  3. normalizzazione delle normali
  4. associazione dell'energia superficiale
  5. costruzione delle equazioni dei piani
  6. intersezione dei semispazi (via dualita' polare)
  7. costruzione delle facce del poliedro
  8. visualizzazione 3D interattiva (matplotlib)

Nessuna libreria usata implementa direttamente la costruzione di
Wulff: scipy.spatial.ConvexHull e' impiegato SOLO come motore
numerico generico di geometria computazionale (inviluppo convesso di
punti), mentre la trasformazione fisica/geometrica che da' senso al
risultato (dualita' polare, assegnazione delle facce, energie
superficiali) e' scritta esplicitamente in questo file.

Uso rapido:
    python3 wulff_simulator.py

Dipendenze:
    numpy, scipy, matplotlib
"""

import itertools
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import CheckButtons


# =======================================================================
# FASE 1: rappresentazione delle famiglie di piani (indici di Miller)
# =======================================================================

@dataclass(frozen=True)
class MillerFamily:
    """
    Una famiglia di piani cristallografici (hkl) con la sua energia
    superficiale gamma_hkl associata.

    hkl   : tuple(int, int, int)  -- indici di Miller di un rappresentante
    gamma : float                  -- energia superficiale (unita' arbitrarie)
    label : str                    -- etichetta leggibile, es. "(111)"
    """
    hkl: tuple
    gamma: float
    label: str = None

    def __post_init__(self):
        if self.label is None:
            auto_label = "(" + "".join(
                str(i) if i >= 0 else f"-{abs(i)}" for i in self.hkl
            ) + ")"
            object.__setattr__(self, "label", auto_label)


# =======================================================================
# FASE 2-3: Miller -> vettore normale -> normalizzazione
# =======================================================================

def reciprocal_lattice_vectors(lattice_matrix: np.ndarray) -> np.ndarray:
    """
    Vettori di base del reticolo reciproco b_i (b_i . a_j = 2*pi*delta_ij)
    a partire dalla matrice del reticolo diretto (righe = a1, a2, a3).

    Necessario in generale (non solo nel caso cubico) perche' la
    normale fisica al piano (hkl) e' il vettore del reticolo reciproco
    G_hkl = h*b1 + k*b2 + l*b3, non semplicemente (h, k, l) in
    coordinate cartesiane, tranne nel caso cubico dove le due cose
    coincidono a meno di un fattore di scala.
    """
    a1, a2, a3 = lattice_matrix
    volume = np.dot(a1, np.cross(a2, a3))
    if np.isclose(volume, 0.0):
        raise ValueError("Volume di cella nullo: vettori di reticolo degeneri.")
    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume
    return np.array([b1, b2, b3])


def cubic_lattice(a: float = 1.0) -> np.ndarray:
    """Matrice di reticolo per un cristallo cubico di parametro 'a'."""
    return a * np.eye(3)


def miller_to_normal(hkl, lattice_matrix: np.ndarray = None) -> np.ndarray:
    """
    Converte (h,k,l) nel vettore normale cartesiano (non normalizzato).
    Se lattice_matrix e' None, si assume reticolo cubico (n // (h,k,l)).
    """
    h, k, l = hkl
    if h == 0 and k == 0 and l == 0:
        raise ValueError("(0,0,0) non e' un piano cristallografico valido.")
    if lattice_matrix is None:
        return np.array([h, k, l], dtype=float)
    b1, b2, b3 = reciprocal_lattice_vectors(lattice_matrix)
    return h * b1 + k * b2 + l * b3


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalizza un vettore a modulo 1. Essenziale: nell'equazione del
    piano n . x = k*gamma la distanza con segno dall'origine vale
    k*gamma SOLO SE ||n|| = 1, altrimenti la geometria risulta distorta.
    """
    norm = np.linalg.norm(v)
    if np.isclose(norm, 0.0):
        raise ValueError("Impossibile normalizzare un vettore nullo.")
    return v / norm


def get_unit_normal(hkl, lattice_matrix: np.ndarray = None) -> np.ndarray:
    """Miller -> normale unitaria, in un solo passo."""
    return normalize(miller_to_normal(hkl, lattice_matrix))


# =======================================================================
# Espansione per simmetria cubica (necessaria per chiudere il poliedro:
# una "famiglia" cristallografica comprende tutte le facce equivalenti
# per simmetria, che condividono la stessa gamma)
# =======================================================================

def expand_cubic_family(hkl):
    """
    Espande (h,k,l) in tutti gli indici equivalenti per simmetria
    cubica O_h: tutte le permutazioni e tutti i cambi di segno.
    Esempi: (100) -> 6 direzioni, (110) -> 12, (111) -> 8.
    """
    h, k, l = hkl
    variants = set()
    for perm in itertools.permutations((h, k, l)):
        for signs in itertools.product([1, -1], repeat=3):
            variant = tuple(s * p if p != 0 else 0 for s, p in zip(signs, perm))
            variants.add(variant)
    return sorted(variants)


# =======================================================================
# FASE 4-5: energia superficiale -> equazione dei piani
# =======================================================================

def build_planes(families, k: float = 1.0, lattice_matrix: np.ndarray = None,
                  expand_symmetry: bool = True):
    """
    Costruisce, per ogni famiglia, l'equazione del piano di Wulff
    n_hat . x = d con d = k * gamma. Se expand_symmetry=True (default),
    ogni famiglia viene prima espansa in tutti i piani equivalenti per
    simmetria cubica (stessa gamma condivisa).

    Ritorna
    -------
    normals : ndarray (N, 3)
    offsets : ndarray (N,)
    source_family : list[MillerFamily]  -- famiglia genitrice per ciascuna normale
    """
    normals, offsets, source_family = [], [], []

    for fam in families:
        if fam.gamma <= 0:
            raise ValueError(
                f"gamma deve essere positivo (famiglia {fam.label}, gamma={fam.gamma}): "
                f"un'energia superficiale <=0 rompe l'ipotesi che l'origine sia "
                f"interna al poliedro."
            )
        hkl_list = expand_cubic_family(fam.hkl) if expand_symmetry else [fam.hkl]
        for hkl in hkl_list:
            normals.append(get_unit_normal(hkl, lattice_matrix))
            offsets.append(k * fam.gamma)
            source_family.append(fam)

    return np.array(normals), np.array(offsets), source_family


# =======================================================================
# FASE 6: intersezione dei semispazi via dualita' polare
# =======================================================================

def halfspace_intersection_polar(normals: np.ndarray, offsets: np.ndarray,
                                  tol: float = 1e-9):
    """
    Calcola i vertici dell'intersezione dei semispazi n_i.x <= d_i
    (d_i > 0) usando la dualita' polare:

        q_i = n_i / d_i               (punti duali)
        P*  = conv{q_i}                (inviluppo convesso, via Qhull)

    I VERTICI del poliedro originale P corrispondono biunivocamente
    alle FACCE di P*: se una faccia di P* giace sul piano A.y + b = 0
    (forma normalizzata restituita da Qhull), il vertice corrispondente
    di P e' v = A / (-b).

    scipy.spatial.ConvexHull e' usato solo come motore geometrico
    generico (calcolo dell'inviluppo convesso di punti); la
    trasformazione duale che da' significato fisico e' esplicita qui.
    """
    if np.any(offsets <= 0):
        raise ValueError("Tutti gli offset d_i devono essere positivi.")

    dual_points = normals / offsets[:, None]
    hull = ConvexHull(dual_points)

    vertices = []
    for eq in hull.equations:
        A, b = eq[:3], eq[3]
        if np.isclose(b, 0.0, atol=tol):
            continue  # faccia duale degenere (vertice all'infinito)
        vertices.append(A / (-b))

    return np.array(vertices)


# =======================================================================
# FASE 7: costruzione delle facce (raggruppamento + ordinamento)
# =======================================================================

def _order_face_vertices(face_vertices: np.ndarray, n_hat: np.ndarray) -> np.ndarray:
    """
    Ordina i vertici di una faccia planare in senso antiorario (visto
    dal lato verso cui punta n_hat), per poter disegnare un poligono
    corretto e non una nuvola di punti.
    """
    centroid = face_vertices.mean(axis=0)
    arbitrary = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.abs(n_hat), arbitrary, atol=1e-6):
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(n_hat, arbitrary)
    u /= np.linalg.norm(u)
    w = np.cross(n_hat, u)

    rel = face_vertices - centroid
    angles = np.arctan2(rel @ w, rel @ u)
    return face_vertices[np.argsort(angles)]


def assign_faces(vertices, normals, offsets, source_family, tol: float = 1e-6):
    """
    Raggruppa i vertici per faccia: v appartiene alla faccia i se
    n_hat_i . v ~= d_i. Scarta le famiglie che non sopravvivono nel
    poliedro finale (< 3 vertici sul piano): e' un risultato fisico
    corretto (energia superficiale troppo alta rispetto alle altre),
    non un errore.

    Ritorna
    -------
    faces        : list[ndarray]        -- vertici ordinati per faccia
    face_family  : list[MillerFamily]   -- famiglia genitrice di ciascuna faccia
    face_normal  : list[ndarray]        -- normale unitaria di ciascuna faccia
    face_offset  : list[float]          -- offset d_i di ciascuna faccia
    """
    faces, face_family, face_normal, face_offset = [], [], [], []

    for i, (n_hat, d) in enumerate(zip(normals, offsets)):
        residual = vertices @ n_hat - d
        pts = vertices[np.abs(residual) < tol]
        if pts.shape[0] < 3:
            continue
        faces.append(_order_face_vertices(pts, n_hat))
        face_family.append(source_family[i])
        face_normal.append(n_hat)
        face_offset.append(d)

    return faces, face_family, face_normal, face_offset


def build_wulff_polyhedron(families, k: float = 1.0, lattice_matrix: np.ndarray = None,
                            expand_symmetry: bool = True):
    """
    Esegue in sequenza le fasi 4-7.

    Ritorna un dict con: vertices, faces, face_family, face_normal,
    face_offset, normals, offsets (questi ultimi due sono TUTTI i
    piani generati, anche quelli che non sopravvivono nel poliedro
    finale).
    """
    normals, offsets, source_family = build_planes(
        families, k=k, lattice_matrix=lattice_matrix, expand_symmetry=expand_symmetry
    )
    vertices = halfspace_intersection_polar(normals, offsets)
    faces, face_family, face_normal, face_offset = assign_faces(
        vertices, normals, offsets, source_family
    )

    return {
        "vertices": vertices,
        "faces": faces,
        "face_family": face_family,
        "face_normal": face_normal,
        "face_offset": face_offset,
        "normals": normals,
        "offsets": offsets,
    }


# =======================================================================
# FASE 8: visualizzazione 3D
# =======================================================================

def _assign_colors(label_to_gamma, cmap_name="Blues"):
    """
    Assegna a ciascuna famiglia un colore lungo una colormap sequenziale
    (chiaro -> scuro al crescere di gamma), invece di un colore
    categorico arbitrario.

    Perche' una colormap sequenziale e non "tab10": qui il colore deve
    veicolare un'informazione ORDINATA (l'energia superficiale), non
    solo distinguere categorie senza relazione d'ordine tra loro.
    Gamma piu' bassa (faccia piu' stabile, tipicamente piu' estesa nel
    poliedro) -> colore piu' chiaro; gamma piu' alta -> colore piu' scuro.

    Parametri
    ---------
    label_to_gamma : dict[str, float]
        Mappa etichetta di famiglia -> valore di gamma.
    cmap_name : str
        Nome di una colormap sequenziale di matplotlib
        (es. "Blues", "Greys", "OrRd", "viridis" e' invece percettiva
        ma non monocromatica chiaro->scuro in senso stretto).

    Ritorna
    -------
    dict[str, tuple] : etichetta -> colore RGBA
    """
    labels = list(label_to_gamma.keys())
    gammas = np.array([label_to_gamma[lab] for lab in labels])

    gmin, gmax = gammas.min(), gammas.max()
    cmap = plt.get_cmap(cmap_name)

    colors = {}
    for lab, g in zip(labels, gammas):
        if np.isclose(gmax, gmin):
            # Tutte le famiglie hanno la stessa gamma (es. caso isotropo):
            # niente da graduare, uso un valore medio fisso della colormap.
            t = 0.6
        else:
            t_norm = (g - gmin) / (gmax - gmin)          # 0 (min) .. 1 (max)
            t = 0.25 + 0.65 * t_norm                       # evita estremi troppo
                                                            # chiari/scuri (bianco/nero)
        colors[lab] = cmap(t)

    return colors


def plot_wulff_shape(result, show_normals=True, show_center=True,
                      show_reference_sphere=False, sphere_radius=None,
                      color_map="Blues", title="Poliedro di Wulff",
                      save_path=None, save_dpi=300):
    """
    Visualizza il poliedro di Wulff in 3D con matplotlib.

    Caratteristiche:
      - rotazione libera (finestra matplotlib interattiva)
      - assi cartesiani disegnati esplicitamente attraverso l'origine
      - griglia attiva
      - stessa scala sui tre assi (calcolata manualmente)
      - colore delle facce per famiglia cristallografica
      - normali alle facce disegnate come frecce
      - punto centrale del cristallo (opzionale)
      - sfera di riferimento per il confronto col caso isotropo (opzionale)
      - checkbox per mostrare/nascondere singole famiglie di facce

    Parametri
    ---------
    result : dict
        Output di build_wulff_polyhedron().
    show_normals, show_center, show_reference_sphere : bool
    sphere_radius : float, opzionale
        Raggio della sfera di riferimento; se None, si usa la media
        degli offset (k*gamma) delle facce presenti nel poliedro.
    color_map : str
        Nome di una colormap sequenziale matplotlib usata per colorare
        le facce dal piu' chiaro (gamma minima) al piu' scuro (gamma
        massima). Default "Blues"; altre opzioni sensate: "Greys",
        "OrRd", "Purples", "Greens".
    save_path : str | None
        Se fornito, salva anche una copia su file alla fine.
    save_dpi : int
        Risoluzione usata solo quando si salva su file.
    """
    faces = result["faces"]
    face_family = result["face_family"]
    face_normal = result["face_normal"]
    face_offset = result["face_offset"]
    vertices = result["vertices"]

    labels = [fam.label for fam in face_family]
    label_to_gamma = {fam.label: fam.gamma for fam in face_family}
    colors = _assign_colors(label_to_gamma, cmap_name=color_map)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # --- facce, raggruppate per famiglia per il toggle interattivo ---
    family_artists = defaultdict(list)

    for face, fam, n_hat, d in zip(faces, face_family, face_normal, face_offset):
        color = colors[fam.label]
        poly = Poly3DCollection([face], facecolor=color, edgecolor="k",
                                 linewidths=0.6, alpha=0.95)
        ax.add_collection3d(poly)
        family_artists[fam.label].append(poly)

        if show_normals:
            centroid = face.mean(axis=0)
            arrow = ax.quiver(centroid[0], centroid[1], centroid[2],
                               n_hat[0], n_hat[1], n_hat[2],
                               length=0.35 * d, color=color, linewidth=1.5,
                               arrow_length_ratio=0.3)
            family_artists[fam.label].append(arrow)

    # --- punto centrale del cristallo ---
    if show_center:
        center_pt = ax.scatter([0], [0], [0], color="black", s=40,
                                label="Centro del cristallo", depthshade=False)

    # --- sfera di riferimento (limite isotropo) ---
    if show_reference_sphere:
        r = sphere_radius if sphere_radius is not None else np.mean(face_offset)
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        xs = r * np.outer(np.cos(u), np.sin(v))
        ys = r * np.outer(np.sin(u), np.sin(v))
        zs = r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, color="gray", linewidth=0.3, alpha=0.4)

    # --- scala isotropa: stesso range su x, y, z ---
    max_range = np.abs(vertices).max() * 1.3
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    try:
        ax.set_box_aspect((1, 1, 1))  # richiede matplotlib >= 3.3
    except AttributeError:
        pass

    # --- assi cartesiani espliciti attraverso l'origine ---
    axis_len = max_range
    ax.plot([-axis_len, axis_len], [0, 0], [0, 0], color="red", linewidth=0.8)
    ax.plot([0, 0], [-axis_len, axis_len], [0, 0], color="green", linewidth=0.8)
    ax.plot([0, 0], [0, 0], [-axis_len, axis_len], color="blue", linewidth=0.8)
    ax.text(axis_len, 0, 0, "x", color="red")
    ax.text(0, axis_len, 0, "y", color="green")
    ax.text(0, 0, axis_len, "z", color="blue")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.grid(True)

    # --- legenda per famiglia (un patch per etichetta), con gamma mostrata ---
    from matplotlib.patches import Patch

    # Recupero il valore di gamma associato a ciascuna etichetta di famiglia
    # (tutte le facce con la stessa label condividono la stessa gamma, per
    # costruzione: e' la stessa famiglia cristallografica espansa per simmetria)
    label_to_gamma = {fam.label: fam.gamma for fam in face_family}

    legend_handles = [
        Patch(facecolor=colors[lab], edgecolor="k",
              label=f"{lab}  γ={label_to_gamma[lab]:.3f}")
        for lab in sorted(colors, key=lambda l: label_to_gamma[l])
    ]
    legend = ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
          title="Surface energy (J/m²)")

    check = None
    if save_path is None:
        # --- checkbox per mostrare/nascondere famiglie ---
        labels_sorted = sorted(family_artists.keys())
        check_ax = fig.add_axes([0.02, 0.5, 0.13, 0.04 * len(labels_sorted) + 0.02])
        check = CheckButtons(check_ax, labels_sorted, [True] * len(labels_sorted))

        def _toggle(label):
            for artist in family_artists[label]:
                artist.set_visible(not artist.get_visible())
            fig.canvas.draw_idle()

        check.on_clicked(_toggle)

    plt.tight_layout()
    if save_path is not None:
        original_title = ax.get_title()
        legend.set_visible(False)
        ax.set_title("")
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
        ax.set_title(original_title)
        legend.set_visible(True)
    return fig, ax, check  # 'check' va tenuto in vita: altrimenti i click non funzionano


# =======================================================================
# MAIN // VALORI PER un materiale isotropo, indici di Miller fino a 1, energia superficiale arbitraria
# =======================================================================

if __name__ == "__main__":
    families = [
        MillerFamily(hkl=(1, 0, 0), gamma=1.0),
        MillerFamily(hkl=(1, 1, 0), gamma=1.0),
        MillerFamily(hkl=(1, 1, 1), gamma=1.0),
    ]
    result = build_wulff_polyhedron(families, k=1.0)

    print(f'Cristallo isotropo:')
    print(f'INDICI DI MILLER FINO A 1 (famiglie cristallografiche considerate):')
    for fam in families:
        print(f'  {fam.hkl}: γ = {fam.gamma}')
    print(f"Piani totali (dopo espansione di simmetria): {len(result['normals'])}")
    print(f"Vertici del poliedro: {len(result['vertices'])}")
    print(f"Facce nel poliedro finale: {len(result['faces'])}")

    fig, ax, check = plot_wulff_shape(
        result,
        show_normals=False,
        show_center=True,
        show_reference_sphere=False,
        title="The Wulff shape of an isotropic crystal generated with surface energies for facets up to a max Miller index of 1.",
        save_path="wulff_miller_1.png",
    )

    plt.show()



# =======================================================================
# MAIN // VALORI PER un materiale isotropo, indici di Miller fino a 2, energia superficiale arbitraria
# =======================================================================

if __name__ == "__main__":
    families = [
        MillerFamily((1,0,0), 1.0),
        MillerFamily((1,1,0), 1.0),
        MillerFamily((1,1,1), 1.0),
        MillerFamily((2,1,0), 1.0),
        MillerFamily((2,1,1), 1.0),
        MillerFamily((2,2,1), 1.0),
    ]
    result = build_wulff_polyhedron(families, k=1.0)

    print(f'Cristallo isotropo:')
    print(f'INDICI DI MILLER FINO A 2 (famiglie cristallografiche considerate):')
    for fam in families:
        print(f'  {fam.hkl}: γ = {fam.gamma}')
    print(f"Piani totali (dopo espansione di simmetria): {len(result['normals'])}")
    print(f"Vertici del poliedro: {len(result['vertices'])}")
    print(f"Facce nel poliedro finale: {len(result['faces'])}")

    fig, ax, check = plot_wulff_shape(
        result,
        show_normals=False,
        show_center=True,
        show_reference_sphere=False,
        title="The Wulff shape of an isotropic crystal generated with surface energies for facets up to a max Miller index of 2.",
        save_path="wulff_miller_2.png",
    )

    plt.show()



# =======================================================================
# MAIN // VALORI PER un materiale isotropo, indici di Miller fino a 3, energia superficiale arbitraria
# =======================================================================

if __name__ == "__main__":
    families = [
        MillerFamily((1,0,0), 1.0),
        MillerFamily((1,1,0), 1.0),
        MillerFamily((1,1,1), 1.0),
        MillerFamily((2,1,0), 1.0),
        MillerFamily((2,1,1), 1.0),
        MillerFamily((2,2,1), 1.0),
        MillerFamily((3,1,0), 1.0),
        MillerFamily((3,1,1), 1.0),
        MillerFamily((3,2,0), 1.0),
        MillerFamily((3,2,1), 1.0),
        MillerFamily((3,2,2), 1.0),
        MillerFamily((3,3,1), 1.0),
        MillerFamily((3,3,2), 1.0),
    ]
    result = build_wulff_polyhedron(families, k=1.0)

    print(f'Cristallo isotropo:')
    print(f'INDICI DI MILLER FINO A 3 (famiglie cristallografiche considerate):')
    for fam in families:
        print(f'  {fam.hkl}: γ = {fam.gamma}')
    print(f"Piani totali (dopo espansione di simmetria): {len(result['normals'])}")
    print(f"Vertici del poliedro: {len(result['vertices'])}")
    print(f"Facce nel poliedro finale: {len(result['faces'])}")

    fig, ax, check = plot_wulff_shape(
        result,
        show_normals=False,
        show_center=True,
        show_reference_sphere=False,
        title="The Wulff shape of an isotropic crystal generated with surface energies for facets up to a max Miller index of 3.",
        save_path="wulff_miller_3.png",
    )

    plt.show()


# =======================================================================
# MAIN // VALORI PER un materiale isotropo, indici di Miller fino a 4, energia superficiale arbitraria
# =======================================================================

if __name__ == "__main__":
    families = [
        MillerFamily((1,0,0), 1.0),
        MillerFamily((1,1,0), 1.0),
        MillerFamily((1,1,1), 1.0),
        MillerFamily((2,1,0), 1.0),
        MillerFamily((2,1,1), 1.0),
        MillerFamily((2,2,1), 1.0),
        MillerFamily((3,1,0), 1.0),
        MillerFamily((3,1,1), 1.0),
        MillerFamily((3,2,0), 1.0),
        MillerFamily((3,2,1), 1.0),
        MillerFamily((3,2,2), 1.0),
        MillerFamily((3,3,1), 1.0),
        MillerFamily((3,3,2), 1.0),
        MillerFamily((4,1,0), 1.0),
        MillerFamily((4,1,1), 1.0),
        MillerFamily((4,2,0), 1.0),
        MillerFamily((4,2,1), 1.0),
        MillerFamily((4,2,2), 1.0),
        MillerFamily((4,3,1), 1.0),
        MillerFamily((4,3,2), 1.0),
        MillerFamily((4,4,1), 1.0),
        MillerFamily((4,4,2), 1.0),
    ]
    result = build_wulff_polyhedron(families, k=1.0)

    print(f'Cristallo isotropo:')
    print(f'INDICI DI MILLER FINO A 4 (famiglie cristallografiche considerate):')
    for fam in families:
        print(f'  {fam.hkl}: γ = {fam.gamma}')
    print(f"Piani totali (dopo espansione di simmetria): {len(result['normals'])}")
    print(f"Vertici del poliedro: {len(result['vertices'])}")
    print(f"Facce nel poliedro finale: {len(result['faces'])}")

    fig, ax, check = plot_wulff_shape(
        result,
        show_normals=False,
        show_center=True,
        show_reference_sphere=False,
        title="The Wulff shape of an isotropic crystal generated with surface energies for facets up to a max Miller index of 4.",
        save_path="wulff_miller_4.png",
    )

    plt.show()