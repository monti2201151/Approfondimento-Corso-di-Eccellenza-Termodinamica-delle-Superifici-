"""
wulff_simulator.py
==================

Simulatore da zero della costruzione di Wulff (fasi 1-8), esteso alla
costruzione di Winterbottom / Wulff-Kaishew per un cristallo su
substrato rigido, in un unico script autosufficiente, organizzato per
sezioni corrispondenti alle fasi del progetto:

  1. rappresentazione delle famiglie di piani (Miller)
  2. conversione Miller -> vettore normale
  3. normalizzazione delle normali
  4. associazione dell'energia superficiale (+ substrato: Winterbottom)
  5. costruzione delle equazioni dei piani
  6. intersezione dei semispazi (via dualita' polare, generalizzata
     con centro di Chebyshev per reggere offset non positivi)
  7. costruzione delle facce del poliedro
  8. visualizzazione 3D interattiva (matplotlib)

Nessuna libreria usata implementa direttamente la costruzione di
Wulff o di Winterbottom: scipy.spatial.ConvexHull e' impiegato SOLO
come motore numerico generico di geometria computazionale (inviluppo
convesso di punti), scipy.optimize.linprog e' impiegato SOLO come
motore generico di programmazione lineare (per il centro di
Chebyshev, necessario a generalizzare la dualita' polare), mentre la
trasformazione fisica/geometrica che da' senso al risultato (dualita'
polare, formula di Winterbottom, assegnazione delle facce, energie
superficiali/interfacciali) e' scritta esplicitamente in questo file.

COSTRUZIONE DI WINTERBOTTOM (Wulff-Kaishew) -- idea fisica in breve
--------------------------------------------------------------------
Il cristallo libero minimizza sum_i gamma_i * A_i a volume fissato, e
la soluzione e' il poliedro intersezione dei semispazi n_i.x <= d_i
con d_i = k*gamma_i (teorema di Wulff).

Quando il cristallo cresce su un substrato rigido, la faccia rivolta
verso il substrato non e' piu' una superficie libera cristallo/vapore
con energia gamma_film: e' un'interfaccia cristallo/substrato. Il
funzionale di energia totale da minimizzare diventa

    E = sum_{i != contatto} gamma_i * A_i
        + gamma_int * A_contatto - gamma_sub * A_contatto

(si toglie la superficie di substrato "nuda" che c'era prima del
cristallo, gamma_sub * A_contatto, e si crea al suo posto
l'interfaccia gamma_int * A_contatto). Applicando lo stesso principio
variazionale di Wulff a questo funzionale, la faccia di contatto
riceve un OFFSET DIVERSO, mentre tutte le altre facce restano
invariate:

    d_contatto = k * (gamma_film - Delta_gamma),   Delta_gamma = gamma_sub - gamma_int

Questo e' esattamente e SOLO un nuovo vincolo nel problema geometrico
di intersezione dei semispazi (non un taglio a posteriori della forma
di Wulff gia' costruita): si veda SubstrateSpec e la relativa logica
in build_planes().

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
from scipy.optimize import linprog

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


@dataclass(frozen=True)
class SubstrateSpec:
    """
    Definisce il substrato rigido su cui cresce il cristallo, per la
    costruzione di Winterbottom (Wulff-Kaishew).

    Fisicamente il substrato NON aggiunge una nuova "famiglia
    cristallografica": modifica il valore di offset che il problema
    geometrico assegna a UNA SOLA faccia, quella rivolta verso il
    substrato. Tutti gli altri campi qui sotto servono a calcolare
    quel singolo valore modificato (si veda la formula in
    _winterbottom_contact_offset più sotto).

    normal_hkl    : tuple(int,int,int) -- direzione cristallografica
                     della normale al substrato, orientata DAL
                     substrato VERSO il cristallo (stessa convenzione
                     di MillerFamily.hkl). Esempio: (0,0,1) se il
                     cristallo cresce lungo +z.
    gamma_contact : float | None -- energia superficiale "nuda"
                     gamma_film che avrebbe la faccia del cristallo a
                     contatto se fosse una superficie libera (cioe' il
                     valore di riferimento nella formula di
                     Winterbottom, indipendente dal substrato). Se
                     None, viene cercata automaticamente tra le
                     MillerFamily fornite a build_planes() la famiglia
                     la cui normale coincide con -normal_hkl; se non
                     la si trova, e' un errore (il valore va allora
                     fornito esplicitamente).
    gamma_int     : float -- energia dell'interfaccia cristallo/substrato.
    gamma_sub     : float -- energia della superficie libera del
                     substrato (substrato scoperto, non ricoperto dal
                     cristallo).
    label         : str -- etichetta per la faccia di contatto nei
                     grafici/legenda.
    """
    normal_hkl: tuple
    gamma_int: float
    gamma_sub: float
    gamma_contact: float = None
    label: str = "substrato"


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

def _winterbottom_contact_offset(k: float, gamma_contact: float,
                                  substrate: "SubstrateSpec") -> float:
    """
    Formula di Winterbottom per l'offset della faccia di contatto:

        d_contatto = k * (gamma_film - Delta_gamma),
        Delta_gamma = gamma_sub - gamma_int

    Derivazione fisica: sostituendo nel funzionale di Wulff il
    termine di energia della faccia di contatto (che nella
    costruzione libera sarebbe gamma_film * A) con il costo reale di
    "coprire" il substrato (gamma_int * A al posto di gamma_sub * A
    che c'era gia' prima, quindi una variazione netta di
    (gamma_int - gamma_sub) * A = -Delta_gamma * A), il principio
    variazionale di Wulff assegna a QUELLA sola faccia l'offset
    k*(gamma_film - Delta_gamma) invece di k*gamma_film, lasciando
    tutte le altre facce invariate.

    Casi limite (nessun caso speciale nel codice: e' automatico):
      - Delta_gamma = 0            -> d = k*gamma_film (Wulff libero)
      - Delta_gamma > 0 (adesione favorevole, gamma_int<gamma_sub)
                                    -> d < k*gamma_film: faccia "spinta
                                       verso l'interno", cristallo
                                       appiattito (bagnamento)
      - Delta_gamma < 0 (interfaccia sfavorita, gamma_int>gamma_sub)
                                    -> d > k*gamma_film: il vincolo
                                       diventa piu' permissivo di
                                       quello naturale della faccia e
                                       quindi RIDONDANTE rispetto agli
                                       altri vincoli del cristallo: il
                                       poliedro finale coincide col
                                       Wulff libero (il cristallo "si
                                       alza", tocca il substrato solo
                                       nella faccia/spigolo naturale).
                                       Questo emerge da solo in
                                       assign_faces(), senza rilevarlo
                                       esplicitamente qui.
    """
    delta_gamma = substrate.gamma_sub - substrate.gamma_int
    return k * (gamma_contact - delta_gamma)


def build_planes(families, k: float = 1.0, lattice_matrix: np.ndarray = None,
                  expand_symmetry: bool = True, substrate: "SubstrateSpec" = None):
    """
    Costruisce, per ogni famiglia, l'equazione del piano di Wulff
    n_hat . x = d con d = k * gamma. Se expand_symmetry=True (default),
    ogni famiglia viene prima espansa in tutti i piani equivalenti per
    simmetria cubica (stessa gamma condivisa).

    Se substrate (SubstrateSpec) e' fornito, IL PROBLEMA GEOMETRICO
    STESSO viene modificato (non la forma gia' costruita): la faccia
    la cui normale coincide con -substrate.normal_hkl riceve l'offset
    di Winterbottom al posto di k*gamma_film (si veda
    _winterbottom_contact_offset). Se tra le famiglie fornite non
    esiste alcuna faccia con quella normale, ne viene aggiunta una
    nuova, rappresentante il vincolo imposto dal substrato stesso: e'
    comunque un vincolo dello stesso tipo n.x<=d di tutti gli altri,
    che l'intersezione dei semispazi tratta in modo del tutto
    uniforme (nessun ramo speciale a valle).

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

    if substrate is not None:
        # Normale della faccia di CONTATTO: e' rivolta dal cristallo
        # VERSO il substrato, quindi opposta alla normale del
        # substrato (che punta dal substrato verso il cristallo).
        n_contact = -get_unit_normal(substrate.normal_hkl, lattice_matrix)

        # gamma_film di riferimento per la faccia di contatto: o
        # fornita esplicitamente, o cercata tra le famiglie originarie
        # (comportamento comodo, ma mai implicito su valori numerici:
        # se non si trova, e' un errore esplicito).
        gamma_contact = substrate.gamma_contact
        if gamma_contact is None:
            for n_hat, fam in zip(normals, source_family):
                if np.dot(n_hat, n_contact) > 1 - 1e-8:
                    gamma_contact = fam.gamma
                    break
            if gamma_contact is None:
                raise ValueError(
                    "SubstrateSpec.gamma_contact non fornito e nessuna "
                    "MillerFamily tra quelle date ha normale coincidente "
                    "con -normal_hkl: specificare gamma_contact esplicitamente."
                )

        d_contact = _winterbottom_contact_offset(k, gamma_contact, substrate)
        contact_family = MillerFamily(
            hkl=substrate.normal_hkl, gamma=d_contact / k, label=f"({substrate.label})"
        )

        # IMPORTANTE: il vincolo di Winterbottom viene SEMPRE AGGIUNTO,
        # mai usato per sovrascrivere un eventuale vincolo naturale
        # gia' presente con la stessa normale. Motivo fisico: la faccia
        # di contatto e' vincolata dal MINIMO tra il suo offset
        # "naturale" k*gamma_film (se esiste, cioe' se il cristallo
        # avrebbe comunque quella faccia da libero) e l'offset
        # modificato dal substrato k*(gamma_film - Delta_gamma).
        #   - se il vincolo del substrato e' PIU' STRETTO (bagnamento
        #     favorevole), domina lui: quello naturale diventa
        #     ridondante e viene scartato automaticamente in
        #     assign_faces() (nessun vertice giace su di esso entro
        #     tolleranza), senza bisogno di rimuoverlo qui.
        #   - se il vincolo del substrato e' PIU' PERMISSIVO (adesione
        #     sfavorita), e' lui a diventare ridondante e a essere
        #     scartato: il poliedro finale ricade quindi ESATTAMENTE
        #     sulla faccia naturale del cristallo libero, che deve
        #     restare nell'insieme di vincoli per poter vincere.
        # Sovrascrivere il vincolo naturale (invece di aggiungere)
        # sarebbe fisicamente sbagliato: rimuoverebbe l'unico vincolo
        # che, nel regime di adesione sfavorita, impedisce al
        # poliedro di espandersi oltre la faccia naturale del
        # cristallo libero.
        normals.append(n_contact)
        offsets.append(d_contact)
        source_family.append(contact_family)

    return np.array(normals), np.array(offsets), source_family


# =======================================================================
# FASE 6: intersezione dei semispazi via dualita' polare
# =======================================================================

def _chebyshev_center(normals: np.ndarray, offsets: np.ndarray, tol: float = 1e-9):
    """
    Trova un punto x0 STRETTAMENTE INTERNO all'intersezione dei
    semispazi n_i.x <= d_i, come centro della piu' grande sfera
    inscritta (centro di Chebyshev), risolvendo la programmazione
    lineare

        max_{x,r}  r
        s.t.       n_i . x + r*||n_i|| <= d_i   per ogni i
                   r >= 0

    (le n_i sono unitarie in questo codice, quindi ||n_i||=1, ma si
    tiene la forma generale per chiarezza/robustezza).

    Perche' serve: la dualita' polare classica q_i = n_i/d_i assume
    implicitamente che l'origine sia interna al poliedro (d_i>0 per
    ogni i). Nella costruzione di Winterbottom questo NON e' piu'
    garantito: l'offset della faccia di contatto puo' diventare nullo
    o negativo in regime di bagnamento forte (adesione molto
    favorevole), perche' il piano di contatto taglia dalla parte
    opposta rispetto al centro del cristallo libero. Non e' un caso
    patologico da escludere: e' esattamente il regime fisico di
    bagnamento totale richiesto. Il centro di Chebyshev fornisce, in
    OGNI regime (bagnamento, bagnamento parziale, distacco), un punto
    interno valido rispetto al quale ridefinire la dualita' polare,
    senza distinguere i casi esplicitamente.

    scipy.optimize.linprog e' usato solo come motore generico di
    programmazione lineare (non implementa nulla di specifico alla
    costruzione di Wulff/Winterbottom).
    """
    n = normals.shape[0]
    norms = np.linalg.norm(normals, axis=1)

    # variabili: [x, y, z, r]; minimizzo -r per massimizzare r
    c = np.array([0.0, 0.0, 0.0, -1.0])
    A_ub = np.hstack([normals, norms[:, None]])
    b_ub = offsets
    bounds = [(None, None), (None, None), (None, None), (0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not res.success or res.x[3] <= tol:
        raise ValueError(
            "Nessun punto strettamente interno trovato: con questi "
            "parametri (gamma, substrato) l'intersezione dei semispazi "
            "e' vuota o degenere. Non e' un errore numerico da "
            "aggirare: indica una combinazione di energie non fisica "
            "(es. Delta_gamma cosi' grande da rendere il problema "
            "geometrico non ammissibile)."
        )
    return res.x[:3]


def halfspace_intersection_polar(normals: np.ndarray, offsets: np.ndarray,
                                  tol: float = 1e-9):
    """
    Calcola i vertici dell'intersezione dei semispazi n_i.x <= d_i
    usando la dualita' polare, generalizzata per non richiedere che
    l'origine sia interna al poliedro (d_i>0 per ogni i):

        1. si trova x0 strettamente interno a TUTTI i vincoli, col
           centro di Chebyshev (_chebyshev_center);
        2. si trasla il sistema, y = x - x0: i nuovi offset
           d_i' = d_i - n_i.x0 sono per costruzione > 0 per ogni i;
        3. dualita' polare STANDARD nel sistema traslato:

               q_i = n_i / d_i'          (punti duali)
               P*  = conv{q_i}            (inviluppo convesso, via Qhull)

           I VERTICI del poliedro traslato corrispondono alle FACCE di
           P*: se una faccia di P* giace sul piano A.y + b = 0 (forma
           normalizzata restituita da Qhull), il vertice corrispondente
           e' y = A / (-b);
        4. si ritraslano i vertici, x = y + x0.

    Questo e' un'estensione MATEMATICA del metodo di dualita' polare,
    non un post-processing sulla forma: nel caso in cui l'origine sia
    gia' interna (Wulff libero, tutti d_i>0), x0 e' semplicemente un
    punto interno qualunque e il risultato e' identico (a meno di
    traslazione dell'origine ausiliaria) a quello del metodo classico.

    scipy.spatial.ConvexHull e' usato solo come motore geometrico
    generico (calcolo dell'inviluppo convesso di punti); la
    trasformazione duale che da' significato fisico e' esplicita qui.
    """
    x0 = _chebyshev_center(normals, offsets, tol=tol)
    shifted_offsets = offsets - normals @ x0

    if np.any(shifted_offsets <= tol):
        # Non dovrebbe accadere se _chebyshev_center ha avuto successo
        # (per costruzione r>tol garantisce shifted_offsets>=r>tol),
        # ma si controlla comunque esplicitamente per non nascondere
        # eventuali problemi numerici al limite.
        raise ValueError("Traslazione al centro di Chebyshev non strettamente interna.")

    dual_points = normals / shifted_offsets[:, None]
    hull = ConvexHull(dual_points)

    vertices = []
    for eq in hull.equations:
        A, b = eq[:3], eq[3]
        if np.isclose(b, 0.0, atol=tol):
            continue  # faccia duale degenere (vertice all'infinito)
        vertices.append(A / (-b) + x0)  # ritraslo nel sistema originale

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
                            expand_symmetry: bool = True, substrate: "SubstrateSpec" = None):
    """
    Esegue in sequenza le fasi 4-7.

    substrate : SubstrateSpec, opzionale -- se fornito, il vincolo
        della faccia di contatto e' modificato secondo la costruzione
        di Winterbottom (si veda build_planes). Se None, si ottiene
        esattamente la costruzione di Wulff libera di prima.

    Ritorna un dict con: vertices, faces, face_family, face_normal,
    face_offset, normals, offsets (questi ultimi due sono TUTTI i
    piani generati, anche quelli che non sopravvivono nel poliedro
    finale).
    """
    normals, offsets, source_family = build_planes(
        families, k=k, lattice_matrix=lattice_matrix, expand_symmetry=expand_symmetry,
        substrate=substrate,
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
                      substrate_plane=None):
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
      - piano del substrato (opzionale), disegnato alla sua posizione
        REALE anche quando non taglia il poliedro (regime di adesione
        sfavorita): serve a rendere visibile che il vincolo si e'
        allontanato dal centro, anche se e' diventato ridondante.

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
    substrate_plane : ndarray, opzionale
        n_contact: normale unitaria della faccia di contatto (rivolta
        dal cristallo verso il substrato). Se fornito, il piano del
        substrato viene disegnato NON alla posizione "grezza"
        k*(gamma_film - Delta_gamma) della formula di Winterbottom,
        ma alla POSIZIONE DI TANGENZA REALE del poliedro finale, cioe'
        alla funzione di supporto

            d_reale = max_v (n_contact . v)   sui vertici v del poliedro.

        Motivo fisico: un cristallo rigido su un substrato rigido deve
        toccarlo per contatto meccanico, non puo' fluttuare a una
        distanza arbitraria ne' deformarsi/allungarsi per raggiungerlo.
        Quando il vincolo di Winterbottom e' quello attivo (bagnamento),
        d_reale coincide con l'offset della formula; quando invece e'
        ridondante (adesione sfavorita), il vincolo che vince e'
        automaticamente quello naturale della faccia libera del
        cristallo (o il vertice/spigolo piu' sporgente in quella
        direzione, se non esiste una faccia naturale), e d_reale lo
        riflette senza bisogno di distinguere i due casi qui: e' un
        semplice max su un insieme di punti gia' calcolato da
        build_wulff_polyhedron.
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

    # Se richiesto, calcolo la posizione REALE di tangenza col
    # substrato: non il valore "grezzo" della formula di Winterbottom
    # (che puo' essere piu' lontano della faccia naturale e quindi
    # fisicamente irraggiungibile), ma la funzione di supporto del
    # poliedro EFFETTIVO gia' calcolato, cioe' il punto/faccia con cui
    # il cristallo tocca davvero il substrato per contatto meccanico.
    # E' un semplice max su vertici gia' noti: nessun caso speciale,
    # nessuna distinzione esplicita tra "bagna" / "non bagna".
    d_contact_real = None
    if substrate_plane is not None:
        n_contact = substrate_plane
        d_contact_real = float(np.max(vertices @ n_contact))

    # --- scala isotropa: stesso range su x, y, z ---
    extent_points = [vertices]
    if d_contact_real is not None:
        extent_points.append(np.abs(n_contact * d_contact_real)[None, :])
    max_range = np.abs(np.vstack(extent_points)).max() * 1.3

    # --- piano del substrato, disegnato TANGENTE al poliedro reale ---
    # (mai a una distanza arbitraria: il cristallo non puo' fluttuare
    # ne' allungarsi per raggiungere il substrato, deve toccarlo).
    if d_contact_real is not None:
        # base ortonormale del piano n_contact.x = d_contact_real
        arbitrary = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.abs(n_contact), arbitrary, atol=1e-6):
            arbitrary = np.array([0.0, 1.0, 0.0])
        u_hat = np.cross(n_contact, arbitrary)
        u_hat /= np.linalg.norm(u_hat)
        w_hat = np.cross(n_contact, u_hat)
        center_plane = n_contact * d_contact_real
        s = max_range  # semilato del quadrato disegnato per il substrato
        corners = np.array([
            center_plane + s * u_hat + s * w_hat,
            center_plane - s * u_hat + s * w_hat,
            center_plane - s * u_hat - s * w_hat,
            center_plane + s * u_hat - s * w_hat,
        ])
        substrate_poly = Poly3DCollection([corners], facecolor="0.6",
                                           edgecolor="0.3", alpha=0.25,
                                           linewidths=0.8)
        ax.add_collection3d(substrate_poly)
        ax.text(*center_plane, "substrato", color="0.3", fontsize=8)

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
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
          title="Surface energy (J/m²)")

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
    return fig, ax, check  # 'check' va tenuto in vita: altrimenti i click non funzionano


# =======================================================================
# MAIN // VALORI PER cristallo cubico isotropo, piani cristallografici fino a Miller index 1
# =======================================================================

if __name__ == "__main__":
    # Cristallo cubico anisotropo: gamma diversa per famiglia, cosi'
    # la faccia (001) (quella che tocca il substrato) e' visibile e
    # distinguibile nel poliedro anche nella costruzione di Wulff
    # libera, ed e' piu' facile apprezzare l'effetto del substrato.
    families = [
        MillerFamily(hkl=(1, 1, 0), gamma=1.15),
        MillerFamily(hkl=(1, 0, 0), gamma=1.00),
        MillerFamily(hkl=(1, 1, 1), gamma=0.90),
    ]
    k = 1.0

    # Substrato con normale (001): il cristallo cresce lungo +z e la
    # faccia (00-1) (normale opposta) e' quella di contatto. gamma_int
    # e gamma_sub definiscono i tre regimi fisici richiesti.
    regimes = {
        "Bagnamento forte (Delta_gamma > gamma_film: cristallo appiattito)":
            SubstrateSpec(normal_hkl=(0, 0, 1), gamma_int=0.10, gamma_sub=1.30),
        "Bagnamento parziale (0 < Delta_gamma < gamma_film)":
            SubstrateSpec(normal_hkl=(0, 0, 1), gamma_int=0.70, gamma_sub=1.00),
        "Adesione sfavorita (Delta_gamma < 0: il cristallo si alza)":
            SubstrateSpec(normal_hkl=(0, 0, 1), gamma_int=1.40, gamma_sub=0.20),
    }

    print("Famiglie cristallografiche (cristallo libero di riferimento):")
    for fam in families:
        print(f"  {fam.hkl}: γ = {fam.gamma}")

    for title, substrate in regimes.items():
        result = build_wulff_polyhedron(families, k=k, substrate=substrate)

        # Ricalcolo qui (stessa formula di build_planes/_winterbottom_
        # contact_offset) solo per poter DISEGNARE il piano del
        # substrato alla sua posizione reale, anche quando non compare
        # tra le facce del poliedro finale.
        n_contact = -get_unit_normal(substrate.normal_hkl)
        d_contact = None
        for fam in families:
            for hkl in expand_cubic_family(fam.hkl):
                if np.dot(get_unit_normal(hkl), n_contact) > 1 - 1e-8:
                    gamma_contact_used = substrate.gamma_contact or fam.gamma
                    d_contact = _winterbottom_contact_offset(k, gamma_contact_used, substrate)
                    d_natural = k * fam.gamma
                    break
            if d_contact is not None:
                break

        delta_gamma = substrate.gamma_sub - substrate.gamma_int
        print(f"\n--- {title} ---")
        print(f"  Delta_gamma = γ_sub - γ_int = {delta_gamma:.3f}")
        print(f"  d_naturale (faccia libera, k·γ_film) = {d_natural:.3f}")
        print(f"  d_contatto (Winterbottom)             = {d_contact:.3f}  "
              f"({'più vicino: bagna' if d_contact < d_natural else 'più lontano: si stacca, ridondante'})")
        print(f"  Piani totali (dopo espansione + substrato): {len(result['normals'])}")
        print(f"  Vertici del poliedro: {len(result['vertices'])}")
        print(f"  Facce nel poliedro finale: {len(result['faces'])}")
        contact_labels = [fam.label for fam in result["face_family"]
                           if fam.hkl == substrate.normal_hkl]
        print(f"  Faccia di contatto presente nel poliedro finale: "
              f"{'sì (' + contact_labels[0] + ')' if contact_labels else 'no (piano ridondante: forma = Wulff libero)'}")

        fig, ax, check = plot_wulff_shape(
            result,
            show_normals=False,
            show_center=True,
            show_reference_sphere=False,
            title=f"Winterbottom shape — {title}",
            substrate_plane=n_contact,
        )

    plt.show()