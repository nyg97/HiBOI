from ase.spacegroup.symmetrize import check_symmetry
from ase.io import read
import numpy as np
from scipy.optimize import linear_sum_assignment


def find_translational_symmetries(
    atoms,
    ref_index=-1,
    layer_tol=0.15,
    symprec=0.05,
    surface_only=False,
    ignore_chem_symbols = True,
):
    """
    Find pure lateral translational symmetries of a slab.

    Parameters
    ----------
    atoms : ase.Atoms
    ref_index : int
        Reference atom (0-based).
    layer_tol : float
        Tolerance for identifying atoms in the same layer (Angstrom).
    symprec : float
        Maximum matching distance (Angstrom).
    surface_only : bool
        If True, test symmetry only within the reference layer.

    Returns
    -------
    translations : ndarray
        Cartesian translation vectors (N, 3).
    fractional : ndarray
        Fractional lateral translations (N, 2).
    """

    cell = atoms.cell.array
    a, b = cell[:2]

    positions = atoms.positions
    scaled = atoms.get_scaled_positions(wrap=False)
    symbols = np.array(atoms.get_chemical_symbols())

    # Surface normal and atomic heights
    normal = np.cross(a, b)
    normal /= np.linalg.norm(normal)
    heights = positions @ normal

    # Identify reference layer
    ref_height = heights[ref_index]
    layer = np.abs(heights - ref_height) < layer_tol

    # Atoms used to generate candidate translations
    if not ignore_chem_symbols:
        candidates = np.where(
            layer & (symbols == symbols[ref_index])
        )[0]
    else:
        candidates = np.where(layer)[0]

    # Atoms included in symmetry verification
    selected = np.where(layer)[0] if surface_only else np.arange(len(atoms))

    translations = []
    fractional = []

    for j in candidates:

        # Candidate fractional translation
        t = scaled[j, :2] - scaled[ref_index, :2]
        t %= 1.0

        # Cartesian translation
        t_cart = t[0] * a + t[1] * b

        # Calculate distances between translated atoms and targets
        diff = (
            scaled[selected][None, :, :]
            - (scaled[selected][:, None, :] + np.array([*t, 0]))
        )

        # Apply lateral periodic boundary conditions only
        diff[:, :, :2] -= np.round(diff[:, :, :2])

        distances = np.linalg.norm(diff @ cell, axis=2)

        # Disallow matching different chemical elements
        if not ignore_chem_symbols:
            mismatch = symbols[selected][:, None] != symbols[selected][None, :]
            distances[mismatch] = 1e6

        # Find one-to-one atom mapping
        row, col = linear_sum_assignment(distances)

        if np.max(distances[row, col]) < symprec:

            # Avoid duplicate translations
            if not any(
                np.linalg.norm(
                    (t - old + 0.5) % 1.0 - 0.5
                ) < 1e-6
                for old in fractional
            ):
                translations.append(t_cart)
                fractional.append(t)

    return np.array(translations), np.array(fractional)


# translations, fractional = find_translational_symmetries(
#     atoms,
#     ref_index=8,
#     surface_only=True,
# )

# print(translations)
# print(fractional)
