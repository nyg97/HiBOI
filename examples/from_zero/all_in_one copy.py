
"""Intended use with UMA.
Generate coadsorbate structures and infer bond-order descriptors."""

from datetime import datetime
from pathlib import Path

import numpy as np
from ase.calculators.emt import EMT
from ase.build import add_adsorbate
from ase.constraints import FixedLine
from sklearn.linear_model import LinearRegression

import sys
sys.path.append('../../src/hiboi/') #Until package has been developed
from slab_utils import slab_generator, ads_dict
import matplotlib.pyplot as plt
from ase.data import chemical_symbols
from ase.data.colors import jmol_colors

jmol_dict = dict(zip(chemical_symbols, jmol_colors))


# Settings
TEST = True
ELEMENTS = ["Pt", "Pd", "Au", "Ag"]
SLAB_DIM = (4, 4, 5)
N_SLABS = 4
SEED = 42
CALCULATOR = EMT()

np.random.seed(SEED)

# Output directories
output_dir = Path("test" if TEST else datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
(output_dir / "coadsorbates").mkdir(parents=True, exist_ok=True)
(output_dir / "descriptors").mkdir(parents=True, exist_ok=True)


def generate_structures(elements, slab_dim, n_slabs):
    """Generate slabs, coadsorbates, and individual adsorbate descriptors."""

    structures = []
    colors = []
    for _ in range(n_slabs):
        slab = slab_generator(elements, "FCC111", slab_dim)

        for i in range(slab_dim[0]):
            for j in range(slab_dim[1]):
                # Individual adsorbate descriptors
                A = slab.copy()
                add_adsorbate(A, ads_dict["OH"], 2.0, offset=(i, j))
                A.constraints.append(FixedLine(-2, [0, 0, 1]))

                B = slab.copy()
                add_adsorbate(
                    B, ads_dict["OH"], 2.0,
                    offset=((i + 1) % slab_dim[0], j),
                )
                B.constraints.append(FixedLine(-2, [0, 0, 1]))

                # Coadsorbate structure
                coads = slab.copy()
                add_adsorbate(coads, ads_dict["H2O_deg0"], 2.3, offset=(i, j))
                add_adsorbate(
                    coads, ads_dict["OH"], 2.1,
                    offset=((i + 1) % slab_dim[0], j),
                )
                OH_ens = slab[np.prod(slab_dim) - np.prod(slab_dim[:2]) + (i + 1) % slab_dim[0] + j*3].symbol
                colors.append(jmol_dict[OH_ens])
                structures.append((slab, coads, A, B))

    return structures,colors


def get_energies(structures,calc=CALCULATOR):
    """Calculate reference-subtracted adsorption energies."""

    energies = []

    # Pure metal reference
    reference = [atoms.copy() for atoms in structures[0]]

    for atoms in reference:
        atoms.set_chemical_symbols([
            "Pt" if atom.tag != 0 else atom.symbol
            for atom in atoms
        ])
        atoms.calc = calc

    reference_energies = [atoms.get_potential_energy() for atoms in reference]

    for structures_i in structures[1:]:
        calculated = []

        for atoms in structures_i:
            atoms.calc = calc
            calculated.append(atoms.get_potential_energy())

        slab, coads, A, B = calculated
        slab_ref, coads_ref, A_ref, B_ref = reference_energies

        y = coads - coads_ref - (slab - slab_ref)
        x = [
            A - A_ref - (slab - slab_ref),
            B - B_ref - (slab - slab_ref),
        ]

        energies.append((x, y))

    X, y = zip(*energies)
    return np.asarray(X), np.asarray(y).reshape(-1,1)


def main():
    structures, colors = generate_structures(ELEMENTS, SLAB_DIM, N_SLABS)

    X, y = get_energies(structures)
    print(colors[:2])

    model = LinearRegression()
    model.fit(X, y)


    # Create scaling plane
    xx,yy=np.meshgrid(np.arange(-0.,2.,0.1),np.arange(-0.,2.,0.1))

    zz=xx*model.coef_[0][0] +yy*model.coef_[0][1]+ model.intercept_[0]

    # Create the 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.rc('font', size=13)         # Set the label size
    plt.rc('axes', labelsize=13)

    # Set labels for the axes
    ax.set_xlabel(f'$\Delta E_{{OH,pos1}}$',fontsize=11,labelpad=-5)
    ax.xaxis.set_tick_params(pad=-5)
    ax.set_xticklabels(['','','1','','2'])

    ax.set_ylabel(f'$\Delta E_{{OH,pos2}}$',fontsize=11,labelpad=-5)
    ax.yaxis.set_tick_params(pad=-5)
    ax.set_yticklabels(['(0,0,-1)','','1','','2'])
    ax.set_zlabel(f'$\Delta E_{{*(OH-H_{{2}}O)}}$',rotation=90,fontsize=11,labelpad=-12)
    ax.zaxis.set_tick_params(pad=-5)
    ax.set_zticklabels(['','','0','','1'])

    ax.text(0.5, 2.8, 2,f'$\Delta E_{{*(OH-H_{{2}}O)}}={model.coef_[0][0]:.2f}\Delta E_{{OH,pos1}}+{model.coef_[0][1]:.2f}\Delta E_{{OH,pos2}}{model.intercept_[0]:.2f}eV$',fontsize=10)
    ax.plot_surface(xx, yy, zz,color='b',zorder=0)

    # Set the limits for the axes
    ax.set_xlim(-0.,2.)
    ax.set_ylim(-0.,2.)
    ax.set_zlim(-0.99,1)

    print(X[:,0].shape)
    print(y.shape)
    colors = np.array(colors)[1:,:]


    # Scatter plot
    ax.scatter(X[:,0].flatten(), X[:,1].flatten(), y.flatten(), s=5,c=colors,edgecolor='k',lw=0.3,alpha=1)

    # Setting the viewing angle
    ax.view_init(elev=10, azim=220)  # elev is the elevation angle in the z plane. azim is the azimuth angle in the x,y plane.
    plt.savefig('test_run.png')

if __name__ == "__main__":
    main()
