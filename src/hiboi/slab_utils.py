from ase import Atoms
from ase.build import fcc111,fcc110,fcc100
import numpy as np
from ase.constraints import FixAtoms

fcc_functions = {
    'FCC111': fcc111,
    'FCC110': fcc110,
    'FCC100': fcc100,
}

def slab_generator(composition: list[str], 
                   facet: str,
                   slab_dim: tuple[int,int,int],
                   surface_adj: bool = True,
                   fix_bottom: int =2,
                   pbc: tuple[bool,bool,bool] = (True,True,False),
                   vac: float = 10,
                   lat_param: float|None = None):
        '''Generic slab generator'''
        
        facet = facet.upper()
        if facet not in {'FCC111', 'FCC110', 'FCC100'}:
            raise ValueError(f'Unsupported facet: {facet}')


        #Make Template slab
        if lat_param is None:
            lat_param=np.mean([lat_param_dict[ele] for ele in composition])
        fcc = fcc_functions[facet]
        slab=fcc('Cu',slab_dim,lat_param,vac)
        slab.set_pbc(pbc) 

        #Making alloy composition
        symbols=np.random.choice(composition,np.prod(slab_dim)) #Drawing rnd symbols
        slab.set_chemical_symbols(symbols)

        if surface_adj: #Adjusting lateral lattice vectors by surface composition, see doi.org/10.1007/s12274-021-3544-3
            surf_lat = np.mean([lat_param_dict[ele] for ele in symbols[-slab_dim[0] * slab_dim[1]:]])
            new_cell=slab.get_cell()*np.array([[surf_lat/lat_param,surf_lat/lat_param,1]])
            slab.set_cell(new_cell,scale_atoms=True)

        c = FixAtoms([atom.index for atom in slab if atom.tag>slab_dim[2]-fix_bottom]) #Fixes the number of bottom of layers as specified by fix_bottom
        slab.set_constraint(c)

        return slab


ads_dict = {
                  #Monatomic/linear adsorbates
                  'O':Atoms('O', ([0, 0, 0],)),
                  'N':Atoms('N', ([0, 0, 0],)),
                  'F':Atoms('F', ([0, 0, 0],)),
                  'HF':Atoms('FH', ([0, 0, 0],[0.9, 0, 0.1])),
                  'CO':Atoms('CO', ([0, 0, 0], [0, 0, 1.16])),
                  'OH':Atoms('OH', ([0, 0, 0], [0.65, 0.65, 0.40])),

                  #Additional Hydroxyl orientations
                  'OH_deg0':Atoms('OH', ([0, 0, 0], [0.92, 0.01, 0.40])),
                  'OH_deg30':Atoms('OH', ([0, 0, 0], [0.80, 0.46, 0.40])),
                  'OH_deg60':Atoms('OH', ([0, 0, 0], [0.46, 0.8, 0.40])),
                  'OH_mirror':Atoms('OH', ([0, 0, 0], [-0.65, 0.65, 0.40])),
                  'OH_deg0mir':Atoms('OH', ([0, 0, 0], [-0.92, 0.01, 0.40])),
                  
                  #Different water orientations
                  'H2O_lying':Atoms('OHH', ([0,0,0],[0.592,0.797,0.041],[0.592,-0.797,0.041])),
                  'H2O_deg0':Atoms('OHH', ([0,0,0],[0.99,0.0,0.041],[-0.26,0.96,0.041])),
                  'H2O_standing':Atoms('OHH', ([0,0,0],[0,0.,1],[0.992,0,-0.204])),
                  'mirH2O_deg0':Atoms('OHH', ([0,0,0],[0.99,0.0,0.041],[-0.26,-0.96,0.041])),
                  'mirH2O_lying':Atoms('OHH', ([0,0,0],[-0.592,0.797,0.041],[-0.592,-0.797,0.041])),
                  'H2O_lyingmir':Atoms('OHH', ([0,0,0],[-0.592,0.797,0.041],[-0.592,-0.797,0.041])),

                  #Ammonia and differently oriented NH2 adsorbates                  
                  'NH3':Atoms('NHHH',([0,0,0],[1.02,-0.048,0.25],[-0.6,-0.83,0.22],[-0.54,0.855,0.250])),
                  'NH2_lying':Atoms('NHH', ([0,0,0],[0.592,0.797,0.041],[0.592,-0.797,0.041])),
                  'NH2_lying60':Atoms('NHH', ([0,0,0],[0.114,0.986,0.041],[0.911,-0.394,0.041])),
                  'NH2_lying120':Atoms('NHH', ([0,0,0],[-0.986, 0.114, 0.041],[ 0.394, 0.911, 0.041])),
                  'NH2_lying180':Atoms('NHH', ([0,0,0],[-0.592,-0.797,0.041],[-0.592,0.797,0.041])),
                  'NH2_lying240':Atoms('NHH', ([0,0,0],[-0.986, -0.114, 0.041],[ 0.394, -0.911, 0.041])),
                  'NH2_lying300':Atoms('NHH', ([0,0,0],[0.911,0.394,0.041],[0.114,-0.986,0.041])),
                  'NH2_standing':Atoms('NHH', ([0,0,0],[0,0.,1],[0.992,0,-0.204])),

                  #Additional multidentate oxyanions
                  "SO4":Atoms('OSOOO',([0.2,0,0],[1.4,0.5,0.5],[2.7,-0.1,0.2],[1.5,2.,0.2],[1.2,0.4,2.1])),
                  "HSO4":Atoms('OSOOOH',([0.2,0,0],[1.4,0.5,0.5],[2.7,-0.1,0.2],[1.5,2.,0.2],[1.2,0.4,2.1],[1.2,0.4,3.1])),
                  "HPO4":Atoms('OPOOOH',([0,0,0],[1.384,0.703,0.385],[2.666,0,0.062],[1.344,2.197,0.01],[1.366,0.693,2.02],[1.366,-0.24,2.36])),
                  "CO3":Atoms('OCOO',([0,0,0],[0.550, 0.953, 0.700],[1.080, 1.871, 0.000],[0.550, 0.953, 2.000])),
                  "NO3":Atoms('ONOO',([0,0,0],[0.550, 0.953, 0.600],[1.070, 1.853, 0.000],[0.550, 0.953, 1.800])),
                  "HCO3":Atoms('OCOOH',([0,0,0],[0.550, 0.953, 0.700],[1.080, 1.871, 0.000],[0.550, 0.953, 2.000],[1.000, 1.732, 2.200])),

                  #Adsorbates with higher atomic numbers (for non-close-packed planes)
                  "H2S":Atoms('SHH', ([0,0,0],[0.9,0.97,0.36],[0.9,-0.97,0.3])),
                  "H2S_deg0":Atoms('SHH', ([0,0,0],[0,1.32,0.36],[1.32,0,0.3])),
                  "HS":Atoms('SH', ([0,0,0],[0,1.32,0.36])),
                  'Cl':Atoms('Cl', ([0, 0, 0],)),
                  'HCl':Atoms('ClH', ([0, 0, 0],[1.35,0,0.2])),
                  'HBr':Atoms('BrH', ([0, 0, 0],[1.45,0,0.2])),
                  'Br':Atoms('Br', ([0, 0, 0],)),
                  }

lat_param_dict = {'Ag':4.2113, #In units of Ångström, as provided in https://doi.org/10.1007/s44210-022-00006-4
                  'Al': 4.0674,
                  'Au': 4.2149,
                  'B': 2.8822,
                  'Be': 3.2022,
                  'Bi': 5.0699,
                  'Cd': 4.5795,
                  'Co': 3.5625,
                  'Cr': 3.6466,
                  'Cu': 3.6901,
                  'Fe': 3.6951,
                  'Ga': 4.2817,
                  'Ge': 4.304,
                  'Hf': 4.5321,
                  'In': 4.8536,
                  'Ir': 3.8841,
                  'Mg': 4.5673,
                  'Mn': 3.5371,
                  'Mo': 4.0204,
                  'Nb': 4.2378,
                  'Ni': 3.565,
                  'Os': 3.8645,
                  'Pb': 5.0942,
                  'Pd': 3.9814,
                  'Pt': 3.9936,
                  'Re': 3.9293,
                  'Rh': 3.8648,
                  'Ru': 3.8285,
                  'Sc': 4.6809,
                  'Si': 3.8935,
                  'Sn': 4.8612,
                  'Ta': 4.2504,
                  'Ti': 4.158,
                  'Tl': 5.0884,
                  'V': 3.8573,
                  'W': 4.0543,
                  'Y': 5.1289,
                  'Zn': 3.9871,
                  'Zr': 4.562}
