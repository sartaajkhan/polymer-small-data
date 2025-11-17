import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
#from mordred import Calculator, descriptors
import networkx as nx

def smiles_to_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    G = nx.Graph()
    labels = {}

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        sym = atom.GetSymbol()
        labels[idx] = f"{idx}:{sym}"
        G.add_node(idx, symbol=sym)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        order = bond.GetBondTypeAsDouble()
        G.add_edge(i, j, order=order)
    
    return G

class Featurizer:
    def __init__(self, smi : str):
        self.smi = smi
        self.mol = Chem.MolFromSmiles(self.smi)
        self.G = smiles_to_graph(self.smi)
    
    def RDKit_descriptors(self):
        mol = Chem.MolFromSmiles(self.smi)
        descriptor_names = [desc_name for desc_name, _ in Descriptors._descList]
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

        desc_values = calculator.CalcDescriptors(mol)
        desc_dict = dict(zip(descriptor_names, desc_values))

        return desc_dict # {RDkit feature : value}
    

    def balaban_index(self):
        if not nx.is_connected(self.G):
            raise ValueError("balaban index is only for connected graphs")
        
        n, m = self.G.number_of_nodes(), self.G.number_of_edges()
        all_pairs_of_distance = dict(nx.all_pairs_shortest_path_length(self.G))
        w = {u : sum(all_pairs_of_distance[u].values()) for u in self.G.nodes()}

        s = 0
        for u, v in self.G.edges():
            s = s + (np.sqrt(w[u] * w[v]))**(-1)
        
        if s == 0:
            return np.nan
        
        constant_outside_summation = (m)/(m - n + 2)
        return constant_outside_summation * s
    
    def wiener_index(self):
        return nx.wiener_index(self.G) #should we go unweighted graph? should be fine, unless there are very heavy functional groups with a very high covalent distance (where... bond will be broken anyways)
    
    def hyper_wiener_index(self):
        W_H = 0
        for u, v in self.G.edges():
            W_H += (self.G.degree[u] + self.G.degree[v]) ** 2
        
        return W_H
    
    def kappa_indices(self):
        def count_3_paths(G):
            count = 0
            for u in G.nodes():
                for v in G.neighbors(u):
                    for w in G.neighbors(v):
                        if w != u:
                            for x in G.neighbors(w):
                                # Ensure all nodes are distinct
                                if len({u, v, w, x}) == 4:
                                    count += 1
            return count // 2
        
        if not nx.is_connected(self.G):
            raise ValueError("only for connected graphs...")
        
        n, m = self.G.number_of_nodes(), self.G.number_of_edges()
        if n <= 3 or m == 0: #can't imagine a triangle being a polymer and a polymer with no edges...
            return {"kappa1" : np.nan, "kappa2" : np.nan, "kappa3" : np.nan}
        
        P = count_3_paths(self.G)

        k1 = m/(n - 2)
        k2 = ((n - 1) * (n - 2)) / (2 * m)
        k3 = ((n - 1) * (n - 2)**2)/P if P > 0 else np.nan

        return {"kappa1" : k1, "kappa2" : k2, "kappa3" : k3}
    
    def spectral_radius(self):
        """
        included in here just because from my work, it correlates a lot with density :)
        """
        import warnings
        warnings.simplefilter('ignore') #because adjacency matrix raises concerns

        A = nx.adjacency_matrix(self.G).toarray()
        eigs, _ = np.linalg.eigh(A)
        return np.max(eigs)
    
    def number_of_cycles(self):
        return len(nx.cycle_basis(self.G))
    
    def number_of_stars(self):
        return self.smi.count("*")
    
    def diameter_of_graph(self):
        """
        is a decent approximation of length of backbone for polymer
        """
        return nx.diameter(self.G)
    
    def number_of_branches(self):
        """
        for atom to be part of a branch, its degree must be 1 (end of monomer) or 2 (middle of monomer)
        """
        n = self.G.number_of_nodes()

        number_of_branches = 0
        for v in self.G.nodes():
            deg_v = self.G.degree[v]
            number_of_branches += np.max([0, deg_v - 2])
        
        return number_of_branches
    
    def summary_of_results(self):
        rdkit_descriptors = self.RDKit_descriptors()
        kappa_indices = self.kappa_indices()

        topo_results = {"balaban_index" : self.balaban_index(),
                        "wiener_index" : self.wiener_index(),
                        "hyper_wiener_index" : self.hyper_wiener_index(),
                        "kappa1" : kappa_indices["kappa1"],
                        "kappa2" : kappa_indices["kappa2"],
                        "kappa3" : kappa_indices["kappa3"],
                        "spectral_radius" : self.spectral_radius()
                        }
        
        other_physical_traits = {"number_of_cycles" : self.number_of_cycles(),
                                 "number_of_stars" : self.number_of_stars(),
                                 "diameter" : self.diameter_of_graph(),
                                 "number_of_branches" : self.number_of_branches()
                                 }
        
        merged_results = {**rdkit_descriptors, **topo_results, **other_physical_traits}
        return merged_results