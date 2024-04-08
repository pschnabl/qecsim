import functools
import numpy as np
import networkx as nx
from qecsim import graphtools as gt
from qecsim.model import cli_description
from qecsim.models.rotatedplanar import RotatedPlanarMWPMDecoder

@cli_description('RotatedPlanarRMWPM')
class RotatedPlanarReweightedMWPMDecoder(RotatedPlanarMWPMDecoder):
    """
    Implements a reweighted minimum weight perfect matching (rMWPM) decoder for the rotated planar code.
    The edges between plaquettes in the Z and X check graphs are reweighted according to the T1 and T2
    times of the corresponding qubits respectively.    
    """
    # TODO: check the condition T_2 <= 2*T_1
    def __init__(self, code, t, T_1, T_2):
        """
        Initialise decoder for the rotated planar code with time t, and the T_1 and T_2 times of each qubit.
        
        :param code: RotatedPlanarCode
        :type code: RotatedPlanarCode
        :param t: time
        :type t: float
        :param T_1: T_1 times of the qubits
        :type T_1: list of float
        :param T_2: T_2 times of the qubits
        :type T_2: list of float        
        """
        super().__init__()
        self.code = code
        self.t = t
        self.T_1 = T_1
        self.T_2 = T_2
        self.g_z, self.g_x = self.create_check_graphs(code)        
        
    def create_check_graphs(self, code):
        """
        Creates the graphs for the X-checks and Z-checks (see Fig. 6 of https://arxiv.org/pdf/2307.14989.pdf%C3%A7). 
        The edges conncect neighboring plaquettes of the same type and they resemble the qubits between them.
        The weights of the edges are determined by the T_1 and T_2 times of the corresponding qubits.
        
        :param code: RotatedPlanarCode
        :type code: RotatedPlanarCode
        :param plaquettes: list of plaquette indices
        :type plaquettes: list of 2-tuple of int
        :return: graph of checks
        :rtype: nx.Graph
        """
        # create a lists of all the X and Z plaquettes including the virtual plaquettes
        plaquette_indices = code._plaquette_indices
        virtual_z_plaquettes, virtual_x_plaquettes = code._virtual_plaquette_indices
        z_plaquettes = [index for index in plaquette_indices if code.is_z_plaquette(index)] + virtual_z_plaquettes
        x_plaquettes = [index for index in plaquette_indices if code.is_x_plaquette(index)] + virtual_x_plaquettes
        
        # Based on the T_1 and T_2 times of each qubit define the weights of the edges corresponding to the respective qubit 
        weight_T_1 = np.abs(np.log(1-np.exp(-self.t/self.T_1)))
        weight_T_2 = np.abs(np.log(1-np.exp(-self.t/self.T_2)))
        
        # create the graphs for X and Z checks: see row 2 in Fig. 6 of https://arxiv.org/pdf/2307.14989.pdf%C3%A7
        g_x = nx.Graph()
        g_z = nx.Graph()
        max_site_x, max_site_y = code.site_bounds
        for y in range(-1, max_site_y+1):
            for x in range(-1, max_site_x+1):
                # get the index of the current plaquette and the neighboring plaquettes in the row above
                index = (x, y) # current plaquette
                left_up = (x-1, y+1) # plaquette one step up and one step left from the current plaquette
                right_up = (x+1, y+1) # plaquette one step up and one step right from the current plaquette
                
                # the X plaquettes detect Z and Y errors, so the weights are based on T_2 times, since p_z + p_y only depends on T_2
                # see https://arxiv.org/pdf/2203.15695.pdf Eq. (2) for the definition of p_x, p_y, p_z  
                if index in x_plaquettes:
                    if left_up in x_plaquettes:
                        qubit_number = x + (y+1)*(max_site_x+1)
                        g_x.add_edge(index, left_up, weight=weight_T_2[qubit_number]) # other possible attributes: qubit_index=(x,y+1), qubit_number=qubit_number,
                    if right_up in x_plaquettes: 
                        qubit_number = right_up[0] + right_up[1]*(max_site_x+1)
                        g_x.add_edge(index, right_up, weight=weight_T_2[qubit_number]) # qubit_index=right_up, qubit_number=qubit_number,
                
                # the Z plaquettes detect X and Y errors, so the weights are based on T_1 times, since p_x + p_y only depends on T_1       
                elif index in z_plaquettes:
                    if left_up in z_plaquettes:
                        qubit_number = x + (y+1)*(max_site_x+1) 
                        g_z.add_edge(index, left_up, weight=weight_T_1[qubit_number]) # qubit_index=(x,y+1), qubit_number=qubit_number
                    if right_up in z_plaquettes:
                        qubit_number = right_up[0] + right_up[1]*(max_site_x+1)
                        g_z.add_edge(index, right_up, weight=weight_T_1[qubit_number]) # qubit_index=right_up, qubit_number=qubit_number
        return g_z, g_x
        
    @functools.lru_cache(maxsize=2 ** 28)  # for MxN lattice, cache_size <~ 2(MN)(MN-1) so handle 100x100 codes.
    def distance(self, a_index, b_index):
        """
        Calculate the shortest weighted path length in g_x or g_z from a source plaquette to a target plaquette.
        The path length is the sum of the weights of the edges in the path.
        This path length is then used as the edge-weight in the minimum weight perfect matching (MWPM) algorithm.
        
        Overrides the distance method of RotatedPlanarMWPMDecoder.
        """
        if self.code.is_z_plaquette(a_index): # check if both plaquettes are the same type (X, Z) is done in _create_subgraph of the parent class
            return nx.dijkstra_path_length(self.g_z, a_index, b_index)
        elif self.code.is_x_plaquette(a_index):
            return nx.dijkstra_path_length(self.g_x, a_index, b_index)
        
    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        return 'Rotated Planar rMWPM'

    def __repr__(self):
        return '{}()'.format(type(self).__name__)