import functools
import itertools
from scipy.spatial import distance

from qecsim import graphtools as gt
from qecsim.model import Decoder, cli_description


@cli_description('RotatedPlanarMWPM')
class RotatedPlanarMWPMDecoder(Decoder):
    """
    Implements a Minimum Weight Perfect Matching (MWPM) decoder for rotated planar code.

    Decoding algorithm:

    * The syndrome is resolved to plaquettes defects using:
      :meth:`qecsim.models.planar.PlanarCode.syndrome_to_plaquette_indices`.
    * For each defect the nearest off-boundary plaquette defect is added using:
      :meth:`qecsim.models.rotatedplanar.RotatedPlanarCode._virtual_plaquette_indices`.
    * If the total number of defects is odd an extra virtual off-boundary defect is added.
    * A graph between plaquettes is built with weights given by: :meth:`distance`.
    * A MWPM algorithm is used to match plaquettes into pairs.
    * A recovery operator is constructed by applying the shortest path between matching plaquette pairs using:
      :meth:`_path_operator` and returned.
    """
    def __init__(self):
        """
        Initialise new rotated planar MWPM decoder.
        """
        
        # Add extra virtual node if odd number of total nodes
        self._extra_virtual_x_plaquette = (-9,-10)
        self._extra_virtual_z_plaquette = (-10,-10)
        
    @property
    def extra_virtual_x_plaquette(self):
        """
        Index of the extra virtual x plaquette for the case of an odd number of total nodes in the decoding process.

        :rtype: 2-tuple of int
        """
        return self._extra_virtual_x_plaquette
    
    @property
    def extra_virtual_z_plaquette(self):
        """
        Index of the extra virtual z plaquette for the case of an odd number of total nodes in the decoding process.

        :rtype: 2-tuple of int
        """
        return self._extra_virtual_z_plaquette

    @functools.lru_cache(maxsize=2 ** 28)  # for MxN lattice, cache_size <~ 2(MN)(MN-1) so handle 100x100 codes.
    def distance(self, a_index, b_index):
        """
        This function calculates the weight of edges between certain plaquettes based on the number of qubits on the path.
        """
        return distance.chebyshev(a_index, b_index) # Chessboard distance, diagonal moves are allowed and have a distance of 1.

    def _create_subgraph(self, code, syndrome_plaquettes):
        """Create a subgraph for decoding from either X-type or Z-type syndrom plaquette indices.

        :param code: Rotated Planar code.
        :type code: RotatedPlanarCode
        :param syndrome_plaquettes: Syndrome plaquette indices.
        :type syndrome_plaquettes: list of 2-tuple of int
        :return: Subgraph.
        :rtype: gt.SimpleGraph
        """
            
        # check if all of the syndrome plaquettes are of the same type
        if not (all([code.is_x_plaquette(index) for index in syndrome_plaquettes]) or all([code.is_z_plaquette(index) for index in syndrome_plaquettes])):
            raise ValueError("The syndrome plaquette indices should all be of the same type either X or Z.")
            
        graph = gt.SimpleGraph()

        # prepare virtual nodes
        vindices = set()

        # Add an edge between each syndrome plaquette and its closest virtual plaquette            
        for index in syndrome_plaquettes:
            closest_virtual_plaquette = code.closest_virtual_plaquette(index)
            vindices.add(closest_virtual_plaquette)
            graph.add_edge(index, closest_virtual_plaquette, weight=self.distance(index, closest_virtual_plaquette)) 

        # Add extra virtual node if odd number of total nodes
        if (len(syndrome_plaquettes) + len(vindices)) % 2:
            vindices.add(self._extra_virtual_x_plaquette if code.is_x_plaquette(syndrome_plaquettes[0]) else self._extra_virtual_z_plaquette) 
            
        # Add an edge between all syndrome plaquettes
        for a_index, b_index in itertools.combinations(syndrome_plaquettes, 2): # iterates over all pairs of indices of the syndrome plaquettes
            graph.add_edge(a_index, b_index, weight=self.distance(a_index, b_index))

        # Add zero weight edges between all virtual nodes
        for a_index, b_index in itertools.combinations(vindices, 2): # iterates over all pairs of indices of the virtual plaquettes
            graph.add_edge(a_index, b_index, 0)
            
        return graph
    
    def _path_operator(self, code, a_index, b_index):
        """
        Operator consisting of a path of Pauli operators to fuse the syndrome plaquettes indexed by A and B.
        Used to obtain the recovery operator based on the syndrome plaquettes that were matched by the MWPM algorithm.
        
        Assumptions:

        * All indices are within the (virtual) plaquette bounds.
        * A and B are of the same type (X or Z).
        * Either A or B is not a virtual plaquette.

        :param code: Rotated planar code.
        :type code: RotatedPlanarCode
        :param a_index: Plaquette index as (x, y).
        :type a_index: (int, int)
        :param b_index: Plaquette index as (x, y).
        :type b_index: (int, int)
        :return: Path operator in binary symplectic form.
        :rtype: numpy.array (1d)
        :raises ValueError: If plaquettes are not of the same type (i.e. X or Z).
        """
        # assumption checks
        assert code.is_in_plaquette_bounds(a_index) or code.is_virtual_plaquette(a_index)
        assert code.is_in_plaquette_bounds(b_index) or code.is_virtual_plaquette(b_index)

        # check both plaquettes are the same type
        if code.is_z_plaquette(a_index) != code.is_z_plaquette(b_index):
            raise ValueError('Path undefined between plaquettes of different types: {}, {}.'.format(a_index, b_index))
        
        # check if both plaquettes are virtual plaquettes
        if code.is_virtual_plaquette(a_index) and code.is_virtual_plaquette(b_index):
            raise ValueError('Path between two virtual plaquettes is not valid: {}, {}.'.format(a_index, b_index))

        def _start_end_site_coordinate(a_k, b_k):
            """Return start and end coordinates along an axis, where k represents x or y."""
            if a_k < b_k:  # A below B so go from top of A to bottom of B
                start_k = a_k + 1
                end_k = b_k
            elif a_k > b_k:  # A above B so go from bottom of A to top of B
                start_k = a_k
                end_k = b_k + 1
            else:  # A in line with B so go from bottom(top) of A to bottom(top) of B (if k below zero)
                start_k = end_k = max(b_k, 0)
            return start_k, end_k

        # if start and end plaquette indices are the same return identity operator
        if a_index == b_index:
            return code.new_pauli().to_bsf()

        # start and end plaquette indices (Note plaquettes are indexed by their SW corner)
        a_x, a_y = a_index
        b_x, b_y = b_index
        # determine start and end site indices
        start_x, end_x = _start_end_site_coordinate(a_x, b_x)
        start_y, end_y = _start_end_site_coordinate(a_y, b_y)
        # build path (diagonal until inline then straight up/down or left/right)
        path_indices = []
        next_x, next_y = start_x, start_y
        while True:
            # add next_index to path
            path_indices.append((next_x, next_y))
            # test if we got to end
            if (next_x, next_y) == (end_x, end_y):
                break  # we are at the end so stop
            # increment/decrement next_x and/or next_y
            if end_x - next_x > 0:
                next_x += 1
            elif end_x - next_x < 0:
                next_x -= 1
            if end_y - next_y > 0:
                next_y += 1
            elif end_y - next_y < 0:
                next_y -= 1
        # single pauli op
        op = 'X' if code.is_z_plaquette(a_index) else 'Z'
        # full path operator
        path_operator = code.new_pauli().site(op, *path_indices)
        # return as bsf
        return path_operator.to_bsf()
    
    def decode(self, code, syndrome, **kwargs):
        """
        Returns a recovery operator for the rotated planar code using the MWPM algorithm.
        
        :param code: Rotated planar code.
        :type code: RotatedPlanarCode
        :param syndrome: Syndrome as binary vector.
        :type syndrome: numpy.array (1d)
        :return: Recovery operation as binary symplectic form.
        :rtype: numpy.array (1d)
        """
        # split syndrome into X and Z type
        x_syndrome_plaquettes, z_syndrome_plaquettes = [], []
        for plaquette_index in code.syndrome_to_plaquette_indices(syndrome):
            x_syndrome_plaquettes.append(plaquette_index) if code.is_x_plaquette(plaquette_index) else z_syndrome_plaquettes.append(plaquette_index)
        
        # create subgraphs for X and Z syndromes    
        x_subgraph = self._create_subgraph(code, x_syndrome_plaquettes)
        z_subgraph = self._create_subgraph(code, z_syndrome_plaquettes)
        
        # Find MWPM edges for each subgraph: output is a set of matched plaquette indices e.g.: {((2, 3), (1, 4)), ((3, 4), (4, 5)), ((1, 6), (2, 5))}
        mwpm_x_subgraph = gt.mwpm(x_subgraph)
        mwpm_z_subgraph = gt.mwpm(z_subgraph)
        
        extra_virtual_plaquettes = (self._extra_virtual_x_plaquette, self._extra_virtual_z_plaquette)
        
        recovery = code.new_pauli().to_bsf()
        for matching_pairs in [mwpm_x_subgraph, mwpm_z_subgraph]:
            for a_index, b_index in matching_pairs: # e.g.: a_index=(2, 3), b_index=(1, 4) in the above example
                # Ignore the extra virtual plaquettes, that are used in order to enable perfect matching in the case of odd number of nodes
                if (a_index in extra_virtual_plaquettes or b_index in extra_virtual_plaquettes):
                    continue
                # Ignore the matching of two virtual plaquettes
                if code.is_virtual_plaquette(a_index) and code.is_virtual_plaquette(b_index):
                    continue
                recovery ^= self._path_operator(code, a_index, b_index)
        return recovery

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        return 'Rotated Planar MWPM'

    def __repr__(self):
        return '{}()'.format(type(self).__name__)
