import abc
import functools

import numpy as np

from qecsim import paulitools as pt
from qecsim.model import ErrorModel, cli_description


class INIDErrorModel(ErrorModel):
    """
    Implements an independent and non-identically distributed (i.ni.d.) error model, where every qubit experiences
    a different probability of suffering a Pauli error dependent on its individual relaxation and dephasing times T_1 and T_2.
    """
    def __init__(self, T_1, T_2):
        """
        Initialise new i.ni.d. error model using the given a list of relaxation and dephasing times T_1 and T_2 for each qubit.
        
        :param T_1: List of relaxation times T_1 for each qubit.
        :type T_1: list of float
        
        :param T_2: List of dephasing times T_2 for each qubit.
        :type T_2: list of float
        
        :raises ValueError: if T_1 or T_2 are not lists of floats.
        :raises ValueError: if t is not a float.
        """
        if not all(isinstance(x, float) for x in T_1):
            raise ValueError("T_1 must be a list of floats.")
        if not all(isinstance(x, float) for x in T_2):
            raise ValueError("T_2 must be a list of floats.")
        self._T_1 = T_1
        self._T_2 = T_2
    
    @functools.lru_cache()
    def probability_distribution(self, t):
        """
        Return the probability distributions amongst Pauli I, X, Y and Z errors for each qubit according to their individual
        relaxation and dephasing times T_1 and T_2.

        :param t: Time related to the overall probability of an error on a single qubit.
        :type t: float
        :return: Tuple of arrays of probability distributions for Pauli I, X, Y and Z errors for each qubit.
        :rtype: 4-tuple of arrays
        :raises NotImplementedError: Unless implemented in a subclass.
        """
        # for each qubit n, calculate the probability of each Pauli error
        # p_x = 1/4 * (1 - np.exp(-t / self._T_2))
        # p_y = p_x
        # p_z = 1/4 * (1 + np.exp(-t / self._T_1) - 2*np.exp(-t / self._T_2))
        # p_i = 1 - p_x - p_y - p_z
        
        gamma = 1 - np.exp(-t / self._T_1)
        lambda_ = 1 - np.exp(-2*t / self._T_2)
        
        p_x = gamma / 4
        p_y = p_x
        p_z = (2 - gamma - 2*np.sqrt( 1 - gamma - (1-gamma)*lambda_ )) / 4
        p_i = 1 - p_x - p_y - p_z
                
        return (p_i, p_x, p_y, p_z)
        
    def generate(self, code, t, rng=None):
        """
        Generate new error.

        :param code: Stabilizer code.
        :type code: StabilizerCode
        :param t: Time related to the overall probability of an error on a single qubit.
        :type t: float
        :param rng: Random number generator. (default=None resolves to numpy.random.default_rng())
        :type rng: numpy.random.Generator
        :return: New error as binary symplectic vector.
        :rtype: numpy.array (1d)
        """
        rng = np.random.default_rng() if rng is None else rng
        n_qubits = code.n_k_d[0]
        p_i, p_x, p_y, p_z = self.probability_distribution(t)
        
        error_pauli = ''.join([rng.choice(('I', 'X', 'Y', 'Z'), p=(p_i[n], p_x[n], p_y[n], p_z[n])) for n in range(n_qubits)])       
        
        return pt.pauli_to_bsf(error_pauli)


    @property
    # @abc.abstractmethod
    def label(self):
        """
        Label suitable for use in plots.

        :rtype: str
        """
        return "i.ni.d. error model"
    
    def __repr__(self):
        return '{}()'.format(type(self).__name__)
    