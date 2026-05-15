# Copyright 2023 Sergio A. Ortega and Daniel K. Park.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SQWLib: Simulator for quantum walks.

Sparse framework.
"""

import numpy as np
import networkx as nx
from scipy import sparse

transition_matrix_exception = 'The transition matrix is not column-stochastic. See tutorial: https://github.com/qDNA-yonsei/SQWLib/tree/main/Tutorials'

## Operators

class SparseUnitary():
    """Sparse Unitary operator model.

    Attributes:
        string: A string representing the unitary operator in an algebraic form.
        info_string: A string with the information of all the operators inside the unitary.
        class_type: Kind of class.
    """
    
    class_type = 'sparse_unitary'
    
    def __init__(self,space_data,operators=None,name=None):
        """Initializes the unitary model.

        Args:
            operators: A list of operators to create directly the unitary.
            name: Custom name for the unitary operator.
        """
        
        self.space_data = space_data
        self.string = ''
        self.info_string = 'Custom sparse unitary:'
        self.name = name
        if self.name is not None:
            self.info_string += f' {name}'
        if operators is None:  # Initialize an empty unitary.
            self.operators = []
            self.info_index = 0
        else:  # Use the list to initialize the unitary.
            self.operators = operators
            for op_index, operator in enumerate(operators):
                self.string = operator.string + self.string
                self.info_string = self.info_string + f'\n {op_index+1} - ' + operator.info_string
            self.info_index = op_index
    
    def __str__(self):
        """Function for printing the unitary string."""
        
        return self.string
    
    def info(self):
        """Print the information of the operators in the unitary model."""
        
        print(self.info_string)
    
    def append(self,operator):
        """Append a new operator class to the list of operators."""
        
        self.operators.append(operator)
        self.string = operator.string + self.string
        self.info_string += f'\n {self.info_index+1} - ' + operator.info_string
        self.info_index += 1
    
    def operate(self,state,protect=True):
        """Apply the operators over an initial state.
        
        Args:
            state: Initial vector state.
            protect: Whether to protect or no the initial state variable
        
        Returns:
            state: Final vector after the operations.
        """
        
        if protect: state = state.copy()  # Protect the initial state variable in this mode.
                
        for operator in self.operators:  # Operate over the state.
            state = operator.operate(state)
            
        return state
    
    def __mul__(self,unitary_2):
        """Multiplication of two unitary operators in an algebraic form."""
        
        if unitary_2.class_type == 'sparse_unitary':
            return SparseUnitary(self.space_data,unitary_2.operators+self.operators)
        else:
            return SparseUnitary(self.space_data,[unitary_2]+self.operators)
    
    def inverse(self):
        """Invert the unitary operator."""
        
        inverse_operator = SparseUnitary(self.space_data,[operator.inverse() for operator in self.operators][::-1],self.name)
        
        return inverse_operator

class SparseOperator():
    """Base class for sparse operator.

    Attributes:
      string: A string representing the operator in an algebraic form.
      info_string: A string with the information of the operator.
      class_type: Kind of class.
    """
    
    string = None
    info_string = None
    class_type = 'sparse_operator'
    
    def __str__(self):
        """Function for printing the operator string."""
        
        return self.string
    
    def info(self):
        """Print the information of the operator."""
        
        print(self.info_string)
    
    def operate(self,state):
        """Apply the operator over an initial state.
        
        Args:
            state: Initial vector state.
        
        Returns:
            state: Final vector after the operations.
        """
        
        return state
    
    def __mul__(self,unitary_2):
        """Multiplication of two unitary operators in an algebraic form."""
        
        return SparseUnitary(self.space_data,[self]) * unitary_2
    
    def inverse(self):
        """Invert the unitary operator."""
        
        return self

class SparseSwap(SparseOperator):
    """Sparse Swap operator.

    Attributes:
      string: A string representing the operator in an algebraic form.
      info_string: A string with the information of the operator.
      class_type: Kind of class.
    """
    
    string = 'S'
    info_string = 'Sparse Swap'
    
    def __init__(self,space_data):
        self.space_data = space_data
    
    def operate(self,state):
        """Apply the operator over an initial state.
        
        Args:
            state: Initial vector state.
        
        Returns:
            state: Final vector after the operations.
        """
        
        state = state[self.space_data.permutation]
        
        return state

class SparseOracle(SparseOperator):
    """Sparse Oracle operator.

    Attributes:
        string: A string representing the operator in an algebraic form.
        info_string: A string with the information of the operator.
        register: An intenger indicating the register for marking.
        marked_nodes: A list with the nodes to mark.
        class_type: Kind of class.
    """
    
    string = 'Q'
    info_string = 'Sparse Oracle'
    
    def __init__(self,space_data,register,marked_nodes,name=None):
        """Initializes the oracle operator.

        Args:
            register: An intenger indicating the register for marking: 1 or 2.
            marked_nodes: A list with the nodes to mark.
            name: Custom name for the oracle operator.
        """
        
        self.space_data = space_data
        self.name = name
        
        self.marked_nodes = marked_nodes
        self.register = register
        
        # Calculate the marked indexes.
        
        N = self.space_data.N
        
        mask = np.zeros(N)
        mask[self.marked_nodes] = 1
        mask = np.repeat(mask,self.space_data.sizes)
        
        if self.register == 2:  # Swap the indexes.
            mask = mask[self.space_data.permutation]
        
        self.marked_indexes = np.where(mask==1)[0]
        
        if self.name is not None:
            self.info_string += f' {name}'
        if self.register == 1: self.string = self.string + '\N{SUBSCRIPT ONE}'
        if self.register == 2: self.string = self.string + '\N{SUBSCRIPT TWO}'
        self.info_string += f': Register {register}'
        self.info_string += f': nodes {marked_nodes}'
    
    def operate(self,state,protect=True):
        """Apply the operator over an initial state.
        
        Warning: an oracle can modify the original input.
        
        Args:
            state: Initial vector state.
            protect: Whether to protect or no the initial state variable
        
        Returns:
            state: Final vector after the operations.
        """
        
        if protect: state = state.copy()  # Protect the initial state variable in this mode.
        # Mark the elements multiplying by the factor.
        state[self.marked_indexes] = state[self.marked_indexes]*-1
        
        return state

class SparseReflection(SparseOperator):
    """Sparse Reflection operator.

    Attributes:
        string: A string representing the operator in an algebraic form.
        info_string: A string with the information of the operator.
        psi_matrix: Matrix Psi needed for the reflection.
        class_type: Kind of class.
    """
    
    string = 'R'
    info_string = 'Sparse Reflection'
    
    def __init__(self,space_data,transition_matrix,name=None):
        """Initializes the reflection operator.

        Args:
            transition_matrix: Classical column-stochastic transition matrix.
            name: Custom name for the reflection operator.
        
        Raises:
            Exception: If the transition matrix is not column-stochastic.
        """
        
        self.space_data = space_data
        self.name = name
        
        if self.space_data.check_transition_matrix(transition_matrix) != True:
            raise Exception(transition_matrix_exception)
        
        if self.name is not None:
            self.info_string += f' {name}'
        self.psi_matrix = np.expand_dims(np.sqrt(transition_matrix),axis=1)  # Creates the psi_matrix from the transition matrix.
        
    def operate(self,state):
        """Apply the operator over an initial state.
        
        Args:
            state: Initial vector state.
        
        Returns:
            state: Final vector after the operations.
        """
        
        shape = state.shape
        dimension = len(shape)
        if dimension == 1:
            state = np.expand_dims(state,axis=1)
                
        # Apply the operations corresponding to the reflection.
        coefs = np.add.reduceat(state * self.psi_matrix,self.space_data.starts)
        state_par = self.psi_matrix * np.repeat(coefs,self.space_data.sizes,axis=0)
        state = 2*state_par - state
            
        if dimension == 1:
            state = np.squeeze(state)
        
        return state

class SparseUpdate(SparseOperator):
    """Sparse Update operator.

    Attributes:
        string: A string representing the operator in an algebraic form.
        info_string: A string with the information of the operator.
        method: Method for computing the operator.
        psi_matrix: Matrix Psi needed for the reflection.
        class_type: Kind of class.
    """
    
    string = 'V'
    info_string = 'Sparse Update'
    
    def __init__(self,space_data,transition_matrix,method=2,name=None,eps=1e-16):
        """Initializes the update operator.

        Args:
            transition_matrix: Classical column-stochastic transition matrix.
            method: Method to use:
                -1: Minimal subspace.
                -2: Householder reflection.
            name: Custom name for the reflection operator.
            eps: Variable to avoid division by zero eror when normalizing null vectors.
        
        Raises:
            Exception: If the transition matrix is not column-stochastic.
        """
        
        self.space_data = space_data
        self.name = name
        self.method = method
        
        if self.space_data.check_transition_matrix(transition_matrix) != True:
            raise Exception(transition_matrix_exception)
        
        self.info_string = 'Sparse Update'
        if self.name is not None:
            self.info_string += f' {self.name}'
        
        self.psi_matrix = np.expand_dims(np.sqrt(transition_matrix),axis=1)
        
        if self.method == 1:  # Minimal subspace.
            
            # Create other states.
            self.other = self.psi_matrix.copy()
            self.other[self.space_data.starts] = 0
            other_norms = np.add.reduceat(np.abs(self.other)**2,self.space_data.starts)
            self.other /= (np.sqrt(np.repeat(other_norms,self.space_data.sizes,axis=0))+eps)
            
            # Create psi_perp.
            coeff_zero = self.psi_matrix[self.space_data.starts]
            coeff_other = np.add.reduceat(np.conj(self.other) * self.psi_matrix,self.space_data.starts)
            self.psi_perp = np.zeros(self.psi_matrix.shape)
            self.psi_perp = np.repeat(coeff_zero,self.space_data.sizes,axis=0)*self.other
            self.psi_perp[self.space_data.starts] = -coeff_other
        
        else:  # Householder reflection.
            
            self.psi_matrix[self.space_data.starts] -= 1
            psi_matrix_norms = np.add.reduceat(np.abs(self.psi_matrix)**2,self.space_data.starts)
            self.psi_matrix /= (np.sqrt(np.repeat(psi_matrix_norms,self.space_data.sizes,axis=0))+eps)
    
    def operate(self,state):
        """Apply the operator over an initial state.
        
        Args:
            state: Initial vector state.
        
        Returns:
            state: Final vector after the operations.
        """
        
        shape = state.shape
        dimension = len(shape)
        if dimension == 1:
            state = np.expand_dims(state,axis=1)
                
        if self.method == 1:  # Minimal subspace.
                
            coeff_zero = state[self.space_data.starts]
            coeff_other = np.add.reduceat(np.conj(self.other) * state,self.space_data.starts)
            
            state_parallel = np.zeros_like(state)
            state_parallel = np.repeat(coeff_other,self.space_data.sizes,axis=0)*self.other
            state_parallel[self.space_data.starts] = coeff_zero
            
            state = np.repeat(coeff_zero,self.space_data.sizes,axis=0)*self.psi_matrix + np.repeat(coeff_other,self.space_data.sizes,axis=0)*self.psi_perp + state - state_parallel
        
        else:  # Householder reflection.
            
            coefs = np.add.reduceat(state * self.psi_matrix,self.space_data.starts)
            state_par = self.psi_matrix * np.repeat(coefs,self.space_data.sizes,axis=0)
            state = state - 2*state_par
        
        if dimension == 1:
            state = np.squeeze(state)
        
        return state
    
    def inverse(self):
        """Invert the unitary operator."""
        
        update_dagger = SparseUpdateDagger.__new__(SparseUpdateDagger)
        update_dagger.__dict__ = self.__dict__.copy()
        
        update_dagger.info_string = 'Update dagger'
        if update_dagger.name is not None: update_dagger.info_string += f' {update_dagger.name}'
        
        return update_dagger

class SparseUpdateDagger(SparseOperator):
    """Sparse Update operator.

    Attributes:
        string: A string representing the operator in an algebraic form.
        info_string: A string with the information of the operator.
        method: Method for computing the operator.
        psi_matrix: Matrix Psi needed for the reflection.
        class_type: Kind of class.
    """
    
    string = 'V\N{DAGGER}'
    info_string = 'Sparse Update dagger'
    
    def __init__(self,space_data,transition_matrix,method=2,name=None,eps=1e-16):
        """Initializes the update operator.

        Args:
            transition_matrix: Classical column-stochastic transition matrix.
            method: Method to use:
                -1: Minimal subspace.
                -2: Householder reflection.
            name: Custom name for the reflection operator.
            eps: Variable to avoid division by zero eror when normalizing null vectors.
        
        Raises:
            Exception: If the transition matrix is not column-stochastic.
        """
        
        self.space_data = space_data
        self.name = name
        self.method = method
        
        if self.space_data.check_transition_matrix(transition_matrix) != True:
            raise Exception(transition_matrix_exception)
        
        self.info_string = 'Sparse Update dagger'
        if self.name is not None:
            self.info_string += f' {self.name}'
        
        self.psi_matrix = np.expand_dims(np.sqrt(transition_matrix),axis=1)
        
        if self.method == 1:  # Minimal subspace.
            
            # Create other states.
            self.other = self.psi_matrix.copy()
            self.other[self.space_data.starts] = 0
            other_norms = np.add.reduceat(np.abs(self.other)**2,self.space_data.starts)
            self.other /= (np.sqrt(np.repeat(other_norms,self.space_data.sizes,axis=0))+eps)
            
            # Create psi_perp.
            coeff_zero = self.psi_matrix[self.space_data.starts]
            coeff_other = np.add.reduceat(np.conj(self.other) * self.psi_matrix,self.space_data.starts)
            self.psi_perp = np.zeros(self.psi_matrix.shape)
            self.psi_perp = np.repeat(coeff_zero,self.space_data.sizes,axis=0)*self.other
            self.psi_perp[self.space_data.starts] = -coeff_other
        
        else:  # Householder reflection.
            
            self.psi_matrix[self.space_data.starts] -= 1
            psi_matrix_norms = np.add.reduceat(np.abs(self.psi_matrix)**2,self.space_data.starts)
            self.psi_matrix /= (np.sqrt(np.repeat(psi_matrix_norms,self.space_data.sizes,axis=0))+eps)
    
    def operate(self,state):
        """Apply the operator over an initial state.
        
        Args:
            state: Initial vector state.
        
        Returns:
            state: Final vector after the operations.
        """
        
        shape = state.shape
        dimension = len(shape)
        if dimension == 1:
            state = np.expand_dims(state,axis=1)
                
        if self.method == 1:  # Minimal subspace.
                
            coeff_psi = np.add.reduceat(np.conj(self.psi_matrix) * state,self.space_data.starts)
            coeff_perp = np.add.reduceat(np.conj(self.psi_perp) * state,self.space_data.starts)
            
            state_parallel = np.repeat(coeff_psi,self.space_data.sizes,axis=0)*self.psi_matrix + np.repeat(coeff_perp,self.space_data.sizes,axis=0)*self.psi_perp
            
            state = np.repeat(coeff_perp,self.space_data.sizes,axis=0)*self.other + state - state_parallel
            state[self.space_data.starts] += coeff_psi
        
        else:  # Householder reflection.
            
            coefs = np.add.reduceat(state * self.psi_matrix,self.space_data.starts)
            state_par = self.psi_matrix * np.repeat(coefs,self.space_data.sizes,axis=0)
            state = state - 2*state_par
        
        if dimension == 1:
            state = np.squeeze(state)
        
        return state
    
    def inverse(self):
        """Invert the unitary operator."""
        
        update = SparseUpdate.__new__(SparseUpdate)
        update.__dict__ = self.__dict__.copy()
        
        update.info_string = 'Update'
        if update.name is not None: update.info_string += f' {update.name}'
        
        return update

class SparseReflection0(SparseOperator):
    """Sparse Reflection 0 operator.

    Attributes:
        string: A string representing the operator in an algebraic form.
        info_string: A string with the information of the operator.
        class_type: Kind of class.
    """
    
    string = 'R\N{SUBSCRIPT ZERO}'
    info_string = 'Sparse Reflection 0'
    
    def __init__(self,space_data):
        self.space_data = space_data
    
    def operate(self,state):
        """Apply the operator over an initial state.
        
        Args:
            state: Initial vector state.
        
        Returns:
            state: Final vector after the operations.
        """
        
        shape = state.shape
        dimension = len(shape)
        if dimension == 1:
            state = np.expand_dims(state,axis=1)
                
        state = -state
        state[self.space_data.starts] *= -1
        
        if dimension == 1:
            state = np.squeeze(state)
        
        return state

class SparseMeasurement():
    """Class to obtain the probability distributions for a sparse state.

    Attributes:
        info_string: A string with the information of the operator.
        register: Register to measure:
            -'X' or 'x' or 1: register 1.
            -'Y' or 'y' or 2: register 2.
            -'XY' or 'xy' or 12: both registers.
    """
    
    info_string = 'Sparse Measurement'
    
    def __init__(self,space_data,register):
        """Initializes the measurement operator.

        Args:
            register: Register to measure:
                -'X' or 'x' or 1: register 1.
                -'Y' or 'y' or 2: register 2.
                -'XY' or 'xy' or 12: both registers.
        """
        
        self.space_data = space_data
        
        self.register = register
        self.info_string += f': Register {register}'
    
    def info(self):
        """Print the information of the measurement."""
        
        print(self.info_string)
    
    def operate(self,state):
        """Measure an initial state.
        
        Args:
            state: Initial vector state.
        
        Returns:
            measure: Probability distribution after the measurement.
              -If both registers are being measured, a tuple with 2 elements is returned.
        """
        
        shape = state.shape
        dimension = len(shape)
        if dimension == 1:
            state = np.expand_dims(state,axis=1)
        
        # Perform the measurement.
        abs_sq_state = abs(state)**2
        if self.register == 'X' or self.register == 'x' or self.register == 1:
            measure = np.add.reduceat(abs_sq_state,self.space_data.starts,axis=0)
            if dimension == 1:
                measure = np.squeeze(measure)
        elif self.register == 'Y' or self.register == 'y' or self.register == 2:
            measure = np.add.reduceat(abs_sq_state[self.space_data.permutation],self.space_data.starts,axis=0)
            if dimension == 1:
                measure = np.squeeze(measure)
        else:
            measure_x = np.add.reduceat(abs_sq_state,self.space_data.starts,axis=0)
            measure_y = np.add.reduceat(abs_sq_state[self.space_data.permutation],self.space_data.starts,axis=0)
            if dimension == 1:
                measure_x = np.squeeze(measure_x)
                measure_y = np.squeeze(measure_y)
            measure = (measure_x,measure_y)
        
        return measure

class SparseSingleUnitary(SparseUnitary):
    """Sparse single unitary Szegedy operator model.

    Attributes:
        string: A string representing the unitary operator in an algebraic form.
        info_string: A string with the information of all the operators inside the unitary.
    """
    
    def __init__(self,space_data,transition_matrix,name=None):
        """Initializes the unitary model.

        Args:
            transition_matrix: Classical column-stochastic transition matrix.
            name: Custom name for the unitary operator.
        """
        
        self.space_data = space_data
        self.string = 'SR'
        self.operators = []
        self.operators.append(self.space_data.Reflection(transition_matrix))
        self.operators.append(self.space_data.Swap())
        self.info_string = 'Sparse Single Szegedy unitary:'
        self.name = name
        if self.name is not None:
            self.info_string += f' {name}'
        for op_index, operator in enumerate(self.operators):
            self.info_string = self.info_string + f'\n {op_index+1} - ' + operator.info_string

class SparseDoubleUnitary(SparseUnitary):
    """Sparse double unitary Szegedy operator model.

    Attributes:
        string: A string representing the unitary operator in an algebraic form.
        info_string: A string with the information of all the operators inside the unitary.
    """
    
    def __init__(self,space_data,transition_matrix,name=None):
        """Initializes the unitary model.

        Args:
            transition_matrix: Classical column-stochastic transition matrix.
            name: Custom name for the unitary operator.
        """
        
        self.space_data = space_data
        self.string = 'SRSR'
        self.operators = []
        R = self.space_data.Reflection(transition_matrix)
        S = self.space_data.Swap()
        self.operators.append(R)
        self.operators.append(S)
        self.operators.append(R)
        self.operators.append(S)
        self.info_string = 'Sparse Double Szegedy unitary:'
        self.name = name
        if self.name is not None:
            self.info_string += f' {name}'
        for op_index, operator in enumerate(self.operators):
            self.info_string = self.info_string + f'\n {op_index+1} - ' + operator.info_string


## Utils

def create_initial_state_sparse(space_data,transition_matrix,coefficients=None,nodes=None):
    """Creates a suitable sparse initial state from the sparse transition matrix.

    Args:
        transition_matrix: Sparse column-stochastic transition matrix.
        coefficients: List of coefficients for the linear combination of the psi_i states. Default: 1/np.sqrt(len(nodes)).
        nodes: List of nodes corresponding to the psi_i states of the linear combination. Default: all nodes.
    
    Returns:
        initial_state: NumPy vector representing the sparse initial state.
        
    Raises:
        Exception: If the transition matrix is not column-stochastic.
        Exception: If coefficients and nodes have different length.
    """
    
    if space_data.check_transition_matrix(transition_matrix) != True:
        raise Exception(transition_matrix_exception)
    
    N = space_data.N
    
    if nodes is None:
        nodes = np.arange(N)
    
    if coefficients is None:
        coefficients = [1/np.sqrt(len(nodes)) for _ in range(len(nodes))]
    coefficients = np.array(coefficients)
    
    if len(nodes) != len(coefficients):
        raise Exception('The parameters \'coefficients\' and \'nodes\' must be lists of the same size.')
    
    # Create the list with the coefficients of the linear combination including zeroes.
    coefficients_total = np.zeros([N]).astype(coefficients.dtype)
    coefficients_total[nodes] = coefficients
    coefficients_total = np.reshape(np.array(coefficients_total),(1,N))
    
    initial_state = np.repeat(coefficients_total,space_data.sizes)*np.sqrt(transition_matrix)
    
    return initial_state

def create_no_coin_state_sparse(space_data,N=None,coefficients=None,nodes=None):
    """Creates a suitable sparse initial state independent of the transition matrix.

    Args:
        N: Number of nodes of the graph. Not necessary, it is just for compatibility with SQUWALS codes.
        coefficients: List of coefficients for the linear combination of the psi_i states. Default: 1/np.sqrt(len(nodes)).
        nodes: List of nodes corresponding to the psi_i states of the linear combination. Default: all nodes.
    
    Returns:
        initial_state: NumPy vector representing the sparse initial state without coin information.
        
    Raises:
        Exception: If coefficients and nodes have different length.
    """
    
    N = space_data.N
    
    transition_matrix = np.zeros(space_data.N_red)
    transition_matrix[space_data.starts] = 1
    
    if nodes is None:
        nodes = np.arange(N)
    
    if coefficients is None:
        coefficients = [1/np.sqrt(len(nodes)) for _ in range(len(nodes))]
    coefficients = np.array(coefficients)
    
    if len(nodes) != len(coefficients):
        raise Exception('The parameters \'coefficients\' and \'nodes\' must be lists of the same size.')
    
    # Create the list with the coefficients of the linear combination including zeroes.
    coefficients_total = np.zeros([N]).astype(coefficients.dtype)
    coefficients_total[nodes] = coefficients
    coefficients_total = np.reshape(np.array(coefficients_total),(1,N))
    
    initial_state = np.repeat(coefficients_total,space_data.sizes)*np.sqrt(transition_matrix)
    
    return initial_state

def create_psi_states_sparse(space_data,transition_matrix,nodes=None):
    """Creates the psi_i position states from the sparse transition matrix.
    
    Args:
        transition_matrix: Sparse column-stochastic transition matrix.
        nodes: Nodes corresponding to the psi_i states:
            -int: Return a single psi_i state.
            -list: Return a batch with the psi_i states.
            -default: Return a batch with all the psi_i states.

    Returns:
        psi_state: NumPy tensor of shape (N_red,) representing the sparse psi_i state.
        psi_batch: NumPy tensor of shape (N_red,len(nodes)) representing the batch of psi_i states.
        
    Raises:
        Exception: If the transition matrix is not column-stochastic.
    """
    
    if space_data.check_transition_matrix(transition_matrix) != True:
        raise Exception(transition_matrix_exception)
    
    N = space_data.N
    N_red = space_data.N_red
    
    starts_mod = space_data.starts.copy()
    starts_mod.append(N_red)
    
    if nodes is None:
        nodes = np.arange(N)
    
    if type(nodes) == int:  # Create a single psi_i state.
        psi_state = np.zeros([N_red])
        psi_state[starts_mod[nodes]:starts_mod[nodes+1]] = np.sqrt(transition_matrix[starts_mod[nodes]:starts_mod[nodes+1]])
        return psi_state
    
    else:  # Create a batch with the psi_i states.
        psi_batch = np.zeros([N_red,len(nodes)])
        for index, node in enumerate(nodes):
            psi_batch[starts_mod[node]:starts_mod[node+1],index] = np.sqrt(transition_matrix[starts_mod[node]:starts_mod[node+1]])
        return psi_batch

def normalize_transition_matrix_sparse(space_data,transition_matrix):
    """Normalizes the sparse transition matrix.
    
    Args:
        transition_matrix: Sparse column-stochastic transition matrix.

    Returns:
        transition_matrix: Sparse column-stochastic transition matrix.
    """
    
    s = np.add.reduceat(transition_matrix,space_data.starts)
    
    transition_matrix = transition_matrix / np.repeat(s,space_data.sizes)
    
    return transition_matrix

def check_transition_matrix_sparse(space_data,transition_matrix):
    """Checks if the sparse transition matrix is stochastic.
    
    Args:
        transition_matrix: Sparse column-stochastic transition matrix.

    Returns:
        bool: True if the transition matrix is normalized.
    """
    
    N = space_data.N
    
    s = np.add.reduceat(transition_matrix,space_data.starts)
    
    return np.allclose(s,np.ones(N))

def obtain_transition_matrix_sparse(space_data,weights_list):
    """Obtains the sparse transition matrix.
    
    Args:
        weights_list: List of weighted edges.

    Returns:
        transition_matrix: Sparse column-stochastic transition matrix.
    """
    
    N_red = space_data.N_red
    
    transition_matrix = np.zeros(N_red)
    
    for element in weights_list:
        edge = (element[0],element[1])
        weight = element[2]
        transition_matrix[space_data.edge_to_index[edge]] = weight
    
    return transition_matrix

def sparse_to_dense_state(space_data,state):
    """Transform a sparse state into a dense state.
    
    Args:
        state: Sparse state.

    Returns:
        dense_state: Dense state.
    """
    
    N = space_data.N
    N_red = space_data.N_red
    
    dimension = len(state.shape)
    if dimension == 1:
        dense_state = np.zeros([N,N],dtype=state.dtype)
    else:
        dense_state = np.zeros([N,N,state.shape[1]],dtype=state.dtype)
    
    for index in range(N_red):
        edge = space_data.index_to_edge[index]
        
        dense_state[edge[1],edge[0]] = state[index]
    
    if dimension == 1:
        dense_state = dense_state.T.reshape(N**2)
    else:
        dense_state = np.transpose(dense_state,axes=(1,0,2)).reshape([N**2,state.shape[1]])
    
    return dense_state

def dense_to_sparse_state(space_data,state):
    """Transform a dense state into a sparse state.
    
    Args:
        state: Dense state.

    Returns:
        sparse_state: Sparse state.
    """
    
    N = space_data.N
    N_red = space_data.N_red
    
    dimension = len(state.shape)
    if dimension == 1:
        sparse_state = np.zeros([N_red],dtype=state.dtype)
        state = state.reshape(N,N).T
    else:
        sparse_state = np.zeros([N_red,state.shape[1]],dtype=state.dtype)
        state = np.transpose(state.reshape([N,N,state.shape[1]]),axes=(1,0,2))
    
    for index in range(N_red):
        edge = space_data.index_to_edge[index]
        
        sparse_state[index] = state[edge[1],edge[0]]
    
    return sparse_state

def sparse_to_dense_matrix(space_data,transition_matrix):
    """Transform a sparse transition matrix into a dense transition matrix.
    
    Args:
        transition matrix: Sparse transition matrix.

    Returns:
        dense_transition_matrix: Dense transition matrix.
    """
    
    N = space_data.N
    N_red = space_data.N_red
    
    dense_transition_matrix = np.zeros([N,N])
    
    for index in range(N_red):
        edge = space_data.index_to_edge[index]
        dense_transition_matrix[edge[1],edge[0]] = transition_matrix[index]
    
    return dense_transition_matrix

def dense_to_sparse_matrix(space_data,transition_matrix):
    """Transform a dense transition matrix into a sparse transition matrix.
    
    Args:
        transition matrix: Dense transition matrix.

    Returns:
        sparse_transition_matrix: Sparse transition matrix.
    """
    
    N = space_data.N
    N_red = space_data.N_red
    
    sparse_transition_matrix = np.zeros(N_red)
    
    for index in range(N_red):
        edge = space_data.index_to_edge[index]
        sparse_transition_matrix[index] = transition_matrix[edge[1],edge[0]]
    
    return sparse_transition_matrix


## Simulators

def quantum_szegedy_simulator_sparse(space_data,unitary,initial_state,time_steps=100,measure=1,protect=True):
    """Simulator of the Szegedy quantum walk.

    Args:
        unitary: Unitary operator model.
        initial_state: Initial state or batch of states.
        time_steps: Number of steps of the quantum walk.
        measure: Register to measure:
            -'X' or 'x' or 1: register 1.
            -'Y' or 'y' or 2: register 2.
            -'XY' or 'xy' or 12: both registers.
        protect: Whether to protect or no the initial state variable
    
    Returns:
        probability_distributions: A tensor with the probability distributions at each time step.
            -If both registers are being measured, a tuple with 2 elements is returned.
    """
    
    # We must copy the input because the unitary with oracles can change it, if the oracle is the first operator.
    if protect:
        state = initial_state.copy()
    else:
        state = initial_state
    
    N = space_data.N  # Size of the graph.
    
    shape = state.shape
    dimension = len(shape)
    if dimension == 1:
        state = np.expand_dims(state,axis=1)
    
    batch_size = np.shape(state)[1]
    
    # Crete the tensors for saving the results.
    if measure == 'both' or measure == 'XY' or measure == 'xy' or measure == 12:
        probability_distributions_x = np.zeros([time_steps+1,N,batch_size])
        probability_distributions_y = np.zeros([time_steps+1,N,batch_size])
    else:
        probability_distributions = np.zeros([time_steps+1,N,batch_size])
        
    # Measure the probability distribution at time 0.
    abs_sq_state = np.abs(state)**2
    if measure == 'X' or measure == 'x' or measure == 1:
        probability_distributions[0,:,:] = np.add.reduceat(abs_sq_state,space_data.starts,axis=0)
    elif measure == 'Y' or measure == 'y' or measure == 2:
        probability_distributions[0,:,:] = np.add.reduceat(abs_sq_state[space_data.permutation],space_data.starts,axis=0)
    else:
        probability_distributions_x[0,:,:] = np.add.reduceat(abs_sq_state,space_data.starts,axis=0)
        probability_distributions_y[0,:,:] = np.add.reduceat(abs_sq_state[space_data.permutation],space_data.starts,axis=0)
        
    # Time loop
    for time in range(1,time_steps+1):
        state = unitary.operate(state)  # Apply the quantum evolution.
        # Measure the probability distributions.
        abs_sq_state = np.abs(state)**2
        if measure == 'X' or measure == 'x' or measure == 1:
            probability_distributions[time,:,:] = np.add.reduceat(abs_sq_state,space_data.starts,axis=0)
        elif measure == 'Y' or measure == 'y' or measure == 2:
            probability_distributions[time,:,:] = np.add.reduceat(abs_sq_state[space_data.permutation],space_data.starts,axis=0)
        else:
            probability_distributions_x[time,:,:] = np.add.reduceat(abs_sq_state,space_data.starts,axis=0)
            probability_distributions_y[time,:,:] = np.add.reduceat(abs_sq_state[space_data.permutation],space_data.starts,axis=0)
    # End of the time loop
    
    if measure == 'both' or measure == 'XY' or measure == 'xy' or measure == 12:
        probability_distributions = (probability_distributions_x,probability_distributions_y)
    
    if dimension == 1:
        if type(probability_distributions) == tuple:
            probability_distributions = tuple(np.squeeze(element) for element in probability_distributions)
        else:
            probability_distributions = np.squeeze(probability_distributions)
    
    return probability_distributions

def classical_walk_simulator_sparse(space_data,transition_matrix,time_steps=100,initial_distribution=None,only_last=False):
    """Simulator of the classical walk.

    Args:
        transition_matrix: Sparse column-stochastic transition matrix.
        time_steps: Number of steps of the classical walk.
        distribution: NumPy tensor of shape (N,). Initial probability distribution of the walker.
        only_last: If True, only the last step distribution is returned.

    Returns:
        distribution: NumPy tensor of shape (N,). Last step walker distribution if only_last == True.
        probability_distributions: NumPy tensor of shape (time_steps+1, N), distributions of the walker at each time step.
        
    Raises:
        Exception: If the transition matrix is not column-stochastic.
    """
    
    # Obtain a scipy sparse matrix.
    
    N = space_data.N
    N_red = space_data.N_red
    
    index_to_edge = space_data.index_to_edge
    
    i = np.empty(N_red)
    j = np.empty(N_red)
    
    for index in range(N_red):
        edge = index_to_edge[index]
        j[index] = edge[0]
        i[index] = edge[1]
    
    sparse_matrix = sparse.coo_matrix((transition_matrix,(i,j)),shape=(N,N)).tocsc()
    sparse_matrix.sum_duplicates()
    sparse_matrix.eliminate_zeros()
    sparse_matrix.sort_indices()
    
    
    N = space_data.N  # Size of the graph.
    distribution = initial_distribution
    
    if distribution is None:
        distribution = np.ones([N])/N  # The default initial distribution is the uniform one.
    
    if only_last == True:
        # Simulate the classical walk.
        for t in range(1,time_steps+1):
            distribution = sparse_matrix @ distribution
        
        return distribution
    
    else:
        probability_distributions = np.zeros([time_steps+1,N])
        probability_distributions[0] = distribution
        # Simulate the classical walk and save the results.
        for t in range(1,time_steps+1):
            distribution = sparse_matrix @ distribution
            probability_distributions[t] = distribution
        
        return probability_distributions


## Space

class SpaceData():
    """Data of the reduced Hilbert space.

    Attributes:
        edges: List of the edges forming the reduced subspace.
        N: Number of nodes of the graph.
        N_red: Dimension of the reduced subspace.
        edge_to_index: Dictionary relating the edges to the indexes in the reduced subspace.
        index_to_edge: Dictionary relating the indexes in the reduced subspace to the edges.
        permutation: List of indexes to swap the edges.
        starts: List of indexes indicating the first edge for each node.
        sizes: List containing the number of edges for each node.
    """
    
    def __init__(self,edge_list,add_zeroes=False,swap_zeroes=True):
        """Initializes the data of the space.

        Args:
            edge_list: List of edges of the graph.
            add_zeroes: If True, edges starting from 0 are added.
            swap_zeroes: If True, edges starting from 0 are symmetrized.
        """
        
        self.edges = []
        for edge in edge_list:
            self.edges.append((edge[0],edge[1]))
        self.edges.sort()
        
        self.N = self.edges[-1][0] + 1
        
        graph = nx.DiGraph()
        graph.add_edges_from(self.edges)
        
        # Add zeroes (the swapped edges are also added).
        
        if add_zeroes == True and swap_zeroes == True:
        
            graph.add_edges_from((n,0) for n in graph.nodes)
        
        # Symmetrize
        
        graph = graph.to_undirected().to_directed()
        
        # Add zeroes (the swapped edges are not added).
        
        if add_zeroes == True and swap_zeroes == False:
        
            graph.add_edges_from((n,0) for n in graph.nodes)
        
        self.edges = list(set(graph.edges))
        self.edges.sort()
        
        self.N_red = len(self.edges)
        
        slices = []
        for a in range(self.N):
            slices.append([])
        
        self.edge_to_index = {}
        self.index_to_edge = {}
        
        for index, edge in enumerate(self.edges):
            node = edge[0]
            slices[node].append(index)
            
            self.edge_to_index[edge] = index
            self.index_to_edge[index] = edge
        
        self.permutation = [0]*self.N_red
        for index, edge in enumerate(self.edges):
            sw_edge = (edge[1],edge[0])
            new_index = self.edge_to_index.get(sw_edge)
            if new_index is not None:
                self.permutation[index] = new_index
            else:
                self.permutation[index] = index
        
        
        self.starts = [s[0] for s in slices]
        self.sizes = [len(s) for s in slices]
    
    
    # Define the classes and functions as methods.
    
    def Unitary(self,operators=None,name=None):
        return SparseUnitary(self,operators=operators,name=name)
    
    def Reflection(self,transition_matrix,name=None):
        return SparseReflection(self,transition_matrix,name=name)
    
    def Swap(self):
        return SparseSwap(self)
    
    def Oracle(self,register,marked_nodes,name=None):
        return SparseOracle(self,register,marked_nodes,name=name)
    
    def Update(self,transition_matrix,method=2,name=None,eps=1e-16):
        return SparseUpdate(self,transition_matrix,method=method,name=name,eps=eps)
    
    def UpdateDagger(self,transition_matrix,method=2,name=None,eps=1e-16):
        return SparseUpdateDagger(self,transition_matrix,method=method,name=name,eps=eps)
    
    def Reflection0(self):
        return SparseReflection0(self)
    
    def Measurement(self,register):
        return SparseMeasurement(self,register)
    
    def SingleUnitary(self,transition_matrix,name=None):
        return SparseSingleUnitary(self,transition_matrix,name=name)
    
    def DoubleUnitary(self,transition_matrix,name=None):
        return SparseDoubleUnitary(self,transition_matrix,name=name)
    
    def create_initial_state(self,transition_matrix,coefficients=None,nodes=None):
        return create_initial_state_sparse(self,transition_matrix,coefficients=coefficients,nodes=nodes)
    
    def create_no_coin_state(self,N=None,coefficients=None,nodes=None):
        return create_no_coin_state_sparse(self,N=N,coefficients=coefficients,nodes=nodes)
    
    def create_psi_states(self,transition_matrix,nodes=None):
        return create_psi_states_sparse(self,transition_matrix,nodes=nodes)
    
    def obtain_transition_matrix(self,weights_list):
        return obtain_transition_matrix_sparse(self,weights_list)
    
    def normalize_transition_matrix(self,transition_matrix):
        return normalize_transition_matrix_sparse(self,transition_matrix)
    
    def check_transition_matrix(self,transition_matrix):
        return check_transition_matrix_sparse(self,transition_matrix)
    
    def to_dense_state(self,state):
        return sparse_to_dense_state(self,state)
    
    def to_sparse_state(self,state):
        return dense_to_sparse_state(self,state)
    
    def to_dense_matrix(self,transition_matrix):
        return sparse_to_dense_matrix(self,transition_matrix)
    
    def to_sparse_matrix(self,transition_matrix):
        return dense_to_sparse_matrix(self,transition_matrix)
    
    def quantum_szegedy_simulator(space_data,unitary,initial_state,time_steps=100,measure=1,protect=True):
        return quantum_szegedy_simulator_sparse(space_data,unitary,initial_state,time_steps=time_steps,measure=measure,protect=protect)
    
    def classical_walk_simulator(space_data,transition_matrix,time_steps=100,initial_distribution=None,only_last=False):
        return classical_walk_simulator_sparse(space_data,transition_matrix,time_steps=time_steps,initial_distribution=initial_distribution,only_last=only_last)
    
    
    Unitary.__doc__ = SparseUnitary.__doc__
    Reflection.__doc__ = SparseReflection.__doc__
    Swap.__doc__ = SparseSwap.__doc__
    Oracle.__doc__ = SparseOracle.__doc__
    Update.__doc__ = SparseUpdate.__doc__
    UpdateDagger.__doc__ = SparseUpdateDagger.__doc__
    Reflection0.__doc__ = SparseReflection0.__doc__
    Measurement.__doc__ = SparseMeasurement.__doc__
    SingleUnitary.__doc__ = SparseSingleUnitary.__doc__
    DoubleUnitary.__doc__ = SparseDoubleUnitary.__doc__
    
    create_initial_state.__doc__ = create_initial_state_sparse.__doc__
    create_no_coin_state.__doc__ = create_no_coin_state_sparse.__doc__
    create_psi_states.__doc__ = create_psi_states_sparse.__doc__
    obtain_transition_matrix.__doc__ = obtain_transition_matrix_sparse.__doc__
    normalize_transition_matrix.__doc__ = normalize_transition_matrix_sparse.__doc__
    check_transition_matrix.__doc__ = check_transition_matrix_sparse.__doc__
    to_dense_state.__doc__ = sparse_to_dense_state.__doc__
    to_sparse_state.__doc__ = dense_to_sparse_state.__doc__
    to_dense_matrix.__doc__ = sparse_to_dense_matrix.__doc__
    to_sparse_matrix.__doc__ = dense_to_sparse_matrix.__doc__
    
    quantum_szegedy_simulator.__doc__ = quantum_szegedy_simulator_sparse.__doc__
    classical_walk_simulator.__doc__ = classical_walk_simulator_sparse.__doc__
    
    __all__ = [
        'Unitary',
        'Reflection',
        'Swap',
        'Oracle',
        'Update',
        'UpdateDagger',
        'Reflection0',
        'Measurement',
        'SingleUnitary',
        'DoubleUnitary',
        'create_initial_state',
        'create_no_coin_state',
        'create_psi_states',
        'obtain_transition_matrix',
        'normalize_transition_matrix',
        'check_transition_matrix',
        'quantum_szegedy_simulator',
        'classical_walk_simulator',
        'to_dense_state',
        'to_sparse_state',
        'to_dense_matrix',
        'to_sparse_matrix']