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

SQUWALS vendor with additional operators.
"""

import numpy as np

from squwals import *

import squwals
__version__ = squwals.__version__
__all__ = squwals.__all__

__all__.extend(['Update','UpdateDagger','Reflection0','create_no_coin_state'])

class Operator():
    """Base class for operator.

    Attributes:
      string: A string representing the operator in an algebraic form.
      info_string: A string with the information of the operator.
      class_type: Kind of class.
    """
    
    string = None
    info_string = None
    class_type = 'operator'
    
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
        
        return Unitary([self]) * unitary_2
    
    def inverse(self):
        """Invert the unitary operator."""
        
        return self

class Update(Operator):
    """Reflection operator.

    Attributes:
        string: A string representing the operator in an algebraic form.
        info_string: A string with the information of the operator.
        method: Method for computing the operator.
        psi_matrix: Matrix Psi needed for the operations.
        class_type: Kind of class.
    """
    
    string = 'V'
    info_string = 'Update'
    
    def __init__(self,transition_matrix,method=2,name=None,eps=1e-16):
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
        
        if np.allclose(np.sum(transition_matrix,axis=0),np.ones(np.shape(transition_matrix)[1])) != True:
            raise Exception('The transition matrix is not column-stochastic. See tutorial: https://github.com/OrtegaSA/squwals-repo/tree/main/Tutorials')
        
        self.method = method
        
        self.name = name
        if name is not None:
            self.info_string += f' {name}'
        
        N = np.shape(transition_matrix)[1];  # Size of the graph.
        self.psi_matrix = np.sqrt(transition_matrix).reshape(N,N,1)  # Creates the psi_matrix from the transition matrix.
        
        if self.method == 1:  # Minimal subspace.
            
            # Create other states.
            self.other = np.zeros(self.psi_matrix.shape)
            self.other[1:] = self.psi_matrix[1:] / (np.sqrt(np.sum(np.abs(self.psi_matrix[1:])**2,axis=0))+eps)
            
            # Create psi_perp.
            coeff_zero = self.psi_matrix[0:1]
            coeff_other = np.sum(np.conj(self.other[1:]) * self.psi_matrix[1:], axis=0, keepdims=True)
            self.psi_perp = np.zeros(self.psi_matrix.shape)
            self.psi_perp = coeff_zero*self.other
            self.psi_perp[0] = -coeff_other
            
        else:  # Householder reflection.
            
            self.psi_matrix[0,:] -= 1
            self.psi_matrix = self.psi_matrix / (np.sqrt(np.sum(np.abs(self.psi_matrix)**2,axis=0))+eps)

    def operate(self,state,mode='vector'):
        """Apply the operator over an initial state.
        
        Args:
            state: Initial vector state.
            mode: A string to indicate whether the initial state is a vector or
              a matrix state.
        
        Returns:
            state: Final vector after the operations.
        """
        
        if mode == 'vector':  # Tensorize into a matrix state.
            shape = state.shape
            dimension = len(shape)
            N = int(np.sqrt(shape[0]))
            if dimension == 1:
                state = np.expand_dims(state.reshape(N,N).T, axis=2)
            else:
                state = np.transpose(state.reshape([N,N,state.shape[1]]),axes=(1,0,2))
                
        if self.method == 1:  # Minimal subspace.
            
            coeff_zero = state[0:1]
            coeff_other = np.sum(np.conj(self.other[1:]) * state[1:], axis=0, keepdims=True)
            
            state_parallel = np.zeros_like(state)
            state_parallel = coeff_other*self.other
            state_parallel[0] = coeff_zero
            
            state = coeff_zero*self.psi_matrix + coeff_other*self.psi_perp + state - state_parallel
            
        else:  # Householder reflection.
                
            coeff = np.sum(np.conj(self.psi_matrix) * state, axis=0, keepdims=True);
            state_parallel = self.psi_matrix*coeff
            state = state - 2*state_parallel
        
        if mode == 'vector':  # Detensorize to the original shape.
            if dimension == 1:
                state = np.transpose(state,axes=(1,0,2)).reshape(N**2)
            else:
                state = np.transpose(state,axes=(1,0,2)).reshape([N**2,shape[1]])
        
        return state
    
    def inverse(self):
        """Invert the unitary operator."""
        
        update_dagger = UpdateDagger.__new__(UpdateDagger)
        update_dagger.__dict__ = self.__dict__.copy()
        
        update_dagger.info_string = 'Update dagger'
        if update_dagger.name is not None: update_dagger.info_string += f' {update_dagger.name}'
        
        return update_dagger

class UpdateDagger(Operator):
    """Reflection operator.

    Attributes:
        string: A string representing the operator in an algebraic form.
        info_string: A string with the information of the operator.
        method: Method for computing the operator.
        psi_matrix: Matrix Psi needed for the operations.
        class_type: Kind of class.
    """
    
    string = 'V\N{DAGGER}'
    info_string = 'Update dagger'
    
    def __init__(self,transition_matrix,method=2,name=None,eps=1e-16):
        """Initializes the update dagger operator.

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
        
        if np.allclose(np.sum(transition_matrix,axis=0),np.ones(np.shape(transition_matrix)[1])) != True:
            raise Exception('The transition matrix is not column-stochastic. See tutorial: https://github.com/OrtegaSA/squwals-repo/tree/main/Tutorials')
        
        self.method = method
        
        self.name = name
        if name is not None:
            self.info_string += f' {name}'
        
        N = np.shape(transition_matrix)[1];  # Size of the graph.
        self.psi_matrix = np.sqrt(transition_matrix).reshape(N,N,1)  # Creates the psi_matrix from the transition matrix.
        
        if self.method == 1:  # Minimal subspace.
            
            # Create other states.
            self.other = np.zeros(self.psi_matrix.shape)
            self.other[1:] = self.psi_matrix[1:] / (np.sqrt(np.sum(np.abs(self.psi_matrix[1:])**2,axis=0))+eps)
            
            # Create psi_perp.
            coeff_zero = self.psi_matrix[0:1]
            coeff_other = np.sum(np.conj(self.other[1:]) * self.psi_matrix[1:], axis=0, keepdims=True)
            self.psi_perp = np.zeros(self.psi_matrix.shape)
            self.psi_perp = coeff_zero*self.other
            self.psi_perp[0] = -coeff_other
        
        else:  # Householder reflection.
            
            self.psi_matrix[0,:] -= 1
            self.psi_matrix = self.psi_matrix / (np.sqrt(np.sum(np.abs(self.psi_matrix)**2,axis=0))+eps)

    def operate(self,state,mode='vector'):
        """Apply the operator over an initial state.
        
        Args:
            state: Initial vector state.
            mode: A string to indicate whether the initial state is a vector or
              a matrix state.
        
        Returns:
            state: Final vector after the operations.
        """
        
        if mode == 'vector':  # Tensorize into a matrix state.
            shape = state.shape
            dimension = len(shape)
            N = int(np.sqrt(shape[0]))
            if dimension == 1:
                state = np.expand_dims(state.reshape(N,N).T, axis=2)
            else:
                state = np.transpose(state.reshape([N,N,state.shape[1]]),axes=(1,0,2))
        
        if self.method == 1:  # Minimal subspace.
            
            coeff_psi = np.sum(np.conj(self.psi_matrix) * state, axis=0, keepdims=True)
            coeff_perp = np.sum(np.conj(self.psi_perp) * state, axis=0, keepdims=True)
            
            state_parallel = coeff_psi*self.psi_matrix + coeff_perp*self.psi_perp
            
            state = coeff_perp*self.other + state - state_parallel
            state[0:1] += coeff_psi
        
        else:  # Householder reflection.
                
            coeff = np.sum(np.conj(self.psi_matrix) * state, axis=0, keepdims=True);
            state_parallel = self.psi_matrix*coeff
            state = state - 2*state_parallel
        
        if mode == 'vector':  # Detensorize to the original shape.
            if dimension == 1:
                state = np.transpose(state,axes=(1,0,2)).reshape(N**2)
            else:
                state = np.transpose(state,axes=(1,0,2)).reshape([N**2,shape[1]])
        
        return state
    
    def inverse(self):
        """Invert the unitary operator."""
        
        update = Update.__new__(Update)
        update.__dict__ = self.__dict__.copy()
        
        update.info_string = 'Update'
        if update.name is not None: update.info_string += f' {update.name}'
        
        return update

class Reflection0(Operator):
    """Reflection 0 operator.

    Attributes:
        string: A string representing the operator in an algebraic form.
        info_string: A string with the information of the operator.
        class_type: Kind of class.
    """
    
    string = 'R\N{SUBSCRIPT ZERO}'
    info_string = 'Reflection 0'
    
    def operate(self,state,mode='vector'):
        """Apply the operator over an initial state.
        
        Args:
            state: Initial vector state.
            mode: A string to indicate whether the initial state is a vector or
              a matrix state.
        
        Returns:
            state: Final vector after the operations.
        """
        
        if mode == 'vector':  # Tensorize into a matrix state.
            shape = state.shape
            dimension = len(shape)
            N = int(np.sqrt(shape[0]))
            if dimension == 1:
                state = np.expand_dims(state.reshape(N,N).T, axis=2)
            else:
                state = np.transpose(state.reshape([N,N,state.shape[1]]),axes=(1,0,2))
                
        state = state.copy()
        state[1:] *= -1
        
        if mode == 'vector':  # Detensorize to the original shape.
            if dimension == 1:
                state = np.transpose(state,axes=(1,0,2)).reshape(N**2)
            else:
                state = np.transpose(state,axes=(1,0,2)).reshape([N**2,shape[1]])
        
        return state

def create_no_coin_state(N,coefficients=None,nodes=None):
    """Creates a suitable initial state independent of the transition matrix.

    Args:
        N: Number of nodes of the graph.
        coefficients: List of coefficients for the linear combination of the psi_i states. Default: 1/np.sqrt(len(nodes)).
        nodes: List of nodes corresponding to the psi_i states of the linear combination. Default: all nodes.
    
    Returns:
        initial_state: NumPy tensor of shape (N,) representing the initial state.
        
    Raises:
        Exception: If coefficients and nodes have different length.
    """
    
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
    
    # The initial state is created unrolling the matrix that results of allocating the
    # coefficients in the first row of a null matrix.
    matrix = np.zeros([N,N])
    matrix[0] = coefficients_total
    initial_state = np.ravel((matrix).T)
    
    return initial_state
