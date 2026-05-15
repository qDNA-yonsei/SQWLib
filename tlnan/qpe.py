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

Quantum phase estimation module.
"""

import numpy as np

def create_qpe_tensor(sz_state,ph_qubits,reps=1,hadamard=False):
    """Creates a suitable initial state for the QPE.

    Args:
        sz_state: A Szegedy state.
        ph_qubits: Number of qubits of a single phase register.
        reps: Number of phase registers.
        hadamard: If True, the Szegedy state is repited.
    
    Returns:
        tensor: NumPy tensor representing the initial state.
    """
    
    sz_dim = len(sz_state)
    ph_dim = 2**ph_qubits
    
    if hadamard == True:
        tensor = (sz_state.reshape((-1,) + (1,)*reps)* np.ones((1,) + (ph_dim,)*reps)) / np.sqrt(ph_dim**reps)
    
    else:
        tensor = np.zeros([sz_dim]+[ph_dim]*reps)
        tensor[(slice(None),) + (0,)*reps] = sz_state
    
    return tensor

def direct_qpe(operator,sz_state,ph_qubits,reps=1):
    """Simulate the QPE algorithm directly from a Szegedy state.

    Args:
        operator: A Szegedy unitary operator.
        sz_state: A Szegedy state.
        ph_qubits: Number of qubits of a single phase register.
        reps: Number of phase registers.
    
    Returns:
        tensor: NumPy tensor representing the final state.
    """
    
    sz_dim = len(sz_state)
    ph_dim = 2**ph_qubits
    
    # Hadamard and controlled unitary.
    
    # Calculate the number of steps for each position in the tensor.
    tensor = np.zeros([sz_dim]+[ph_dim]*reps)+0j
    evolutions = np.zeros([1]+[ph_dim]*reps)
    final_time = ph_dim - 1
    
    for rep in range(reps):
        for t in range(1,final_time+1):
            idx = [slice(None)] * evolutions.ndim
            idx[rep + 1] = slice(t,None)
            shape = evolutions[tuple(idx)].shape
            evolutions[tuple(idx)] += 1
    
    evolutions = np.squeeze(evolutions)
    
    # Calculate the evolution and allocate the states.
    final_time = (ph_dim - 1)*reps
    state = sz_state
    mask = (evolutions == 0)
    tensor[:,mask] = state[:,None]
    
    for t in range(1,final_time+1):
        state = operator.operate(state)
        mask = (evolutions == t)
        tensor[:,mask] = state[:,None]
    
    tensor = tensor / np.sqrt(ph_dim**reps)
    
    # Perform the iQFT
    for rep in range(reps):
        axis = rep + 1
        tensor = np.fft.fft(tensor,axis=axis,norm="ortho")
    
    return tensor

class QPEUnitary():
    """QPE Unitary operator model.

    Attributes:
        string: A string representing the unitary operator in an algebraic form.
        info_string: A string with the information of all the operators inside the unitary.
        class_type: Kind of class.
    """
    
    class_type = 'qpe_unitary'
    
    def __init__(self,operators=None,name=None):
        """Initializes the unitary model.

        Args:
            operators: A list of operators to create directly the unitary.
            name: Custom name for the unitary operator.
        """
        
        self.string = ''
        self.info_string = 'Custom QPE unitary:'
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
    
    def operate(self,state,protect=False):
        """Apply the operators over an initial state.
        
        Args:
            state: Initial vector state.
            protect: Whether to protect or no the initial state variable
        
        Returns:
            state: Final vector after the operations.
        """
        
        if protect: state = state.copy()  # Protect the initial state variable.
        
        for operator in self.operators:  # Operate over the state.
            state = operator.operate(state)
        return state
    
    def __mul__(self,unitary_2):
        """Multiplication of two unitary operators in an algebraic form."""
        
        if unitary_2.class_type == 'qpe_unitary':
            return QPEUnitary(unitary_2.operators + self.operators)
        else:
            return QPEUnitary([unitary_2] + self.operators)
    
    def inverse(self):
        """Invert the unitary operator."""
        
        inverse_operator = QPEUnitary([operator.inverse() for operator in self.operators][::-1],self.name)
        
        return inverse_operator

class QPEOperator():
    """Base class for QPE operator.

    Attributes:
      string: A string representing the operator in an algebraic form.
      info_string: A string with the information of the operator.
      class_type: Kind of class.
    """
    
    string = None
    info_string = None
    class_type = 'qpe_operator'
    
    def __str__(self):
        """Function for printing the operator string."""
        
        return self.string
    
    def info(self):
        """Print the information of the operator."""
        
        print(self.info_string)
    
    def operate(self,tensor):
        """Apply the operator over an initial state.
        
        Args:
            tensor: Initial tensor state.
        
        Returns:
            tensor: Final tensor after the operations.
        """
        
        pass
    
    def __mul__(self,unitary_2):
        """Multiplication of two unitary operators in an algebraic form."""
        
        return QPEUnitary([self]) * unitary_2
    
    def inverse(self):
        """Invert the unitary operator."""
        
        return self

class Controlled(QPEOperator):
    """Operator performing all the controlled-U gates.

    Attributes:
      string: A string representing the operator in an algebraic form.
      info_string: A string with the information of the operator.
      class_type: Kind of class.
      operator: A Szegedy unitary operator.
      ph_qubits: Number of qubits of a single phase register.
      ph_dim: Dimension of a single phase register.
      reps: Number of phase registers.
    """
    
    string = 'C'
    info_string = 'Controlled'
    
    def __init__(self,operator,ph_qubits,reps=1):
        """Initializes the controlled operator.

        Args:
            operator: A Szegedy unitary operator.
            ph_qubits: Number of qubits of a single phase register.
            reps: Number of phase registers. Default: 1.
        """
        
        self.ph_qubits = ph_qubits
        self.ph_dim = 2**self.ph_qubits
        self.operator = operator
        self.reps = reps
    
    def operate(self,tensor):
        """Apply the operator over an initial state.
        
        Args:
            tensor: Initial tensor state.
        
        Returns:
            tensor_c: Final tensor after the operations.
        """
        
        tensor_c = tensor.copy()
        final_time = self.ph_dim - 1
        
        for rep in range(self.reps):
            for t in range(1,final_time+1):
                idx = [slice(None)] * tensor_c.ndim
                idx[rep + 1] = slice(t,None)
                shape = tensor_c[tuple(idx)].shape
                tensor_c[tuple(idx)] = self.operator.operate(tensor_c[tuple(idx)].reshape(shape[0],np.prod(shape[1:]))).reshape(shape)
        
        return tensor_c
    
    def inverse(self):
        """Invert the unitary operator."""
        
        inverse_operator = self.__class__.__new__(self.__class__)
        inverse_operator.__dict__ = self.__dict__.copy()
        inverse_operator.operator = self.operator.inverse()
        
        return inverse_operator

class IQFT(QPEOperator):
    """Inverse Quantum Fourier Transform operator.

    Attributes:
      string: A string representing the operator in an algebraic form.
      info_string: A string with the information of the operator.
      class_type: Kind of class.
      reps: Number of phase registers.
    """
    
    string = 'iF'
    info_string = 'iQFT'
    
    def __init__(self,reps=1):
        """Initializes the iQFT operator.

        Args:
            reps: Number of phase registers. Default: 1.
        """
        
        self.reps = reps
    
    def operate(self,tensor):
        """Apply the operator over an initial state.
        
        Args:
            tensor: Initial tensor state.
        
        Returns:
            tensor_qft: Final tensor after the operations.
        """
            
        tensor_qft = tensor
            
        for rep in range(self.reps):
            axis = rep + 1
            tensor_qft = np.fft.fft(tensor_qft,axis=axis,norm="ortho")
        
        return tensor_qft
    
    def inverse(self):
        """Invert the unitary operator."""
        
        inverse_operator = QFT.__new__(QFT)
        inverse_operator.__dict__ = self.__dict__.copy()
        inverse_operator.string = 'F'
        inverse_operator.info_string = 'QFT'
        
        return inverse_operator

class QFT(QPEOperator):
    """Quantum Fourier Transform operator.

    Attributes:
      string: A string representing the operator in an algebraic form.
      info_string: A string with the information of the operator.
      class_type: Kind of class.
      reps: Number of phase registers.
    """
    
    string = 'F'
    info_string = 'QFT'
    
    def __init__(self,reps=1):
        """Initializes the QFT operator.

        Args:
            reps: Number of phase registers. Default: 1.
        """
        
        self.reps = reps
    
    def operate(self,tensor):
        """Apply the operator over an initial state.
        
        Args:
            tensor: Initial tensor state.
        
        Returns:
            tensor_qft: Final tensor after the operations.
        """
            
        tensor_qft = tensor
            
        for rep in range(self.reps):
            axis = rep + 1
            tensor_qft = np.fft.ifft(tensor_qft,axis=axis,norm="ortho")
        
        return tensor_qft
    
    def inverse(self):
        """Invert the unitary operator."""
        
        inverse_operator = IQFT.__new__(IQFT)
        inverse_operator.__dict__ = self.__dict__.copy()
        inverse_operator.string = 'iF'
        inverse_operator.info_string = 'iQFT'
        
        return inverse_operator

def fht(tensor,axis):
    """Calculates the fast Walsh–Hadamard Transform along a given axis.

    Args:
        tensor: Initial tensor state.
        axis: Axis along which the transform ins performed.
    
    Returns:
        tensor: Final tensor after the operations.
    """
    
    ph_dim = tensor.shape[axis]
    tensor = np.swapaxes(tensor,axis,-1)
    base_shape = tensor.shape
    num_stages = int(np.log2(ph_dim))

    for s in range(num_stages):
        block_size = 2**s
        tensor = tensor.reshape(*base_shape[:-1],ph_dim//(2*block_size),2*block_size)
        
        a = tensor[...,:block_size].copy()
        b = tensor[...,block_size:2 * block_size].copy()
        
        tensor[...,:block_size] = a+b
        tensor[...,block_size:2*block_size] = a-b
        tensor = tensor.reshape(base_shape)

    tensor /= np.sqrt(ph_dim)

    return np.swapaxes(tensor,axis,-1)

class Hadamard(QPEOperator):
    """Hadamard gates operator.

    Attributes:
      string: A string representing the operator in an algebraic form.
      info_string: A string with the information of the operator.
      class_type: Kind of class.
      reps: Number of phase registers.
    """
    
    string = 'H'
    info_string = 'Hadamard'
    
    def __init__(self,reps=1):
        """Initializes the Hadamard operator.

        Args:
            reps: Number of phase registers. Default: 1.
        """
        
        self.reps = reps
    
    def operate(self,tensor):
        """Apply the operator over an initial state.
        
        Args:
            tensor: Initial tensor state.
        
        Returns:
            tensor_h: Final tensor after the operations.
        """
        
        tensor_h = tensor
        
        for rep in range(self.reps):
            tensor_h = fht(tensor_h,axis=rep+1)
        
        return tensor_h

class SzegedyMeasurement():
    """Class to obtain the probability distributions of the Szegedy register.

    Attributes:
      operator: A Szegedy unitary operator.
    """
    
    def __init__(self,operator):
        """Initializes the Szegedy measurement operator.

        Args:
            operator: A Szegedy unitary operator.
        """
        
        self.operator = operator
    
    def operate(self,tensor):
        """Measure an initial state in the Szegedy register.
        
        Args:
            tensor: Initial tensor state.
        
        Returns:
            measure: Probability distribution after the measurement.
        """
        
        shape = tensor.shape
        tensor = self.operator.operate(tensor.reshape(shape[0],np.prod(shape[1:])))
        measure = np.sum(tensor,axis=1)
        
        return measure

class PhaseMeasurement():
    """Class to obtain the probability distributions of the phase registers.

    Attributes:
      reps: Number of phase registers.
    """
    
    def __init__(self,reps=1):
        """Initializes the phase measurement operator.

        Args:
            reps: Number of phase registers. Default: 1.
        """
        
        self.reps = reps
    
    def operate(self,tensor,binary=False):
        """Measure an initial state in the phase registers.
        
        Args:
            tensor: Initial tensor state.
            binary: If True, the result is a dicionary with the probaiblity for each bitstring.
        
        Returns:
            measure: Probability distribution after the measurement.
        """
        
        shape = tensor.shape
        tensor = tensor.reshape(shape[0],np.prod(shape[1:]))
        measure = np.sum(abs(tensor)**2,axis=0)
        
        if binary == False:
            
            return measure
        
        else:
            
            measure_dict = {}
            ph_qubits = int(np.log2(len(measure))) // self.reps
            
            for num_decimal in range(len(measure)):
                num_binary = format(num_decimal,f'0{int(np.log2(len(measure)))}b')
                num_binary = ' '.join(num_binary[i*ph_qubits:(i+1)*ph_qubits] for i in range(self.reps))
                measure_dict[num_binary] = float(measure[num_decimal])
            
            return measure_dict

class SzegedyOperator(QPEOperator):
    """Class to simulate a unitary evolution in the Szegedy register.

    Attributes:
      operator: A Szegedy unitary operator.
      string: A string representing the operator in an algebraic form.
      info_string: A string with the information of the operator.
      class_type: Kind of class.
    """
    
    def __init__(self,operator):
        """Initializes the Szegedy operator.

        Args:
            operator: A Szegedy unitary operator.
        """
    
        self.string = operator.string
        self.info_string = operator.info_string
        self.operator = operator
    
    def operate(self,tensor):
        """Apply the operator over an initial state.
        
        Args:
            tensor: Initial tensor state.
        
        Returns:
            tensor: Final tensor after the operations.
        """
        
        shape = tensor.shape
        tensor = self.operator.operate(tensor.reshape(shape[0],np.prod(shape[1:]))).reshape(shape)
        return tensor
    
    def inverse(self):
        """Invert the unitary operator."""
        
        inverse_operator = self.__class__.__new__(self.__class__)
        inverse_operator.__dict__ = self.__dict__.copy()
        inverse_operator.operator = self.operator.inverse()
        inverse_operator.string = inverse_operator.operator.string
        inverse_operator.info_string = inverse_operator.operator.info_string
        
        return inverse_operator

class ReflectionPhase0(QPEOperator):
    """Class to simulate a reflection around the state 0 in the phase registers.

    Attributes:
      ph_qubits: Number of qubits of a single phase register.
      ph_dim: Dimension of a single phase register.
      reps: Number of phase registers.
    """
    
    string = 'RPh0'
    info_string = 'Reflection Phase 0'
    
    def operate(self,tensor):
        """Apply the operator over an initial state.
        
        Args:
            tensor: Initial tensor state.
        
        Returns:
            tensor: Final tensor after the operations.
        """
        
        tensor *= -1
        idx = (slice(None),) + (0,) * (tensor.ndim - 1)
        tensor[idx] *= -1
        
        return tensor