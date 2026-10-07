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

This package is a simulator of the Szegedy Quantum Walk allowing the efficient simulation
on sparse graphs, and also expanding the library SQUWALS for dense graphs with further operators.
Moreover, it also provides a module for the simulation of quantum phase estimation algorithms based on Szegedy quantum walk.
"""

__version__ = '1.0'

import warnings

from sqwlib.sparse import *
import sqwlib.qpe as qpe

__all__ = [
    'SpaceData',
    'qpe.direct_qpe',
    'qpe.create_qpe_tensor',
    'qpe.QPEUnitary',
    'qpe.QPEOperator',
    'qpe.Hadamard',
    'qpe.Controlled',
    'qpe.IQFT',
    'qpe.QFT',
    'qpe.SzegedyMeasurement',
    'qpe.PhaseMeasurement',
    'qpe.SzegedyOperator',
    'qpe.ReflectionPhase0',]

try:
    import sqwlib .squwals_mod as squwals
    __all__.append('squwals')
except ModuleNotFoundError:
    warnings.warn("SQUWALS is not installed. Dense simulator framework not available.")
