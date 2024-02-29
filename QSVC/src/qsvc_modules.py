import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn import linear_model, svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import roc_curve, DetCurveDisplay, RocCurveDisplay

from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap, RealAmplitudes
from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM, QNSPSA, GradientDescent
from qiskit.primitives import Sampler, Estimator
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit.quantum_info import Statevector
from qiskit.visualization import circuit_drawer
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit import ParameterVector
from qiskit.providers import Options

from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer

import qiskit_ibm_runtime as IBMR
from qiskit_ibm_provider import IBMProvider

from IPython.display import clear_output

import matplotlib

import random