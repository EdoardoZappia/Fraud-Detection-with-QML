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

from fmQSVC import fmQSVC

if __name__ == "__main__":

    # dataset managenet -> can be contained in a function
    file_path = "dataset/dataset.csv"
    try:
        df = pd.read_csv(file_path)
        print("Dataset imported successfully.")
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        exit()
    
    df.drop(columns=['step'], inplace=True)
    df.drop(columns=['zipcodeOri'], inplace=True)
    df.drop(columns=['zipMerchant'], inplace=True)
    df.drop(columns=['customer'], inplace=True)
    df.drop(columns=['merchant'], inplace=True)

    encoder = LabelEncoder()
    encoded_age = encoder.fit_transform(df['age'])
    df['age'] = encoded_age
    encoded_gender = encoder.fit_transform(df['gender'])
    df['gender'] = encoded_gender
    encoded_category = encoder.fit_transform(df['category'])
    df['category'] = encoded_category

    X = df.drop('fraud', axis=1)
    y = df['fraud']

    sampling_strategy = {0: 100, 1: 100}
    # Undersample the majority class
    undersample = RandomUnderSampler(sampling_strategy = sampling_strategy, random_state = 12345)
    X_resampled, y_resampled = undersample.fit_resample(X, y)
    # Transform all the features in the interval [0, 1]
    X_resampled = MinMaxScaler().fit_transform(X_resampled)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)

    num_features = X.shape[1]
    # create the feature maps
    Z = ZFeatureMap(feature_dimension = num_features, reps = 1)
    ZZ = ZZFeatureMap(feature_dimension = num_features, reps = 1)
    P = PauliFeatureMap(feature_dimension = num_features, reps = 1, paulis = ['Z', 'XY'])
    Z2 = ZFeatureMap(feature_dimension = num_features, reps = 2)
    ZZ2 = ZZFeatureMap(feature_dimension = num_features, reps = 2)

    # Create a parametrized layer to train. We can rotate each qubit the same amount, or different amount.
    training_params = ParameterVector("θ", num_features)
    ansatz_layer = QuantumCircuit(num_features)
    for i in range(num_features):
        ansatz_layer.ry(training_params[i], i)

    # Create the trainable feature maps, composed of two circuits: the ansatz and the previous feature map
    Z_param   = ansatz_layer.compose(Z)
    ZZ_param  = ansatz_layer.compose(ZZ)
    P_param   = ansatz_layer.compose(P)
    Z2_param  = ansatz_layer.compose(Z2)
    ZZ2_param = ansatz_layer.compose(ZZ2)

    parametric_feature_map_dict = {'Z': Z_param, 'ZZ': ZZ_param, 'Pauli(Z, XY)': P_param, 'Z with 2 layers': Z2_param, 'ZZ with 2 layers': ZZ2_param}

    fm_qsvc = fmQSVC(parametric_feature_map_dict, training_params, name = "test")
    fm_qsvc.optimize_kernels(X_train, y_train)
    fm_qsvc.run(X_train, y_train, X_test, y_test)
