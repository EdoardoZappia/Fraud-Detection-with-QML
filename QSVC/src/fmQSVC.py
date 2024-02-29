import qsvc_modules

# callback class for SPSA and QNSPSA optimizers
class QKTCallback:

    def __init__(self) -> None:
        self._data = [[] for i in range(5)]

    def callback(self, x0, x1 = None, x2 = None, x3 = None, x4 = None):
        """
        Args (from the qiskit documentation):
            x0: number of function evaluations
            x1: the parameters
            x2: the function value
            x3: the stepsize
            x4: whether the step was accepted
        """
        self._data[0].append(x0)
        self._data[1].append(x1)
        self._data[2].append(x2)
        self._data[3].append(x3)
        self._data[4].append(x4)

    def get_callback_data(self):
        return self._data

    def clear_callback_data(self):
        self._data = [[] for i in range(5)]



# quantum support vector classifier class
class fmQSVC(object):

    def __init__(self, feature_map_dict, training_params = None, name = "default", local = True):
        self.feature_map_dict = feature_map_dict
        self.training_params = training_params
        self.name = name
        self.local = local
        self.optimized_kernels = {}
        self.callback_data = []

    # initialize the kernels' dictionary for all the feature maps, in case of FidelityQuantumKernel(s)
    # in this case there is no parameter optimization
    def init_kernels(self):
        sampler = Sampler()
        fidelity = ComputeUncompute(sampler = sampler)
        for feature_map_name, feature_map in self.feature_map_dict.items():
            self.optimized_kernels[feature_map_name] = FidelityQuantumKernel(fidelity = fidelity, feature_map = feature_map)

    # optimize the quantum kernels in case of TrainableQuantumKernel(s) 
    # -> find the optimal combination of parameters of the ansatz-feature_map
    def optimize_kernels(self, X_train, y_train):
        if (self.local):
            # local run: instantiate the Sampler class from qiskit
            sampler = Sampler()
        else:
            # run on ibm quntum, import the Sampler class from IBM Qiskit Runtime
            service = IBMR.QiskitRuntimeService()#channel = "ibm_quantum")
            backend = service.get_backend("ibmq_qasm_simulator")
            # Set options to (eventually to include noise models)
            options = IBMR.Options()
            options.execution.shots = 1000
            options.optimization_level = 3
            options.resilience_level = 0
            sampler = IBMR.Sampler(backend = backend)

        fidelity = ComputeUncompute(sampler = sampler)
        qkt_callback = QKTCallback()
        optimizer = SPSA(
            maxiter = 20, 
            callback = qkt_callback.callback, 
            learning_rate = 0.05, 
            perturbation = 0.05
        )
        #optimizer = COBYLA(maxiter = 20)

        for feature_map_name, feature_map in self.feature_map_dict.items():
            quantum_kernel = TrainableFidelityQuantumKernel(
                fidelity = fidelity, 
                feature_map = feature_map, 
                training_parameters = self.training_params
            )
            qkt = QuantumKernelTrainer(
                quantum_kernel = quantum_kernel, 
                loss = "svc_loss", 
                optimizer = optimizer, 
                initial_point = [np.pi / 2] * len(self.training_params)
            )
            # Train the kernel
            qkt_results = qkt.fit(X_train, y_train)
            self.optimized_kernels[feature_map_name] = qkt_results.quantum_kernel
            #optimized_kernels.append(optimized_kernel)
            self.callback_data.append(qkt_callback.get_callback_data())
            qkt_callback.clear_callback_data()
    
    def run(self, X_train, y_train, X_test, y_test, write = True, plot = True):
        for feature_map_name, optimized_kernel in self.optimized_kernels.items():
            qsvc = QSVC(quantum_kernel = optimized_kernel)
            # Fit the QSVC
            qsvc.fit(X_train, y_train)
            if (write):
                evaluate_with_feature_map(qsvc, X_train, y_train.values, X_test, y_test.values, feature_map_name)
        if (plot):
            plot_ROC_DET(X_train, y_train, X_test, y_test)

    #@staticmethod
    # Perform predictions and evaluate performance metrics
    def evaluate_with_feature_map(self, model, X_train, y_train, X_test, y_test, feature_map_name):
        filename = f"trainable_QSVC_{self.name}.txt"
        file = open(filename, "w")
        file.write(f"Evaluating performance for {feature_map_name} Feature Map:\n")

        # Perform predictions on training and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Print classification report and confusion matrix for training set
        file.write(f"Classification Report for {feature_map_name} Feature Map (Train Set):\n")
        file.write(classification_report(y_train, y_train_pred))
        file.write("Confusion Matrix for Train Set:")
        file.write(confusion_matrix(y_train, y_train_pred))
        file.write("\n")

        # write classification report and confusion matrix for test set
        file.write(f"Classification Report for {feature_map_name} Feature Map (Test Set):\n")
        file.write(classification_report(y_test, y_test_pred))
        file.write("Confusion Matrix for Test Set:")
        file.write(confusion_matrix(y_test, y_test_pred))

        # write ROC AUC score for test set
        file.write(f"ROC AUC Score for {feature_map_name} Feature Map (Test Set):")
        file.write(round(roc_auc_score(y_test, y_test_pred), 4))
        file.write(70*'=')
        file.close()

    #@staticmethod
    def plot_ROC_DET(self, X_train, y_train, X_test, y_test):
        fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize = (22, 10))
        for feature_map_name, optimized_kernel in self.optimized_kernels.items():
            qsvc = QSVC(quantum_kernel = optimized_kernel)
            # Fit the QSVC
            qsvc.fit(X_train, y_train.values)
            RocCurveDisplay.from_estimator(qsvc, X_test, y_test.values, ax = ax_roc, name = feature_map_name, lw = 2)
            DetCurveDisplay.from_estimator(qsvc, X_test, y_test.values, ax = ax_det, name = feature_map_name, lw = 2)
        
        ax_roc.set_title("ROC curves")
        ax_det.set_title("DET curves")
        ax_roc.grid(linestyle = "--")
        ax_det.grid(linestyle = "--")
        ax_roc.axline((0, 0), slope = 1, color = 'k', linestyle = '--')
        ax_det.axline((0, 0), slope = -1, color = 'k', linestyle = '--')
        plt.legend()
        plt.savefig(f"ROC_DET_{self.name}.png")


class QSVC_embedded(object):
    
    def __init__(self):
        print("developement...")