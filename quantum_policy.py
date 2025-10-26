import pennylane as qml
import torch
import torch.nn as nn

class QuantumPolicy(nn.Module):
    def __init__(self, n_qubits=6, n_actions=4, n_layers=4, preprocessing_dim=64, backend="default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.n_layers = n_layers

        # Larger preprocessing for richer representation
        self.pre_fc = nn.Sequential(
            nn.Linear(n_qubits, preprocessing_dim),
            nn.ReLU(),
            nn.Linear(preprocessing_dim, n_qubits),
            nn.ReLU()
        )

        self.dev = qml.device(backend, wires=n_qubits)

        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
                for i in range(n_qubits):
                    qml.RY(weights[l][i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.q_layer = qml.qnn.TorchLayer(
            qml.QNode(circuit, self.dev, interface="torch"),
            weight_shapes
        )

        self.policy_head = nn.Sequential(
            nn.Linear(n_qubits, n_actions),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x_proj = self.pre_fc(x)
        quantum_out = self.q_layer(x_proj)
        policy = self.policy_head(quantum_out)
        value = self.value_head(quantum_out)
        return policy, value.squeeze(-1)
