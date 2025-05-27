import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
import numpy as np
import pennylane as qml
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


##########################################################################################
#############################    Condition Encoder   #####################################
##########################################################################################
class ConditionEncoder(nn.Module):
    def __init__(self, input_dim, n_condition_qubits, block_size, stride):
        """
        Partition the input vector into overlapping blocks.
        For each block k, compute:
            μ_k = (1/b) ∑ x_i
            σ²_k = (1/b) ∑ (x_i - μ_k)²
        Then compute:
            θ_k = w₁ μ_k + w₂ σ²_k + c,
        and bound via:
            θ'_k = π tanh(θ_k).
        Adaptive pooling ensures exactly n_condition_qubits outputs.
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_condition_qubits = n_condition_qubits
        self.block_size = block_size
        self.stride = stride
        self.n_blocks = (input_dim - block_size) // stride + 1
        self.linear = nn.Linear(2, 1)

    def forward(self, x):  # x: (B, input_dim)
        B, D = x.shape
        features = []
        for k in range(self.n_blocks):
            start = k * self.stride
            end = start + self.block_size
            block = x[:, start:end]  # shape: (B, block_size)
            m = torch.mean(block, dim=1)  # shape: (B,)
            v = torch.var(block, dim=1, unbiased=False)  # avoid NaNs
            features.append(torch.stack([m, v], dim=1))  # (B, 2)

        features = torch.stack(features, dim=1)  # shape: (B, n_blocks, 2)
        theta = self.linear(features).squeeze(-1)  # shape: (B, n_blocks)
        theta = torch.tanh(theta) * np.pi  # bound to [-π, π]

        if self.n_blocks != self.n_condition_qubits:
            theta = F.adaptive_avg_pool1d(theta.unsqueeze(1), self.n_condition_qubits).squeeze(1)
            # input: (B, 1, n_blocks) → output: (B, n_condition_qubits)

        return theta  # shape: (B, n_condition_qubits)


##########################################################################################
#############################     Quantum Circuit    #####################################
##########################################################################################
class QC:
    def __init__(self, circuit, q_depth, n_qubits, dev='default.qubit', diff_method='backprop'):
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.circuit = circuit
        self.quantum_circuit = None
        self.diff_method = diff_method
        self.quantum_circuit = self.quantum_circuit_1 if self.circuit == 1 else self.quantum_circuit_2
        self.dev = qml.device(dev, wires=n_qubits)
        self.qnode = qml.QNode(self.quantum_circuit, self.dev, interface='torch', diff_method=diff_method)

    def quantum_circuit_1(self, weights, effective_angles):
        """
        The circuit applies an initial embedding with RY(effective_angles) on each qubit.
        followed by entangling layers consisting of CNOT and RY gates,
        and re-uploads the effective angles.
        """
        weights = weights.reshape(self.q_depth, self.n_qubits)
        for i in range(self.n_qubits):
            qml.RY(effective_angles[i], wires=i)

        for d in range(self.q_depth):
            for i in range(self.n_qubits):
                qml.RY(weights[d][i], wires=i)

            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

        # Reuploadinhg
        for i in range(self.n_qubits):
            qml.RY(effective_angles[i], wires=i)

        return qml.probs(wires=list(range(self.n_qubits)))

    def quantum_circuit_2(self, weights, effective_angles):
        """
        The circuit applies an initial embedding with RY(effective_angles) on each qubit.
        Then, for each layer, it applies parameterized rotations, all-to-all CRY entanglement,
        and re-uploads the effective angles.
        """
        weights = weights.reshape(self.q_depth, self.n_qubits)
        for i in range(self.n_qubits):
            qml.RY(effective_angles[i], wires=i)

        for layer in range(self.q_depth):
            for i in range(self.n_qubits):
                qml.RY(weights[layer][i], wires=i)
            for control in range(self.n_qubits):
                for target in range(self.n_qubits):
                    if control != target:
                        qml.CRY(0.5, wires=[control, target])

            for i in range(self.n_qubits):
                qml.RY(effective_angles[i], wires=i)

        return qml.probs(wires=list(range(self.n_qubits)))

    def draw_circuit(self, output='ascii'):
        if output == 'mpl':
            fig, ax = qml.draw_mpl(self.quantum_circuit)(torch.randn(self.q_depth * self.n_qubits), torch.randn(self.n_qubits))
            fig.show()
        else:
            print(qml.draw(self.quantum_circuit)(torch.randn(self.q_depth * self.n_qubits), torch.randn(self.n_qubits)))

    def test_circuit(self):
        return self.qnode(torch.randn(self.q_depth * self.n_qubits), torch.randn(self.n_qubits))

    def get_circuit(self):
        return self.quantum_circuit

    def forward(self, weights_batch: torch.Tensor, angles_batch: torch.Tensor) -> torch.Tensor:
        """
        weights_batch: (B, depth * n_qubits)
        angles_batch:  (B, n_qubits)
        returns:       (B, 2**n_qubits) tensor of probabilities
        """
        batch_size = weights_batch.size(0)
        probs_list = []

        for b in range(batch_size):
            probs_list.append(self.qnode(weights_batch[b], angles_batch[b]))
        # stack into a (B, 2**n_qubits) tensor
        return torch.stack(probs_list, dim=0)

#############################    Graph Convolution   #####################################
class GraphConvolution(Module):

    def __init__(self, in_features, out_feature_list, b_dim, dropout):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_feature_list = out_feature_list

        self.linear1 = nn.Linear(in_features, out_feature_list[0])
        self.linear2 = nn.Linear(out_feature_list[0], out_feature_list[1])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, adj, activation=None):

        hidden = torch.stack([self.linear1(input) for _ in range(adj.size(1))], 1)
        hidden = torch.einsum('bijk,bikl->bijl', (adj, hidden))
        hidden = torch.sum(hidden, 1) + self.linear1(input)
        hidden = activation(hidden) if activation is not None else hidden
        hidden = self.dropout(hidden)

        output = torch.stack([self.linear2(hidden) for _ in range(adj.size(1))], 1)
        output = torch.einsum('bijk,bikl->bijl', (adj, output))
        output = torch.sum(output, 1) + self.linear2(hidden)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output
#############################    Graph Aggregation   #####################################
class GraphAggregation(Module):

    def __init__(self, in_features, out_features, b_dim, dropout):
        super(GraphAggregation, self).__init__()
        self.sigmoid_linear = nn.Sequential(nn.Linear(in_features+b_dim, out_features),
                                            nn.Sigmoid())
        self.tanh_linear = nn.Sequential(nn.Linear(in_features+b_dim, out_features),
                                         nn.Tanh())
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, activation):
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        output = torch.sum(torch.mul(i,j), 1)
        output = activation(output) if activation is not None\
                 else output
        output = self.dropout(output)

        return output


##########################################################################################
###########################    Generator Architecture   ##################################
##########################################################################################
class HQWcGANGP_Generator(nn.Module):
    def __init__(self, conv_dims, vertexes, edges, nodes, dropout, qc, n_condition_qubits, n_noise_qubits, n_a_qubits, n_generators=None, q_delta=1): # n_generators deprecated
        super().__init__()
        self.q_depth = qc.q_depth
        self.n_qubits = n_condition_qubits + n_noise_qubits + n_a_qubits
        self.n_condition_qubits = n_condition_qubits
        self.n_noise_qubits = n_noise_qubits
        self.q_params = nn.ParameterList([
            nn.Parameter(q_delta * torch.rand(self.q_depth * self.n_qubits), requires_grad=True)
            #for _ in range(n_generators)
        ])
        #self.n_generators = n_generators
        self.patch_size = 2 ** (self.n_qubits - n_a_qubits)

        self.qc = qc
        self.z_dim = self.patch_size #* n_generators

        self.listify = lambda x: [x] if not (isinstance(x, list) or isinstance(x, tuple)) else x
        self.delistify = lambda x: x[0] if len(x) == 1 else x
        self.postprocess = lambda inputs: self.delistify([F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                       / 1., hard=True).view(e_logits.size())
                       for e_logits in self.listify(inputs)])

        classical_layers = []
        for in_dim, out_dim in zip([self.z_dim] + conv_dims[:-1], conv_dims):
            classical_layers += [nn.Linear(in_dim, out_dim), nn.Tanh(), nn.Dropout(dropout)]
        self.layers = nn.Sequential(*classical_layers)

        final_dim = conv_dims[-1]
        self.edges_layer = nn.Linear(final_dim, edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(final_dim, vertexes * nodes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, condition_batch, noise_batch):
        batch_size = condition_batch.size(0)
        angles = torch.cat([condition_batch, noise_batch], dim=1) # (B, n_qubits)
        # cond:  torch.Size([748])
        # noise:  torch.Size([2])
        # params:  torch.Size([6])
        patches = []
        for params in self.q_params:
            w_batch = params.unsqueeze(0).expand(batch_size, -1) # expand to (B, depth*Q)
            probs = self.qc.forward(w_batch, angles) # (B, 2**n_qubits)
            patch = probs[:, :self.patch_size] # slicing
            patch /= torch.sum(probs)  # Normalize per sample
            patch = patch / torch.max(patch)
            patches.append(patch)

        # (B, N_G * patch_size)
        z = torch.cat(patches, dim=1)
        z = z.to(torch.float32)

        # Classical processing
        x = self.layers(z)  # (batch, 512)

        edges_logits = self.edges_layer(x).view(-1, edges, vertexes, vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropout(edges_logits.permute(0, 2, 3, 1))  # (batch, 64, 64, 5)

        nodes_logits = self.nodes_layer(x).view(-1, vertexes, nodes)  # (batch, 64, 68)
        nodes_logits = self.dropout(nodes_logits)

        return edges_logits, nodes_logits

##########################################################################################
#########################    Discriminator Architecture   ################################
##########################################################################################
class Discriminator(nn.Module):
    def __init__(self, conv_dim, m_dim, b_dim, dropout,):
        super().__init__()
        graph_conv_dim, aux_dim, linear_dim = conv_dim
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, dropout)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1], aux_dim, b_dim, dropout)

        dense_layers = []
        for c0, c1 in zip([aux_dim] + linear_dim[:-1], linear_dim):
            dense_layers.append(nn.Linear(c0, c1))
            dense_layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*dense_layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, hidden, node, activation=None):
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        annotations = torch.cat((hidden, node), -1) if hidden is not None else node
        h = self.gcn_layer(annotations, adj)
        annotations = torch.cat((h, hidden, node) if hidden is not None else (h, node), -1)
        h = self.agg_layer(annotations, torch.tanh)
        h = self.linear_layer(h)
        output = self.output_layer(h)
        return activation(output) if activation else output, h

if __name__ == '__main__':
    
    block_size = 20
    stride = 10
    # n_generators = 4
    edges = 5
    vertexes = 64
    nodes = 34
    d_conv_dims = [[128, 64], 128, [128, 64]]
    g_conv_dims = [128, 256]
    dropout = 0.3
    lambda_gp = 10
    n_condition_qubits = 4
    n_noise_qubits = 2
    n_a_qubits = 0
    q_depth = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("-------------- QUANTUM CIRCUIT --------------")
    config_1 = {
        "circuit": 1,
        "n_qubits": 6,
        "q_depth": 2,
        "diff_method": 'backprop'
    }
    
    config_2 = {
        "circuit": 2,
        "n_qubits": 6,
        "q_depth": 1,
        "diff_method": 'backprop'
    }
    
    qc_1 = QC(**config_1)
    print(qc_1.test_circuit())
    qc_2 = QC(**config_2)
    print(qc_2.test_circuit())
    qc_2.draw_circuit()
    
    print("------------------ENCODER NETWORK----------------")
    enc = ConditionEncoder(input_dim=748, n_condition_qubits=4, block_size=20, stride=10).to(device)
    print(enc)
    
    
    print("-------------- GENERATOR NETWORK --------------")
    g = HQWcGANGP_Generator(g_conv_dims,
                            vertexes,
                            edges, nodes,
                            dropout=0.3,
                            qc=qc_1,
                            n_condition_qubits=4,
                            n_noise_qubits=2,
                            n_a_qubits=0,
                            n_generators=4,
                            q_delta=1).to(device)
    print(g)
    print("------------- DISCRIMINATOR NETWORK ------------")
    d = Discriminator(d_conv_dims,
                      nodes,
                      nodes,
                      dropout).to(device)
    print(d.gcn_layer)
    print(d.agg_layer)
    print(d)
