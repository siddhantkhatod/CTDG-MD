from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .egnn import MLP
from .graph import RadialBasis, radius_graph, scatter_mean


class StaticNodeEncoder(nn.Module):
    """Static atom/residue/role encoder shared by non-temporal baselines."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.atom = nn.Embedding(128, hidden_dim, padding_idx=0)
        self.role = nn.Embedding(2, hidden_dim)
        self.residue = nn.Embedding(64, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, atoms: Tensor, roles: Tensor, residues: Tensor) -> Tensor:
        return self.norm(
            self.atom(atoms.clamp(0, 127))
            + self.role(roles.clamp(0, 1))
            + self.residue(torch.remainder(residues.clamp_min(0), 64))
        )


def role_pool(hidden: Tensor, roles: Tensor) -> Tensor:
    protein = hidden[roles == 0]
    ligand = hidden[roles == 1]
    if protein.numel() == 0 or ligand.numel() == 0:
        raise ValueError("Every graph requires protein and ligand atoms")
    return torch.cat([hidden.mean(0), protein.mean(0), ligand.mean(0)], dim=-1)


class ConstantBaseline(nn.Module):
    """Trainable intercept baseline for honest mean/majority comparisons."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.energy_bias = nn.Parameter(torch.zeros(()))
        self.state_bias = nn.Parameter(torch.zeros(()))

    def forward(self, batch: dict[str, Tensor], memory: Tensor | None = None):
        from .model import TrajectoryOutput

        del memory
        batch_size, length, _nodes, _ = batch["positions"].shape
        position_anchor = 0.0 * batch["positions"].sum(dim=(-1, -2))
        energy = self.energy_bias.expand(batch_size, length) + position_anchor
        state = self.state_bias.expand(batch_size, length)
        embeddings = self.energy_bias.expand(batch_size, length, 3 * self.config.hidden_dim)
        return TrajectoryOutput(energy, state, embeddings, batch["positions"].clone(), None)


class SchNetInteraction(nn.Module):
    def __init__(self, hidden_dim: int, radial_basis: int, dropout: float) -> None:
        super().__init__()
        self.filter_network = MLP(radial_basis, hidden_dim, hidden_dim, dropout)
        self.source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update = MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: Tensor, rbf: Tensor, edge_index: Tensor) -> Tensor:
        source, destination = edge_index
        message = self.source(hidden[source]) * self.filter_network(rbf)
        aggregate = scatter_mean(message, destination, hidden.shape[0])
        return self.norm(hidden + self.update(aggregate))


class SchNetBaseline(nn.Module):
    """Self-contained continuous-filter SchNet comparator on identical pocket graphs."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = StaticNodeEncoder(config.hidden_dim)
        self.rbf = RadialBasis(config.radial_basis, config.cutoff_angstrom)
        self.interactions = nn.ModuleList(
            [
                SchNetInteraction(config.hidden_dim, config.radial_basis, config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.energy_head = MLP(3 * config.hidden_dim, config.hidden_dim, 1, config.dropout)
        self.state_head = MLP(3 * config.hidden_dim, config.hidden_dim, 1, config.dropout)

    def forward(self, batch: dict[str, Tensor], memory: Tensor | None = None):
        from .model import TrajectoryOutput

        del memory
        positions_all = batch["positions"]
        mask_all = batch["node_mask"]
        batch_size, length, nodes, _ = positions_all.shape
        energy = positions_all.new_zeros((batch_size, length))
        state = positions_all.new_zeros((batch_size, length))
        embeddings = positions_all.new_zeros((batch_size, length, 3 * self.config.hidden_dim))
        for b in range(batch_size):
            count = int(mask_all[b].sum())
            atoms = batch["atom_numbers"][b, :count]
            roles = batch["roles"][b, :count]
            residues = batch["residue_ids"][b, :count]
            initial = self.encoder(atoms, roles, residues)
            for t in range(length):
                edge_index, distances = radius_graph(
                    positions_all[b, t, :count],
                    self.config.cutoff_angstrom,
                    self.config.max_neighbors,
                )
                hidden = initial
                radial = self.rbf(distances)
                for interaction in self.interactions:
                    hidden = interaction(hidden, radial, edge_index)
                pooled = role_pool(hidden, roles)
                energy[b, t] = self.energy_head(pooled).squeeze(-1)
                state[b, t] = self.state_head(pooled).squeeze(-1)
                embeddings[b, t] = pooled
        return TrajectoryOutput(energy, state, embeddings, positions_all.clone(), None)


class PaiNNInteraction(nn.Module):
    """Scalar/vector equivariant interaction block following PaiNN's core design."""

    def __init__(self, hidden_dim: int, radial_basis: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.filter_network = MLP(radial_basis, hidden_dim, 3 * hidden_dim, dropout)
        self.scalar_source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.vector_source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scalar_update = MLP(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.vector_gate = nn.Linear(hidden_dim, hidden_dim)
        self.vector_mix = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        scalar: Tensor,
        vector: Tensor,
        positions: Tensor,
        rbf: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source, destination = edge_index
        relative = positions[destination] - positions[source]
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True).clamp_min(1e-8)
        direction = relative / distance
        scalar_filter, vector_filter, radial_filter = self.filter_network(rbf).chunk(3, dim=-1)
        source_scalar = self.scalar_source(scalar[source])
        scalar_message = scalar_filter * source_scalar
        vector_message = (
            vector[source] * vector_filter[:, None, :]
            + direction[:, :, None] * (radial_filter * source_scalar)[:, None, :]
        )
        scalar = scalar + scatter_mean(scalar_message, destination, scalar.shape[0])
        vector_flat = vector_message.reshape(vector_message.shape[0], -1)
        vector_aggregate = scatter_mean(vector_flat, destination, scalar.shape[0]).reshape_as(vector)
        vector = vector + vector_aggregate

        mixed = self.vector_mix(vector)
        invariant_norm = torch.sqrt((mixed * mixed).sum(dim=1) + 1e-8)
        scalar = self.norm(scalar + self.scalar_update(torch.cat([scalar, invariant_norm], dim=-1)))
        vector = vector + mixed * self.vector_gate(scalar)[:, None, :]
        return scalar, vector


class PaiNNBaseline(nn.Module):
    """Self-contained PaiNN scalar/vector baseline on the same dynamic radius graph."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = StaticNodeEncoder(config.hidden_dim)
        self.rbf = RadialBasis(config.radial_basis, config.cutoff_angstrom)
        self.interactions = nn.ModuleList(
            [
                PaiNNInteraction(config.hidden_dim, config.radial_basis, config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.energy_head = MLP(3 * config.hidden_dim, config.hidden_dim, 1, config.dropout)
        self.state_head = MLP(3 * config.hidden_dim, config.hidden_dim, 1, config.dropout)

    def forward(self, batch: dict[str, Tensor], memory: Tensor | None = None):
        from .model import TrajectoryOutput

        del memory
        positions_all = batch["positions"]
        mask_all = batch["node_mask"]
        batch_size, length, nodes, _ = positions_all.shape
        energy = positions_all.new_zeros((batch_size, length))
        state = positions_all.new_zeros((batch_size, length))
        embeddings = positions_all.new_zeros((batch_size, length, 3 * self.config.hidden_dim))
        for b in range(batch_size):
            count = int(mask_all[b].sum())
            atoms = batch["atom_numbers"][b, :count]
            roles = batch["roles"][b, :count]
            residues = batch["residue_ids"][b, :count]
            initial = self.encoder(atoms, roles, residues)
            for t in range(length):
                positions = positions_all[b, t, :count]
                edge_index, distances = radius_graph(
                    positions, self.config.cutoff_angstrom, self.config.max_neighbors
                )
                scalar = initial
                vector = positions.new_zeros((count, 3, self.config.hidden_dim))
                radial = self.rbf(distances)
                for interaction in self.interactions:
                    scalar, vector = interaction(scalar, vector, positions, radial, edge_index)
                pooled = role_pool(scalar, roles)
                energy[b, t] = self.energy_head(pooled).squeeze(-1)
                state[b, t] = self.state_head(pooled).squeeze(-1)
                embeddings[b, t] = pooled
        return TrajectoryOutput(energy, state, embeddings, positions_all.clone(), None)
