from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .egnn import MLP, EGNNLayer
from .graph import RadialBasis, edge_categories, radius_graph
from .temporal import ContinuousTimeEncoding, DecayedNodeMemory, PatchFourierTransform
from .virtual_nodes import augment_with_temporal_virtual_nodes


@dataclass
class TrajectoryOutput:
    energy: Tensor
    state_logits: Tensor
    embeddings: Tensor
    coordinates: Tensor
    memory: Tensor | None


class NodeEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.atom = nn.Embedding(128, hidden_dim, padding_idx=0)
        self.role = nn.Embedding(2, hidden_dim)
        self.residue = nn.Embedding(64, hidden_dim)
        self.speed = MLP(1, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, atom_numbers: Tensor, roles: Tensor, residues: Tensor, speed: Tensor) -> Tensor:
        atoms = atom_numbers.clamp(0, 127)
        residue_index = torch.remainder(residues.clamp_min(0), 64)
        hidden = (
            self.atom(atoms)
            + self.role(roles.clamp(0, 1))
            + self.residue(residue_index)
            + self.speed(speed.unsqueeze(-1))
        )
        return self.norm(hidden)


class MolecularGraphBackbone(nn.Module):
    def __init__(self, config: ModelConfig, use_virtual_nodes: bool) -> None:
        super().__init__()
        self.config = config
        self.use_virtual_nodes = use_virtual_nodes
        self.node_encoder = NodeEncoder(config.hidden_dim)
        self.rbf = RadialBasis(config.radial_basis, config.cutoff_angstrom)
        self.time_encoder = ContinuousTimeEncoding(config.time_dim)
        edge_dim = config.radial_basis + 5
        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    config.hidden_dim,
                    config.message_dim,
                    edge_dim,
                    config.time_dim,
                    config.coordinate_scale,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        readout_dim = 3 * config.hidden_dim
        self.energy_head = MLP(readout_dim, config.hidden_dim, 1, config.dropout)
        self.state_head = MLP(readout_dim, config.hidden_dim, 1, config.dropout)

    def _frame_graph(
        self,
        hidden: Tensor,
        positions: Tensor,
        roles: Tensor,
        delta_t: Tensor,
        frame_seed: int,
    ) -> tuple[Tensor, Tensor]:
        edge_index, _ = radius_graph(
            positions,
            cutoff=self.config.cutoff_angstrom,
            max_neighbors=self.config.max_neighbors,
            min_neighbors=1,
        )
        physical_nodes = hidden.shape[0]
        if self.use_virtual_nodes:
            k = min(self.config.num_virtual_nodes, physical_nodes)
            hidden, positions, edge_index, _ = augment_with_temporal_virtual_nodes(
                hidden,
                positions,
                edge_index,
                k=k,
                iterations=self.config.vn_kmeans_iterations,
                tolerance=self.config.vn_tolerance,
                seed=frame_seed,
            )
            virtual_start: int | None = physical_nodes
        else:
            virtual_start = None

        source, destination = edge_index
        distances = torch.linalg.vector_norm(positions[destination] - positions[source], dim=-1)
        edge_features = torch.cat(
            [self.rbf(distances), edge_categories(roles, edge_index, virtual_start)], dim=-1
        )
        time_features = self.time_encoder(delta_t.reshape(()))
        for layer in self.layers:
            hidden, positions = layer(
                hidden, positions, edge_index, edge_features, time_features
            )
        return hidden[:physical_nodes], positions[:physical_nodes]

    @staticmethod
    def _pool(hidden: Tensor, roles: Tensor) -> Tensor:
        protein = hidden[roles == 0]
        ligand = hidden[roles == 1]
        if protein.numel() == 0 or ligand.numel() == 0:
            raise ValueError("Every graph must contain at least one protein and one ligand atom")
        return torch.cat([hidden.mean(dim=0), protein.mean(dim=0), ligand.mean(dim=0)], dim=-1)

    def _heads(self, hidden: Tensor, roles: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pooled = torch.stack([self._pool(hidden[t], roles) for t in range(hidden.shape[0])])
        return self.energy_head(pooled).squeeze(-1), self.state_head(pooled).squeeze(-1), pooled


class EGNNBaseline(MolecularGraphBackbone):
    """Frame-independent EGNN energy/state predictor, optionally with k-TVNs."""

    def __init__(self, config: ModelConfig, use_virtual_nodes: bool = False) -> None:
        super().__init__(config, use_virtual_nodes)

    def forward(self, batch: dict[str, Tensor], memory: Tensor | None = None) -> TrajectoryOutput:
        del memory
        positions_all = batch["positions"]
        atom_numbers_all = batch["atom_numbers"]
        roles_all = batch["roles"]
        residues_all = batch["residue_ids"]
        mask_all = batch["node_mask"]
        times_all = batch["times"]
        batch_size, length, _max_nodes, _ = positions_all.shape
        energies = positions_all.new_zeros((batch_size, length))
        logits = positions_all.new_zeros((batch_size, length))
        embeddings = positions_all.new_zeros((batch_size, length, 3 * self.config.hidden_dim))
        output_coordinates = positions_all.clone()

        for b in range(batch_size):
            count = int(mask_all[b].sum().item())
            positions = positions_all[b, :, :count]
            atoms = atom_numbers_all[b, :count]
            roles = roles_all[b, :count]
            residues = residues_all[b, :count]
            frame_hidden: list[Tensor] = []
            frame_coordinates: list[Tensor] = []
            frame_offset = int(batch.get("frame_offset", 0))
            for t in range(length):
                # The EGNN ablation is deliberately frame independent: no speed,
                # timestamp, temporal memory, or Fourier context enters the baseline.
                speed = positions.new_zeros(count)
                delta_t = times_all[b, t].new_tensor(0.0)
                hidden = self.node_encoder(atoms, roles, residues, speed)
                hidden, coordinates = self._frame_graph(
                    hidden, positions[t], roles, delta_t, frame_seed=frame_offset + t
                )
                frame_hidden.append(hidden)
                frame_coordinates.append(coordinates)
            hidden_sequence = torch.stack(frame_hidden)
            energy, state, pooled = self._heads(hidden_sequence, roles)
            energies[b], logits[b], embeddings[b] = energy, state, pooled
            output_coordinates[b, :, :count] = torch.stack(frame_coordinates)
        return TrajectoryOutput(energies, logits, embeddings, output_coordinates, None)


class CTDGModel(MolecularGraphBackbone):
    """Continuous-time dynamic EGNN with decayed memory and patch Fourier encoding."""

    def __init__(self, config: ModelConfig, use_virtual_nodes: bool = True) -> None:
        super().__init__(config, use_virtual_nodes)
        self.memory_cell = DecayedNodeMemory(config.hidden_dim)
        self.patch_fourier = PatchFourierTransform(
            config.hidden_dim, config.patch_size, config.time_dim, config.fourier_dropout
        )

    def forward(self, batch: dict[str, Tensor], memory: Tensor | None = None) -> TrajectoryOutput:
        positions_all = batch["positions"]
        atom_numbers_all = batch["atom_numbers"]
        roles_all = batch["roles"]
        residues_all = batch["residue_ids"]
        mask_all = batch["node_mask"]
        times_all = batch["times"]
        batch_size, length, max_nodes, _ = positions_all.shape
        energies = positions_all.new_zeros((batch_size, length))
        logits = positions_all.new_zeros((batch_size, length))
        embeddings = positions_all.new_zeros((batch_size, length, 3 * self.config.hidden_dim))
        output_coordinates = positions_all.clone()
        next_memory = positions_all.new_zeros((batch_size, max_nodes, self.config.hidden_dim))

        for b in range(batch_size):
            count = int(mask_all[b].sum().item())
            positions = positions_all[b, :, :count]
            atoms = atom_numbers_all[b, :count]
            roles = roles_all[b, :count]
            residues = residues_all[b, :count]
            node_memory = (
                positions.new_zeros((count, self.config.hidden_dim))
                if memory is None
                else memory[b, :count]
            )
            frame_memory: list[Tensor] = []
            frame_coordinates: list[Tensor] = []
            previous_time_all = batch.get("previous_time")
            previous_positions_all = batch.get("previous_positions")
            frame_offset = int(batch.get("frame_offset", 0))
            for t in range(length):
                if t == 0 and memory is not None:
                    if previous_time_all is None or previous_positions_all is None:
                        raise ValueError(
                            "Carried CTDG memory requires previous_time and previous_positions "
                            "to preserve exact segmented temporal continuity"
                        )
                    delta_t = (times_all[b, 0] - previous_time_all[b]).clamp_min(1e-8)
                    speed = torch.linalg.vector_norm(
                        positions[0] - previous_positions_all[b, :count], dim=-1
                    ) / delta_t
                elif t == 0:
                    speed = positions.new_zeros(count)
                    delta_t = times_all[b, t].new_tensor(0.0)
                else:
                    delta_t = (times_all[b, t] - times_all[b, t - 1]).clamp_min(1e-8)
                    speed = torch.linalg.vector_norm(positions[t] - positions[t - 1], dim=-1) / delta_t
                # Detaching speed makes -dE/dx the conservative coordinate derivative
                # while still allowing the speed encoder itself to learn.
                observation = self.node_encoder(atoms, roles, residues, speed.detach())
                decay_rate = torch.nn.functional.softplus(self.memory_cell.log_decay)
                decayed = node_memory * torch.exp(-delta_t.reshape(1, 1) * decay_rate.reshape(1, -1))
                spatial, coordinates = self._frame_graph(
                    observation + decayed,
                    positions[t],
                    roles,
                    delta_t,
                    frame_seed=frame_offset + t,
                )
                node_memory = self.memory_cell(spatial, node_memory, delta_t)
                frame_memory.append(node_memory)
                frame_coordinates.append(coordinates)
            hidden_sequence = torch.stack(frame_memory)
            time_origin_all = batch.get("time_origin")
            time_origin = times_all[b, 0] if time_origin_all is None else time_origin_all[b]
            temporal_hidden = self.patch_fourier(hidden_sequence, times_all[b], time_origin=time_origin)
            energy, state, pooled = self._heads(temporal_hidden, roles)
            energies[b], logits[b], embeddings[b] = energy, state, pooled
            output_coordinates[b, :, :count] = torch.stack(frame_coordinates)
            next_memory[b, :count] = node_memory
        return TrajectoryOutput(energies, logits, embeddings, output_coordinates, next_memory)


def build_model(config: ModelConfig, variant: str | None = None) -> nn.Module:
    name = (variant or config.name).lower()
    if name in {"constant", "schnet", "painn"}:
        from .baselines import ConstantBaseline, PaiNNBaseline, SchNetBaseline

        baseline = {
            "constant": ConstantBaseline,
            "schnet": SchNetBaseline,
            "painn": PaiNNBaseline,
        }[name]
        return baseline(config)
    if name == "egnn":
        return EGNNBaseline(config, use_virtual_nodes=False)
    if name == "egnn_tvn":
        return EGNNBaseline(config, use_virtual_nodes=True)
    if name == "ctdg":
        return CTDGModel(config, use_virtual_nodes=False)
    if name == "ctdg_tvn":
        return CTDGModel(config, use_virtual_nodes=True)
    raise ValueError(
        f"Unknown model variant {name!r}; expected constant, schnet, painn, "
        "egnn, egnn_tvn, ctdg or ctdg_tvn"
    )
