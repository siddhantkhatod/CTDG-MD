# Deployment runbook

1. Train and select a checkpoint from the real-data ablation.
2. Copy/retain the checkpoint on a read-only model volume.
3. Set `CTDG_CHECKPOINT` to its container path.
4. Start with `docker compose -f deployment/docker-compose.yml up --build`.
5. Require `/health` to return `status=ok`, `checkpoint_loaded=true`, and `unsafe_untrained=false` before routing traffic.
6. Use `/docs` to inspect the request schema and test a known frame window.

The service does not download training data and does not write user trajectories. It loads one checkpoint at startup. Replace the container to deploy a new checkpoint; do not mutate weights in a live process.

## Production notes

- Put TLS/authentication and request-size limits at the reverse proxy.
- CPU is supported but geometric graph construction is much faster on a suitable GPU.
- Requests are validated for dimensions, roles and monotonically increasing time.
- Current batching is one trajectory request per HTTP call.
- Log model checksum and application image digest externally for traceability.
- The API reports dataset-native interaction energy, not experimental affinity.
