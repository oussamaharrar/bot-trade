# Training at Scale

- Prefer `SubprocVecEnv` when `n_envs > 1`.
- Increase replay buffer for large runs; watch RAM usage.
- For GPUs, enable mixed precision (TF32) where supported.

