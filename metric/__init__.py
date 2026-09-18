"""Pixel-wise evaluation metrics: mse, ssim, lpips, dino.

Each submodule exposes a ``compute(...)`` function; model-based metrics
(:mod:`metric.lpips`, :mod:`metric.dino`) additionally expose ``load(...)``
returning a small dataclass holding the loaded model. The orchestrator
(:mod:`modules.evaluator`) is responsible for loading models once and
passing them into ``compute``.
"""
