"""Sleep scoring from a single cortical channel.

Sub-packages, in the order data flows through them:

- ``data``: label codes, raw recordings, the prepared-data cache and the GUI's file tools.
- ``models``: encoders, heads, losses and checkpoint helpers.
- ``training``: one module per training stage (contrastive, adversarial, heads).
- ``evaluation``: metrics, reports and diagnostic plots.
- ``inference``: ``SleepScorer``, which scores new recordings with a trained run.
- ``pipeline``: the TOML config and the stage runner (``python -m scorer.pipeline.run``).
- ``gui``: the PyQt5 application (``python -m scorer.gui``).

See ``README.md`` for usage.
"""
