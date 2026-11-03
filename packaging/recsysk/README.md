# recsysk has been renamed to rec-benchmark

This project now lives on PyPI as [`rec-benchmark`](https://pypi.org/project/rec-benchmark/) and on GitHub as [iamsikun/rec-bench](https://github.com/iamsikun/rec-bench). The import package and CLI are still `recsys`, so your code does not change.

This final `recsysk` release contains no code and only depends on `rec-benchmark`. Switch your requirements to:

```bash
pip install rec-benchmark
# or with uv
uv add rec-benchmark
```

If you already have `recsysk` 0.1.0 installed, uninstall it first so the two packages don't fight over the same files:

```bash
pip uninstall recsysk
pip install rec-benchmark
```

If you ran `pip install -U recsysk` and `import recsys` now fails, pip removed the shared files while uninstalling the old version. Reinstall to restore them:

```bash
pip install --force-reinstall --no-deps rec-benchmark
```
