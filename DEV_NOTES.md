# Developer Notes

## Running tools directly vs module mode

Project helper scripts live under the `tools/` package. When executing these
scripts it is recommended to run them as modules from the project root, e.g.

```bash
python -m tools.export_charts --help
```

This ensures that the project root is on `sys.path` and avoids `ImportError`
issues. For convenience the scripts also include a small `sys.path` fix-up so
that direct execution (`python tools/export_charts.py`) still works if needed.
