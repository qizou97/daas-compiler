# daas-compiler

Compile SpatialData zarr samples into a cell-centered HE patch WebDataset for
ML training. Three phases: extract per-sample → compile global manifest →
train via `CellPatchDataset`.

The repo is shipped as a Claude Code skill plugin: the AI agent reads
`skills/daas-compiler/SKILL.md` and runs the bundled scripts on the user's
behalf. The same scripts can be used directly without the agent.

## Install (via Claude Code)

```bash
/plugin marketplace add qizou97/daas-compiler
/plugin install daas-compiler@daas-compiler
```

The first time the skill activates in a project, the agent will run:

```bash
pip install -e "${SKILL_DIR}"
pip install -r "${SKILL_DIR}/requirements.txt"
```

where `${SKILL_DIR}` is the installed plugin's
`skills/daas-compiler/` directory.

## Manual install

```bash
git clone https://github.com/qizou97/daas-compiler.git
cd daas-compiler/skills/daas-compiler
pip install -e .
pip install -r requirements.txt
```

## Quick start

```bash
# 1. Extract patches from one sample
python3 scripts/extract_sample.py --zarr /data/sample.zarr --output /data/out/sample

# 2. Compile multiple per-sample dirs into a global dataset
python3 scripts/compile_dataset.py --per-sample-dir /data/out --output /data/compiled

# 3. Train against the compiled dataset
python3 -c "
from daas.dataset import CellPatchDataset
ds = CellPatchDataset('/data/compiled/manifest.parquet',
                      '/data/compiled/expression.h5ad')
print(len(ds), ds[0]['image'].size)
"
```

Full reference: [`skills/daas-compiler/SKILL.md`](skills/daas-compiler/SKILL.md).
User-facing prompts: [`skills/daas-compiler/usage-guide.md`](skills/daas-compiler/usage-guide.md).

## Tests

```bash
cd skills/daas-compiler
pip install pytest
pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).
