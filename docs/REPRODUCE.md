# Reproducing the competition package

This guide rebuilds the training package that we submitted to Track 1 of the NeurIPS 2026 RealPDE Competition, Codabench submission 948641 with a final score of 82.01. The package is a Docker build context plus a ZIP holding the same files. Its own README, with the hardware, environment and randomness details the organizers asked for, is [docs/package/README.md](package/README.md). The method is described in [docs/METHOD.md](METHOD.md).

## Data

The organizers mount a directory holding `train_real/` and `train_sim/` at `/data`. `train_real/7575_0.h5` duplicates another trajectory and is left out. The data is published on Hugging Face as [AI4Science-WestlakeU/RealPDE-Competition-Data](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data).

## Assemble the package

Run from the repository root.

```bash
python tools/make_final_package.py
docker build -t trussnet dist/final/package
```

The first command assembles `dist/final/package/` and writes `dist/final/physicsoracle_track1.zip`, named after the team as in the original submission. The second builds the Docker image from the same folder, so the image and the ZIP hold the same code. Rerun the packaging command after every change to the sources, because an existing package is not updated automatically.

The package folder is copied into `/workspace` in the image.

```text
/workspace/
├── Dockerfile                        from docs/package/Dockerfile
├── README.md                         from docs/package/README.md
├── METHOD.md                         from docs/METHOD.md
├── solution.py
├── configs/final.yaml
├── figures/                          architecture and progress figures, light and dark
├── rpde/                             model, training and inference code
├── submission/submission.py
├── third_party/realpde_kit/          scoring.py and README.md
└── training_logs_948641/
    ├── fm_all_s3/                    output_log.txt, history.txt, config.yaml
    └── fm_all_s4/                    output_log.txt, history.txt, config.yaml
```

Training reads `/data` and writes the trained weights, compiled graphs and predictor runtime to `/output`. The development `tools/`, the saved `weights/` and the reproduction logs are not part of the package.

## Test the image

Running the image needs Docker with the NVIDIA Container Toolkit. A smoke run checks the whole pipeline in about 5 minutes. The second command is the organizers' training command, unchanged.

```bash
docker run --rm --network none --gpus '"device=0"' --shm-size 32g -e RPDE_SMOKE=1 \
  -v /path/to/train_data:/data:ro -v /tmp/trussnet_smoke:/output trussnet \
  python -c "import solution; solution.train('/data', '/output', 'cuda')"
grep -E "WARNING|compiled batch sizes|clean-room" /tmp/trussnet_smoke/logs/solution.log   # expect [1, 5], no WARNING

docker run --rm --network none --gpus '"device=0"' --shm-size 32g \
  -v /path/to/train_data:/data:ro -v /path/to/output:/output trussnet \
  python -c "import solution; solution.train('/data', '/output', 'cuda')"
```

The full run logs `compiled batch sizes: [1, 2, 3, 4, 5, 10]` and no `WARNING`. Use a fresh output directory for each run. On one RTX 4090 per member, training took 156.6 minutes.

## Train members locally

With the environment from `uv sync`, build the cache and train the two members.

```bash
python rpde/tools/make_cache.py --raw /path/to/train_data
python -m rpde.train --config configs/final.yaml --fold full --seed 3 --name fm_all_s3
python -m rpde.train --config configs/final.yaml --fold full --seed 4 --name fm_all_s4
```

The default cache is `data/cache32/`, and `RPDE_CACHE` selects another location. Training writes to `runs/<name>/`, including `ep59.pth`, `log.txt` and the resolved configuration. These commands train the members only. The full `solution.train()` pipeline also fits the gate tables, compiles the ensemble and assembles the predictor, so use the Docker image for the pinned compilation toolchain.

## Code entry points

| File | Responsibility |
|---|---|
| `solution.py` | `train(data_dir, output_dir, device)` and `load_predictor(output_dir, device)` |
| `rpde/train.py`, `rpde/data.py`, `rpde/losses.py` | training loop, data sampling and losses |
| `rpde/scaffold.py`, `rpde/cores/axial.py` | physics first guess and axial transformer |
| `rpde/ensemble_predict.py`, `rpde/gated_decode.py`, `rpde/decode.py` | energy-preserving ensemble and output decoding |
| `rpde/tools/aot_build_ens.py` | ahead-of-time compilation of the ensemble |
| `rpde/tools/pack_submission.py` | assembly of the inference artifacts after training |
| `tools/make_final_package.py` | assembly of the training package and ZIP |

`load_predictor()` loads the artifacts in the output directory. Keep the weights, `ens_cfg.json`, the runtime code and the compiled graphs together.

## Regenerate the figures and the banner

The figure scripts need Matplotlib, and the banner also needs SciPy. Neither is a training dependency.

```bash
pip install matplotlib scipy
python tools/make_architecture_figure.py                  # docs/figures/architecture.svg and architecture_dark.svg
python tools/make_improvement_figure.py --font-scale 1.15  # docs/figures/progress.svg and progress_dark.svg
python tools/banner_flow.py --output build/banner_flow.npz # lattice Boltzmann flow past a NACA 4418, a few minutes on CPU
python tools/make_banner.py --flow build/banner_flow.npz --snapshot 4 --output docs/assets/banner.png
```

The progress figure reads the score history from `tools/figure_data/submissions.csv`. Titles and the banner text use JetBrains Mono when it is installed or found in `TRUSSNET_FONT_DIR`, and fall back to DejaVu Sans Mono otherwise.
