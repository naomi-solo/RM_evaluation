# Latent Adversarial Attacks on Reward Models

This repository studies **latent-space adversarial attacks on reward models (RMs)** and analyzes the directions in representation space that cause reward preferences to flip.

The pipeline:

1. Apply **PGD attacks in the residual stream** of a reward model
2. Extract resulting **perturbation directions** in hidden space
3. Run **PCA** to identify dominant attack directions
4. Use LLM labeling to interpret these directions

The goal is to understand **reward model robustness** and whether adversarial perturbations correspond to **interpretable semantic axes**.

---

# Repo Structure

```
src/        core attack, data loading, and analysis code
runners/    experiment scripts and sweeps
analysis/   notebooks and exploratory analysis
results/    saved attack directions, PCA outputs, labels
legacy/     earlier experiment scripts
```

Key files:

```
src/attacks/layer_attack_direction.py
src/data/load_pairs.py
src/analysis/extract_directions.py
src/analysis/pca_directions.py
src/analysis/label_components.py
```

---

# Running Experiments

Core sweep:

```bash
python runners/run_core_sweep.py
```

Epsilon sweep:

```bash
python runners/run_eps_sweep.py
```

Label PCA components:

```bash
python runners/run_label_all_pca.py
```

---

# Datasets

Experiments use several preference-style datasets:

* BBQ
* GSM-MC / MATH-MC
* MMLU
* SGXS

The MC datasets are expected in:

```
~/lat/MC-Evaluation
```

---

# Outputs

Results are stored in:

```
results/
    directions/
    pca/
    labels/
```

