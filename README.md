 # Speech-Driven Gesture Generation for Virtual Characters

Code accompanying the MSc dissertation *Speech-Driven Gesture Generation for Virtual Characters Using Transformer and Diffusion-Based Models* (EP4DIS, MSc Artificial Intelligence and Data Science, Aston University, 2026).

The system generates 3D co-speech gesture from speech audio and aligned transcriptions using a Transformer speech encoder and a denoising diffusion probabilistic model (DDPM) decoder.

## Versions in this repository

| File | Contents |
|---|---|
| `gesture_generation_v2_pipeline.py` | **Improved model (v2), used for the final results.** Time-aligned word features, training, evaluation on the official BEAT2 test split, and rendering of the SMPL-X clips |
| `gesture_generation_pipeline.ipynb` | Baseline model (v1), described below |
| `requirements.txt` | Python libraries used |

### Improved model (v2)

- Acoustic features resampled to 30 fps; word-level BERT vectors aligned to motion frames using TextGrid timings
- Sinusoidal positional encoding; four Transformer decoder layers with temporal self-attention and cross-attention to speech
- Cosine noise schedule (200 steps), clean-motion prediction, MSE plus velocity loss
- Official BEAT2 split for the four speakers: 396 training, 24 validation, 52 test sequences
- AdamW (lr 1 × 10⁻⁴, weight decay 0.01), batch size 32, 200 epochs, best checkpoint at epoch 181, seed 42

The sections below describe the baseline model (v1).

## Contents

All stages are contained in a single Colab notebook, `gesture_generation_pipeline.ipynb`, organised into the following sections:

| Section | Purpose |
|---|---|
| Preprocessing | Extracts acoustic, linguistic and motion features from BEAT2 and caches them as `.npz` files |
| Model definition | Transformer speech encoder and cross-attention-conditioned diffusion decoder |
| Training | 30-epoch training loop with checkpointing |
| Evaluation | Fréchet Gesture Distance (FGD) and BeatAlign computation |
| Rendering | Visualisation of generated motion sequences |

## Data

The model is trained on the English release of the [BEAT2 corpus](https://huggingface.co/datasets/H-Liu1997/BEAT2) (`beat_english_v2.0.0`), credited to Liu et al. (2024) and released under the Apache License 2.0. The corpus is not redistributed in this repository and must be obtained separately using `git-lfs`.

Training uses a four-speaker subset (speakers 2, 4, 6 and 7), drawing on the `wave16k`, `textgrid` and `smplxflame_30` components. After preprocessing this yields 472 aligned audio–text–motion sequences.

## Feature representation

| Modality | Representation | Dimensions |
|---|---|---|
| Acoustic | 13 MFCCs + F0 contour (pyin) | 14 per frame |
| Linguistic | `bert-base-uncased` contextual embeddings | 768 per token |
| Motion | SMPL-X body pose parameters | 165 per frame |

Sequences are truncated or zero-padded to a fixed length of 200 frames.

## Model

- **Speech encoder** — four-layer Transformer, 256-dimensional shared projection space; acoustic and linguistic features are projected into this space and encoded jointly.
- **Motion decoder** — DDPM diffusion decoder conditioned on the per-frame encoder output via cross-attention.

## Training configuration

| Setting | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 1 × 10⁻⁴ |
| Batch size | 8 |
| Epochs | 30 |
| Checkpoint interval | Every 5 epochs |
| Hardware | Single NVIDIA T4 GPU (Google Colab) |

## Running the pipeline

1. Open `gesture_generation_pipeline.ipynb` in Google Colab (use the *Open in Colab* badge at the top of the file).
2. Set the runtime to GPU: **Runtime → Change runtime type → T4 GPU**.
3. Download the BEAT2 English release and place it in your Drive.
4. Update the `path` variable in the second cell to point at your own Drive location.
5. Run the preprocessing section to generate the feature cache.
6. Run the training section, then the evaluation section.

Feature extraction is the slowest stage; the cache is written once and reused across training runs.

## Evaluation

Two quantitative metrics are reported:

- **Fréchet Gesture Distance (FGD)** — computed from pose and velocity statistics rather than a learned motion-autoencoder embedding. Values are therefore not directly comparable with FGD figures obtained using learned feature representations.
- **BeatAlign** — motion beats are detected from peaks in frame-to-frame velocity; audio beats are approximated from the cached acoustic features rather than detected from the raw waveform.

No human perceptual evaluation is reported. The methodological implications of both simplifications are discussed in Chapters 4 and 6 of the dissertation.

## Reproducibility

- **v2 (final model):** the seed is fixed at 42 for training, and evaluation uses a fixed seed for each test sequence (42 plus its index), so the reported v2 results can be regenerated.
- **v1 (baseline):** random seeds were not fixed. Because the diffusion decoder samples stochastically, repeated v1 generation from the same speech input produces different motion sequences, and metric values may vary slightly between runs.

## Dependencies

See `requirements.txt` for the full list. The code was run on the Google Colab runtime (August–September 2026) using its pre-installed library versions; only PyRender is pinned (0.1.45).
## Citation

Liu, H., Zhu, Z., Becherini, G., Peng, Y., Su, M., Zhou, Y., Zhe, X., Iwamoto, N., Zheng, B. and Black, M.J. (2024) 'EMAGE: towards unified holistic co-speech gesture generation via expressive masked audio gesture modeling', *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 1144–1154.
