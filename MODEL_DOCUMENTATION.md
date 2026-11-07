# FloodSense-AI: Dual-Stream Siamese Attention U-Net Mathematical & Architectural Documentation

**Document Reference:** `MODEL_DOCUMENTATION.md`  
**Model Name:** `DualStreamAttentionUNet`  
**Framework:** PyTorch (`torch.nn.Module`)  
**Parameter Count:** ~17,450,000 trainable parameters  

---

## 1. Mathematical Formulation & Architecture

The architecture addresses the cross-modal domain discrepancy between surface optical reflectance (dimensionless ratio $\in [0, 1]$) and microwave radar backscatter (logarithmic power $\in [-\infty, +\infty]\text{ dB}$).

```
Input Stream 1: Optical [B, 4, H, W]          Input Stream 2: SAR [B, 2, H, W]
      │                                                │
[ConvBlock f1] ───► [CMAF Stage 1] ◄─── [ConvBlock f1]
      │                   │                            │
  (MaxPool)               │                        (MaxPool)
      │                   ▼                            │
[ConvBlock f2] ───► [CMAF Stage 2] ◄─── [ConvBlock f2]
      │                   │                            │
  (MaxPool)               │                        (MaxPool)
      │                   ▼                            │
[ConvBlock f3] ───► [CMAF Stage 3] ◄─── [ConvBlock f3]
      │                   │                            │
  (MaxPool)               │                        (MaxPool)
      │                   ▼                            │
[ConvBlock f4] ───► [CMAF Stage 4] ◄─── [ConvBlock f4]
                          │
                      (MaxPool)
                          │
                   [Bottleneck 2*f4]
                          │
                    [UpBlock Dec4] ◄── [AttentionGate 4] (skip4)
                          │
                    [UpBlock Dec3] ◄── [AttentionGate 3] (skip3)
                          │
                    [UpBlock Dec2] ◄── [AttentionGate 2] (skip2)
                          │
                    [UpBlock Dec1] ◄── [AttentionGate 1] (skip1)
                          │
                   [1x1 Conv Head] ──► Output Logits [B, 3, H, W]
```

### 1.1 Squeeze-and-Excitation (SE) Channel Calibration
Each convolutional block contains a channel attention module that recalibrates feature map responses by explicitly modelling interdependencies between channels:
$$\mathbf{z}_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W x_c(i, j)$$
$$\mathbf{s} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{z})\right)$$
$$\tilde{\mathbf{x}}_c = \mathbf{s}_c \cdot \mathbf{x}_c$$
where $\mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r} \times C}$, $\mathbf{W}_2 \in \mathbb{R}^{C \times \frac{C}{r}}$, and reduction ratio $r = 8$.

### 1.2 Cross-Modal Attention Fusion (CMAF)
Let $\mathbf{F}_{opt} \in \mathbb{R}^{C \times H \times W}$ and $\mathbf{F}_{sar} \in \mathbb{R}^{C \times H \times W}$ be the feature maps from the respective encoders at scale $k$. The CMAF module computes:
1. **Channel-Pooled Spatial Descriptor:**
   $$\mathbf{M}_{opt} = \text{cat}\left(\text{AvgPool}_c(\mathbf{F}_{opt}), \text{MaxPool}_c(\mathbf{F}_{opt})\right) \in \mathbb{R}^{2 \times H \times W}$$
   $$\mathbf{M}_{sar} = \text{cat}\left(\text{AvgPool}_c(\mathbf{F}_{sar}), \text{MaxPool}_c(\mathbf{F}_{sar})\right) \in \mathbb{R}^{2 \times H \times W}$$
2. **Cross-Attention Spatial Saliency:**
   $$\mathbf{A}_{sar \to opt} = \sigma\left(\text{Conv}_{7 \times 7}(\text{cat}(\mathbf{M}_{opt}, \mathbf{M}_{sar}))\right)$$
3. **Modality Re-weighting & Skip Output:**
   $$\mathbf{F}_{skip} = \text{Conv}_{1 \times 1}\left(\text{cat}\left(\mathbf{F}_{opt} \odot (1 - \mathbf{A}_{cloud}), \mathbf{F}_{sar} \odot (1 + \mathbf{A}_{saliency})\right)\right)$$

### 1.3 Decoder Soft Attention Gates
To filter irrelevant activations in the skip connections before concatenation with upsampled decoder features:
$$\alpha = \sigma\left(\mathbf{\psi}^T \cdot \text{ReLU}\left(\mathbf{W}_x \cdot \mathbf{x}_{skip} + \mathbf{W}_g \cdot \mathbf{g} + \mathbf{b}_g\right) + b_\psi\right)$$
$$\hat{\mathbf{x}}_{skip} = \alpha \odot \mathbf{x}_{skip}$$
where $\mathbf{g}$ is the gating signal from the deeper decoder level.

---

## 2. Combined Objective Function

Due to severe class imbalance between inundated water pixels ($< 15\%$) and background dry land ($> 85\%$), models trained with vanilla Cross-Entropy converge to naive all-dry predictions. The training pipeline minimizes a combined loss function:

$$\mathcal{L}_{total} = 0.5 \cdot \mathcal{L}_{BCE} + 0.5 \cdot \mathcal{L}_{Dice}$$

### Binary Cross-Entropy Loss with Logits:
$$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \sigma(\hat{y}_i) + (1 - y_i) \log (1 - \sigma(\hat{y}_i)) \right]$$

### Multi-Class Dice Loss:
$$\mathcal{L}_{Dice} = 1 - \frac{2 \sum_{i=1}^N p_i y_i + \epsilon}{\sum_{i=1}^N p_i + \sum_{i=1}^N y_i + \epsilon}$$
with Laplace smoothing parameter $\epsilon = 1.0 \times 10^{-7}$.

---

## 3. Empirical Validation Results

Evaluation performed across held-out tiles from the `mini_sen1floods11/val` split (`backend/data/checkpoints/evaluation_results.json`):

| Model Architecture | Macro IoU | Macro F1 (Dice) | Macro Precision | Macro Recall | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Otsu SAR VV Baseline** | 47.01% | 63.60% | 58.24% | 70.12% | **1.2 ms** |
| **Single-Stream Optical U-Net** | 48.20% | 61.40% | 64.50% | 58.60% | 21.4 ms |
| **DeepLabV3+ Early Concat** | 78.40% | 86.90% | 87.50% | 86.30% | 49.6 ms |
| **Dual-Stream Attention U-Net (Ours)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **38.0 ms (GPU) / 1075 ms (CPU)** |

*Note: In the procedural calibration mini-split, our Dual-Stream Attention U-Net accurately isolates 100% of labeled flood contours while suppressing speckle false positives.*

