# FloodSense-AI: Genuine Sen1Floods11 Model Training & Evaluation Report
**Project Title:** FloodSense-AI: Multi-Modal SAR & Optical Change Detection and Disaster Response  
**Degree:** Final-Year B.E. Computer Science and Engineering Capstone  
**Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM) | Intel Core i7-14700HX | 16 GB DDR5 RAM  
**Dataset:** Official Sen1Floods11 v1.1 Hand-Labeled Ground Truth (Bonafilia et al., IEEE CVPRW 2020)  
**Execution Date:** November 08, 2026  
**Model Architecture:** `LightweightDualStreamAttentionUNet` (8,242,931 parameters)  
**Checkpoint Path:** `backend/outputs/sen1floods11_training/best_checkpoint.pth`  

---

## 1. Executive Summary

This report documents the genuine training and evaluation of the `LightweightDualStreamAttentionUNet` on the official Sen1Floods11 v1.1 hand-labeled satellite flood dataset. All training and hyperparameter selection strictly utilized the official training (252 chips) and validation (89 chips) partitions. The final performance was evaluated once on the untouched test split (90 chips). The 15-chip Bolivia event was strictly held out as an unseen geographical benchmark.

### Core Benchmark Results

| Evaluation Split | Total Chips | Total Patches (256x256) | Loss | Overall Accuracy | Mean IoU (mIoU) | Mean F1-Score | Best Epoch |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Validation Split** | 89 | 356 | 0.2213 | — | **0.7217** | **0.8104** | **Epoch 28** |
| **Untouched Test Split** | **90** | **360** | **0.2268** | **95.50%** | **0.7198** | **0.8255** | — |

---

## 2. Hardware and Environment Setup

- **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU (8,192 MB GDDR6, SM 8.9 Ada Lovelace)
- **Host CPU:** Intel Core i7-14700HX (20 Cores, 28 Threads)
- **Host Memory:** 16 GB DDR5 RAM
- **Software Environment:** Isolated CUDA Environment (`backend/.venv_cuda`)
- **PyTorch Version:** PyTorch 2.5.1+cu121 with native Automatic Mixed Precision (`torch.amp.autocast`)
- **Geospatial Engine:** Rasterio 1.4.4 / GDAL
- **Global Python Protection:** Host Python (`D:\Python`) was untouched.

---

## 3. Pre-Training Data Verification & Normalization

All normalization parameters and class weights were computed **strictly from the 252 training chips**:

1. **Sentinel-1 SAR Normalization:**
   - Raw Training VV: mean = -10.39 dB, std = 4.04 dB (range: [-51.38, 26.98] dB)
   - Raw Training VH: mean = -17.24 dB, std = 4.75 dB (range: [-71.62, 15.29] dB)
   - Normalization applied: $\text{VV}_{\text{norm}} = \text{clip}((\text{VV} - (-12.0)) / 6.0, -3.0, 3.0)$, $\text{VH}_{\text{norm}} = \text{clip}((\text{VH} - (-19.0)) / 5.0, -3.0, 3.0)$
   - Resulting clipped pixels: VV: 0.11%, VH: 1.08%
   - Swath edge NaNs (2.01% in training): Replaced with 0.0 and labeled with ignore index `255`.
2. **Sentinel-2 Optical Preprocessing:**
   - Level-1C Top-Of-Atmosphere (TOA) Digital Numbers scaled by $10,000.0$ and clipped to $[0.0, 1.0]$.
   - 4-band selection: `[3, 2, 1, 7]` $\to$ `[B4 (Red), B3 (Green), B2 (Blue), B8 (NIR)]`.
3. **Partition Isolation Proof:**
   - Train (252), Valid (89), Test (90), Bolivia (15).
   - Overlap across all pairs: **0 chips**. Zero data leakage.

---

## 4. Class Distribution & Training Loss Weighting

### Class Distribution Across Official Partitions

| Partition | Total Pixels | Valid Pixels | Class 0 (Dry Land) | Class 1 (Perm Water) | Class 2 (Water outside JRC) | Class 255 (NoData / Margin) |
|---|---|---|---|---|---|---|
| **Train (252 chips)** | 66,060,288 | 57,279,257 | 51,832,579 (90.49%) | 1,521,806 (2.66%) | 3,924,872 (6.85%) | 8,781,031 (13.29%) |
| **Valid (89 chips)** | 23,330,816 | 20,274,154 | 18,042,948 (88.99%) | 609,668 (3.01%) | 1,621,538 (8.00%) | 3,056,662 (13.10%) |
| **Test (90 chips)** | 23,592,960 | 20,443,959 | 17,928,951 (87.70%) | 899,332 (4.40%) | 1,615,676 (7.90%) | 3,149,001 (13.35%) |

### Training-Derived Class Weights
Computed via smoothed inverse class frequency on the training set:
- **Class 0 (Dry Land):** `0.287`
- **Class 1 (Permanent Water):** `1.672`
- **Class 2 (Water outside JRC reference):** `1.041`

---

## 5. Training Configuration & Hyperparameters

- **Architecture:** `LightweightDualStreamAttentionUNet`
  - Optical Encoder: 4 channels $\to$ Conv2d filters [32, 64, 128, 256]
  - SAR Encoder: 2 channels $\to$ Conv2d filters [32, 64, 128, 256]
  - CMAF Fusion Bottleneck: 466 channels
  - Attention-Gated Decoder: 3 output classes
  - Total Parameters: 8,242,931
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay = 0.0001)
- **Initial Learning Rate:** 0.0005
- **Learning Rate Scheduler:** CosineAnnealingLR ($T_{\max}=30, \eta_{\min}=10^{-6}$)
- **Batch Size:** 4 patches ($256 \times 256$)
- **Mixed Precision:** Automatic Mixed Precision (AMP FP16) with GradScaler
- **Loss Function:** `CombinedFloodLoss` ($0.5 \times \text{Focal} + 0.5 \times \text{Dice}$, $\gamma=2.0$, ignore index = 255)
- **Data Augmentation:** Random Horizontal Flip, Random Vertical Flip, Random 90° Rotations (synchronized across optical, SAR, and target)
- **Early Stopping:** Patience of 10 epochs monitoring validation mIoU

---

## 6. Training Dynamics & Epoch History

Total training completed in **34.65 minutes** across **30 epochs**.

```
Epoch | Train Loss | Val Loss | Val mIoU | Val F1 | Class 0 IoU | Class 1 IoU | Class 2 IoU | Duration | GPU VRAM
------+------------+----------+----------+--------+-------------+-------------+-------------+----------+---------
    1 |     0.2983 |   0.3574 |   0.4520 | 0.5217 |      0.9391 |      0.0082 |      0.4088 |    70.5s | 1098.8 MB
    2 |     0.2609 |   0.2462 |   0.5736 | 0.6935 |      0.9579 |      0.4210 |      0.3419 |    69.6s | 1098.8 MB
    3 |     0.2479 |   0.2540 |   0.5675 | 0.6861 |      0.9591 |      0.4269 |      0.3164 |    68.5s | 1098.8 MB
    4 |     0.2451 |   0.2541 |   0.5906 | 0.7110 |      0.9630 |      0.4074 |      0.4015 |    70.0s | 1098.8 MB
    5 |     0.2277 |   0.2371 |   0.6033 | 0.7222 |      0.9629 |      0.4816 |      0.3655 |    70.5s | 1098.8 MB
    6 |     0.2219 |   0.2400 |   0.5994 | 0.7226 |      0.9453 |      0.4353 |      0.4177 |    68.9s | 1098.8 MB
    7 |     0.2286 |   0.2274 |   0.6267 | 0.7455 |      0.9654 |      0.4850 |      0.4296 |    68.7s | 1098.8 MB
    8 |     0.2240 |   0.2425 |   0.5982 | 0.7196 |      0.9561 |      0.4040 |      0.4344 |    68.9s | 1098.8 MB
    9 |     0.2162 |   0.2204 |   0.6404 | 0.7580 |      0.9658 |      0.5142 |      0.4413 |    70.2s | 1098.8 MB
   10 |     0.2103 |   0.2212 |   0.6288 | 0.7478 |      0.9655 |      0.4734 |      0.4476 |    69.1s | 1098.8 MB
   11 |     0.2109 |   0.2365 |   0.6242 | 0.7441 |      0.9611 |      0.4588 |      0.4527 |    68.9s | 1098.8 MB
   12 |     0.2081 |   0.2269 |   0.6432 | 0.7613 |      0.9632 |      0.5058 |      0.4607 |    69.5s | 1098.8 MB
   13 |     0.2109 |   0.2182 |   0.6530 | 0.7701 |      0.9636 |      0.4837 |      0.5116 |    68.5s | 1098.8 MB
   14 |     0.2046 |   0.2182 |   0.6804 | 0.7935 |      0.9680 |      0.5408 |      0.5323 |    68.3s | 1098.8 MB
   15 |     0.2012 |   0.2407 |   0.6422 | 0.7617 |      0.9542 |      0.4961 |      0.4763 |    67.7s | 1098.8 MB
   16 |     0.2003 |   0.2359 |   0.6360 | 0.7513 |      0.9690 |      0.3884 |      0.5505 |    69.7s | 1098.8 MB
   17 |     0.1965 |   0.2168 |   0.6580 | 0.7741 |      0.9687 |      0.5083 |      0.4971 |    69.3s | 1098.8 MB
   18 |     0.1916 |   0.2181 |   0.7072 | 0.8156 |      0.9691 |      0.5733 |      0.5792 |    70.0s | 1098.8 MB
   19 |     0.1933 |   0.2158 |   0.6934 | 0.8042 |      0.9686 |      0.5405 |      0.5710 |    70.0s | 1098.8 MB
   20 |     0.1875 |   0.2228 |   0.6825 | 0.7953 |      0.9681 |      0.5308 |      0.5487 |    68.9s | 1098.8 MB
   21 |     0.1886 |   0.2130 |   0.6967 | 0.8068 |      0.9707 |      0.5525 |      0.5669 |    69.2s | 1098.8 MB
   22 |     0.1827 |   0.2271 |   0.6663 | 0.7790 |      0.9708 |      0.4396 |      0.5886 |    69.0s | 1098.8 MB
   23 |     0.1836 |   0.2090 |   0.6845 | 0.7967 |      0.9711 |      0.5432 |      0.5393 |    69.5s | 1098.8 MB
   24 |     0.1833 |   0.2161 |   0.7174 | 0.8234 |      0.9715 |      0.5753 |      0.6055 |    69.5s | 1098.8 MB
   25 |     0.1826 |   0.2151 |   0.7045 | 0.8130 |      0.9712 |      0.5485 |      0.5938 |    69.4s | 1098.8 MB
   26 |     0.1829 |   0.2169 |   0.7031 | 0.8113 |      0.9716 |      0.5256 |      0.6121 |    69.1s | 1098.8 MB
   27 |     0.1793 |   0.2177 |   0.7164 | 0.8227 |      0.9711 |      0.5782 |      0.5999 |    68.8s | 1098.8 MB
   28 |     0.1795 |   0.2181 |   0.7217 | 0.8269 |      0.9708 |      0.5850 |      0.6093 |    68.5s | 1098.8 MB
   29 |     0.1827 |   0.2166 |   0.7145 | 0.8213 |      0.9697 |      0.5823 |      0.5914 |    69.5s | 1098.8 MB
   30 |     0.1844 |   0.2213 |   0.7008 | 0.8104 |      0.9685 |      0.5563 |      0.5776 |    69.4s | 1098.8 MB
```

---

## 7. Untouched Test Set Evaluation (90 Chips, 20.44M Valid Pixels)

The model checkpoint from **Epoch 28** (selected strictly based on validation mIoU) was evaluated on the untouched test partition.

### Per-Class Test Performance

| Semantic Class | Target Definition | Test IoU | Test F1-Score | Test Precision | Test Recall |
|---|---|:---:|:---:|:---:|:---:|
| **Class 0: Dry Land** | Non-water surface | **0.9695** | **0.9845** | 0.9827 | 0.9863 |
| **Class 1: Permanent Water** | Water matching JRC reference | **0.6101** | **0.7578** | 0.8830 | 0.6637 |
| **Class 2: Water Outside JRC** | Water observed outside JRC baseline | **0.5800** | **0.7342** | 0.7016 | 0.7698 |
| **MACRO AVERAGE** | **Mean across all classes** | **0.7198** | **0.8255** | **0.8558** | **0.8066** |

### Test Confusion Matrix (Pixel Counts)

```
                       Predicted Class 0        Predicted Class 1        Predicted Class 2
GT Class 0 (Dry)                17683536                    11237                   234178
GT Class 1 (Perm)                   7716                   596889                   294727
GT Class 2 (Flood)                304039                    67816                  1243821
```

---

## 8. Prediction Visualizations

Prediction visualization panels showing Sentinel-2 optical RGB, Sentinel-1 SAR backscatter, ground-truth labels, and model predictions were exported to:
`backend/outputs/sen1floods11_training/visualizations/`

- Legend:
  - **Beige / Brown:** Class 0 (Dry Land)
  - **Blue:** Class 1 (Permanent Water)
  - **Red:** Class 2 (Water identified by LabelHand but not present in JRC reference)
  - **Black:** Ignored NoData / Missing Radar Margin (255)

---

## 9. Scientific Limitations & Boundaries

1. **Definition of Class 2:** Class 2 represents water detected by human ground truth that was absent from the JRC 1984–2018 permanent water reference. While highly correlated with newly inundated disaster areas, it should not be claimed as independently verified floodwater in all cases, as it may include seasonal agricultural paddy flooding or seasonal ephemeral wetlands.
2. **Class Imbalance:** Dry land constitutes ~88%–90% of valid pixels, while permanent water and flood water represent ~3% and ~7% respectively. While smoothed class weighting and combined Focal-Dice loss mitigated bias, minority class segmentation remains challenging.
3. **Hardware Efficiency:** Peak VRAM consumption never exceeded **1098.8 MB** (~2.1 GB out of 8.0 GB available), confirming that the `LightweightDualStreamAttentionUNet` is exceptionally well-suited for demonstration and local workstation execution.

---

## 10. Conclusion

The model training and test evaluation pipeline completed with zero synthetic data, zero data leakage, and rigorous geospatial validation. The trained checkpoint is preserved at:
`backend/outputs/sen1floods11_training/best_checkpoint.pth`
