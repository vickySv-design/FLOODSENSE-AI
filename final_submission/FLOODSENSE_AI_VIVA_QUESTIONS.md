# FloodSense-AI: Comprehensive Viva & Oral Examination Guide

**Degree:** Bachelor of Engineering (B.E.) in Computer Science and Engineering  
**Subject:** Final Year Project Defense  
**Project Title:** FloodSense-AI: Multi-Modal Flood Detection, Flood Dynamics, Infrastructure Exposure Mapping and Emergency Rescue Routing  
**Candidate Preparation Manual:** 56 Comprehensive Questions & Model Answers across 8 Technical Domains

---

## Table of Contents
1. [Domain 1: Project Motivation & Problem Statement (Q1–Q7)](#domain-1-project-motivation--problem-statement)
2. [Domain 2: Remote Sensing Physics: SAR vs Optical (Q8–Q15)](#domain-2-remote-sensing-physics-sar-vs-optical)
3. [Domain 3: Satellite Preprocessing & Feature Engineering (Q16–Q22)](#domain-3-satellite-preprocessing--feature-engineering)
4. [Domain 4: Deep Learning Architecture & Spatial Cross-Attention (Q23–Q30)](#domain-4-deep-learning-architecture--spatial-cross-attention)
5. [Domain 5: Training Strategy, Loss Functions & Sen1Floods11 Benchmarking (Q31–Q38)](#domain-5-training-strategy-loss-functions--sen1floods11-benchmarking)
6. [Domain 6: Scientific Model Provenance: Model A vs Model B (Q39–Q44)](#domain-6-scientific-model-provenance-model-a-vs-model-b)
7. [Domain 7: GIS Infrastructure Exposure & Geodesic Area Mathematics (Q45–Q50)](#domain-7-gis-infrastructure-exposure--geodesic-area-mathematics)
8. [Domain 8: Road Network Topology, Directed DiGraphs & A* Routing (Q51–Q56)](#domain-8-road-network-topology-directed-digraphs--a-routing)

---

## Domain 1: Project Motivation & Problem Statement

### Q1: What is the primary problem that FloodSense-AI solves?
**Answer:**  
Conventional disaster management relies on optical satellite imagery or post-disaster manual ground surveys. Optical sensors are blinded by heavy cloud decks and rainstorms during active flood events, leading to a 48–72 hour information blackout. Furthermore, existing academic models output unreferenced segmentation masks without connecting predictions to real-world infrastructure. FloodSense-AI solves this by combining cloud-penetrating Sentinel-1 SAR with Sentinel-2 optical data, generating real-time 3-class flood inundation rasters, intersecting them with OpenStreetMap infrastructure to identify exposed buildings and cut-off roads, and computing turn-by-turn A* rescue routes for emergency convoys.

### Q2: Why is FloodSense-AI termed a "multi-modal" system?
**Answer:**  
It simultaneously fuses two fundamentally distinct physical imaging modalities:
1. Active microwave Synthetic Aperture Radar (SAR) from Sentinel-1 (C-band, 5.405 GHz, VV and VH polarizations).
2. Passive multi-spectral optical surface reflectance from Sentinel-2 (Visible, NIR, SWIR bands).  
These two sensors provide complementary physical data: SAR penetrates clouds and provides surface roughness information, while optical provides precise multi-spectral biochemical water signatures (NDWI/MNDWI).

### Q3: What are the three semantic classes predicted by the deep learning model?
**Answer:**  
- **Class 0:** Dry Land (unaffected soil, vegetation, built-up surfaces).
- **Class 1:** Permanent Water (pre-existing baseline rivers, lakes, reservoirs, coastlines).
- **Class 2:** Mapped Floodwater (newly inundated terrestrial areas caused by the disaster).  
Distinguishing Class 1 from Class 2 is vital so civil defense responders are not dispatched to evacuate pre-existing natural lakes.

### Q4: How does FloodSense-AI achieve zero-cost execution?
**Answer:**  
The system relies entirely on open-access Copernicus Earth Observation satellite data (via ESA/Copernicus Open Access), open-source vector data from OpenStreetMap (Overpass API), open-source software libraries (PyTorch, Rasterio, GeoPandas, NetworkX, FastAPI, React), and executes 100% locally on standard workstation GPU hardware (NVIDIA RTX 4070 8GB VRAM) without cloud hosting fees or paid APIs.

### Q5: What makes FloodSense-AI operationally ready compared to typical academic research papers?
**Answer:**  
Academic literature typically stops at generating a floating-point prediction tensor or IoU score on a sanitized benchmark. FloodSense-AI is a closed-loop system: it converts deep learning rasters into geospatial vector layers, builds a directed road connectivity graph honoring traffic laws, computes evacuation routes with kinematic ETAs, and delivers results via an interactive web dashboard with Server-Sent Events (SSE).

### Q6: What geographic testbed was used for system demonstration?
**Answer:**  
The Adyar River basin in Chennai, Tamil Nadu, India (bounding box: 80.18° E to 80.28° E, 12.98° N to 13.06° N), an area historically prone to catastrophic urban flooding (e.g., the 2015 and 2023 South Indian floods).

### Q7: What are the system requirements to run FloodSense-AI?
**Answer:**  
A 64-bit Windows or Linux PC with a CUDA-compatible NVIDIA GPU (>= 6GB VRAM; RTX 4070 8GB used), Python 3.10+, and Node.js v18+. The entire pipeline executes locally in under 10 seconds per scene.

---

## Domain 2: Remote Sensing Physics: SAR vs Optical

### Q8: Explain the physical principle of SAR specular reflection on open water.
**Answer:**  
Active radar sends microwave pulses to the Earth's surface and measures backscattered power ($\sigma^0$). Because calm open water has low surface roughness relative to the radar wavelength (C-band $\lambda \approx 5.6$ cm), incident microwave energy reflects away from the satellite sensor like a mirror (specular reflection). Consequently, water returns virtually zero backscatter to the sensor, appearing distinctively dark (typically $-22$ dB to $-16$ dB) compared to surrounding rough dry terrain ($-12$ dB to $-6$ dB).

### Q9: What are the dual polarizations VV and VH in Sentinel-1, and how do they differ in flood detection?
**Answer:**  
- **VV (Vertical transmit, Vertical receive):** Highly sensitive to surface roughness, wind-induced capillary water ripples, and open water boundaries. Provides strong contrast between calm water and rough soil.
- **VH (Vertical transmit, Horizontal receive):** Cross-polarization is less sensitive to wind-roughened surface water waves and provides superior discrimination of urban built-up structures and vegetation volume scattering. Fusing VV and VH suppresses false-positive water classifications.

### Q10: What is radar speckle and how is it handled?
**Answer:**  
Speckle is an inherent granular noise caused by constructive and destructive interference of coherent microwave echoes within a single resolution cell. FloodSense-AI suppresses speckle through radiometric decibel transformation, dynamic range clipping to $[-30.0\text{ dB}, 0.0\text{ dB}]$, and spatial convolutions within the deep learning encoder that learn multi-scale spatial contextual smoothing.

### Q11: What are radar layover and radar shadow in urban environments?
**Answer:**  
In high-relief urban areas, tall buildings cause geometric distortions:
- **Layover:** The top of a tall building is closer to the radar antenna than its base, causing the top to be imaged earlier in slant range, overlaying ground features.
- **Radar Shadow:** The area behind a tall building receives no radar illumination, creating dark pixels that can mimic water backscatter.  
Our multi-modal cross-attention UNet mitigates this by fusing optical spectral context to verify whether dark radar pixels are genuine water or urban shadows.

### Q12: Why does pure optical remote sensing fail during active flood events?
**Answer:**  
Optical sensors operate in visible, NIR, and SWIR wavelengths (0.4–2.2 $\mu\text{m}$), which cannot penetrate cloud droplets and atmospheric aerosols. During heavy monsoon storms or cyclones, continuous cloud decks completely obscure ground surface reflectance, causing a 100% data void in optical imagery.

### Q13: What is the Normalized Difference Water Index (NDWI) and how is it calculated?
**Answer:**  
Formulated by McFeeters (1996):
$$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$$
Water features exhibit high reflectance in the Green band (Sentinel-2 Band 3, 560 nm) and very low reflectance / strong absorption in the Near-Infrared band (Sentinel-2 Band 8, 842 nm). Positive NDWI values typically indicate open water surfaces, while terrestrial vegetation and dry land yield negative values.

### Q14: What is the Modified NDWI (MNDWI) and why is it preferred in urban areas?
**Answer:**  
Formulated by Xu (2006):
$$\text{MNDWI} = \frac{\text{Green} - \text{SWIR}}{\text{Green} + \text{SWIR}}$$
In built-up urban regions, asphalt and concrete often reflect NIR similarly to water, causing false positives with standard NDWI. Shortwave-Infrared (SWIR, Sentinel-2 Band 11, 1610 nm) is strongly absorbed by water while being highly reflected by built-up land, drastically suppressing urban background noise.

### Q15: What is the spatial resolution (Ground Sample Distance) of Sentinel-1 and Sentinel-2?
**Answer:**  
- Sentinel-1 GRD: Nominal 10-meter pixel spacing in Interferometric Wide (IW) swath mode.
- Sentinel-2: 10 meters for visible (B2, B3, B4) and NIR (B8); 20 meters for red edge and SWIR bands (B11, B12). In our pipeline, all bands are resampled and co-registered onto a uniform 10-meter metric grid.

---

## Domain 3: Satellite Preprocessing & Feature Engineering

### Q16: How are raw Sentinel-1 Digital Numbers (DN) converted to backscatter in decibels?
**Answer:**  
Using the standard ESA radiometric calibration equation:
$$\sigma^0_{\text{dB}} = 10 \cdot \log_{10}(\text{DN}) - 83.0\text{ dB}$$
where $-83.0\text{ dB}$ is the sensor-specific calibration offset constant for Sentinel-1 GRD products.

### Q17: Why do we normalize SAR backscatter to [0.0, 1.0]?
**Answer:**  
Unclamped backscatter spans from $-35\text{ dB}$ to $+10\text{ dB}$, which can cause vanishing or exploding gradients in deep neural networks. We clip backscatter to $[-30.0\text{ dB}, 0.0\text{ dB}]$ and normalize linearly:
$$\text{VV}_{\text{norm}} = \frac{\sigma^0_{\text{dB,VV}} + 30.0}{30.0}$$
This maps typical open water (around $-20\text{ dB}$) to $\approx 0.33$, and dry urban/land (around $-8\text{ dB}$) to $\approx 0.73$.

### Q18: How is Sentinel-2 Level-2A surface reflectance scaled?
**Answer:**  
Sentinel-2 Level-2A Bottom-Of-Atmosphere (BOA) products are stored as 16-bit integers with a quantification factor of 10,000. Dividing raw integer values by 10,000.0 scales them into true physical reflectance values in the range $[0.0, 1.0]$.

### Q19: What coordinate reference system (CRS) is used across the pipeline?
**Answer:**  
Input satellite tiles and OSM vectors are aligned and projected onto WGS 84 (EPSG:4326) for standardized geographic web serving, or dynamically transformed to Universal Transverse Mercator (UTM Zone 44N, EPSG:32644) for accurate metric distance calculations.

### Q20: How are multi-modal scenes co-registered when they have slightly different bounds?
**Answer:**  
The preprocessing module calculates the overlapping geographic bounding box intersection, establishes a master 10-meter affine transform, and uses Rasterio bilinear/nearest-neighbor resampling to project all bands into identical $(H, W)$ tensor grids.

### Q21: What is temporal differencing in flood dynamics analysis?
**Answer:**  
It computes the change in radar backscatter between a pre-disaster baseline scene ($t_{\text{pre}}$) and a post-disaster event scene ($t_{\text{post}}$):
$$\Delta\sigma^0_{\text{VV}} = \sigma^0_{\text{VV}}(t_{\text{post}}) - \sigma^0_{\text{VV}}(t_{\text{pre}})$$
A significant negative drop ($\Delta\sigma^0_{\text{VV}} < -5.0\text{ dB}$) indicates that dry land has transitioned into specular open water.

### Q22: What happens if identical satellite rasters are passed to the temporal differencing module?
**Answer:**  
The module computes $\Delta\sigma^0_{\text{VV}} = 0.0\text{ dB}$ for all pixels, resulting in 0 detected change pixels. The system detects this condition and logs an informational notice explaining that pre- and post-flood rasters are identical (as in the Chennai demonstration preset), rather than manufacturing artificial change data.

---

## Domain 4: Deep Learning Architecture & Spatial Cross-Attention

### Q23: Why did you choose a Dual-Stream architecture over early fusion (channel concatenation)?
**Answer:**  
Early fusion simply stacks SAR (VV, VH) and optical (RGB, NIR) into a single 6-channel input tensor. This forces the very first convolutional layer to mix active microwave physics with passive optical reflectance, often causing the network to over-rely on optical bands and fail completely when clouds are present. A dual-stream architecture maintains independent, modality-specific convolutional encoders that learn pure SAR representations and pure optical representations before fusing them through cross-attention.

### Q24: What are the mathematical operations in the Spatial Cross-Attention block?
**Answer:**  
Let $F_{\text{SAR}} \in \mathbb{R}^{C \times H \times W}$ and $F_{\text{OPT}} \in \mathbb{R}^{C \times H \times W}$ be the feature maps from the respective encoders at a given scale.
1. The attention gate calculates spatial attention weights:
   $$A_{\text{SAR}} = \sigma(W_{\text{SAR}} * F_{\text{SAR}} + W_{\text{OPT}} * F_{\text{OPT}} + b)$$
   where $*$ is convolution, and $\sigma$ is the sigmoid activation function.
2. The attended features are computed:
   $$F_{\text{attended, SAR}} = A_{\text{SAR}} \odot F_{\text{SAR}}$$
   $$F_{\text{attended, OPT}} = (1 - A_{\text{SAR}}) \odot F_{\text{OPT}}$$
3. The features are concatenated and projected:
   $$F_{\text{fused}} = W_{\text{proj}} * [F_{\text{attended, SAR}} \parallel F_{\text{attended, OPT}}]$$
This allows the network to dynamically attenuate optical features if clouds obscure the region, allowing SAR features to dominate.

### Q25: How do skip connections in the UNet help preserve spatial detail?
**Answer:**  
Downsampling in the encoder reduces spatial resolution to capture high-level semantic context, but loses fine boundary details (such as narrow 10m road corridors and canal banks). Skip connections pass high-resolution feature maps directly from early encoder layers across to corresponding decoder layers, enabling the model to reconstruct sharp 10m spatial boundaries.

### Q26: What is the parameter count of the Sen1Floods11 benchmark model (Model B)?
**Answer:**  
Exactly 8,241,731 parameters (8.24M), with a checkpoint size of 31.5 MB (`best_checkpoint.pth`).

### Q27: What is the parameter count of the procedural baseline model (Model A)?
**Answer:**  
Exactly 40,610,691 parameters (40.6M), with a checkpoint size of 155 MB (`dual_stream_attention_unet.pth`).

### Q28: What is the receptive field of the network, and why does it matter?
**Answer:**  
Through 4 stages of $2\times2$ max-pooling, the receptive field expands to $> 150 \times 150$ pixels. This allows the network to recognize broad hydrological flood basins and river networks, rather than classifying pixels purely on local speckle noise.

### Q29: How does the network handle sliding window inference on large satellite scenes?
**Answer:**  
Full scenes (e.g., $1304 \times 1015$ pixels) exceed GPU memory if processed in a single batch. The inference engine divides the raster into $512 \times 512$ sliding window chips with a 64-pixel overlap, performs batched GPU inference, and blends overlapping edge probabilities using a Hann window to eliminate tile boundary seams.

### Q30: What activation function is applied at the final classification layer?
**Answer:**  
A Softmax activation function across the channel dimension (3 classes: Class 0, Class 1, Class 2), generating a probability distribution for each pixel where $\sum_{c=0}^2 P(c) = 1.0$.

---

## Domain 5: Training Strategy, Loss Functions & Sen1Floods11 Benchmarking

### Q31: What is the Sen1Floods11 dataset and why is it considered the gold standard?
**Answer:**  
Published by Cloud to Street (Bonafilia et al., 2020 at CVPR Workshops), Sen1Floods11 is a curated, globally representative dataset covering 11 flood events across 6 continents. It contains 446 hand-labeled 512x512 chips meticulously annotated by geospatial domain experts, providing high-quality ground truth labels that avoid the noise of automated thresholding masks.

### Q32: What is the hybrid loss function used during training?
**Answer:**  
A combination of Multi-Class Cross-Entropy ($\mathcal{L}_{\text{CE}}$) and Soft Dice Loss ($\mathcal{L}_{\text{Dice}}$):
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{Dice}} \quad (\text{with } \lambda = 1.0)$$
- $\mathcal{L}_{\text{CE}}$ optimizes probabilistic classification accuracy across all pixels.
- $\mathcal{L}_{\text{Dice}}$ directly maximizes the harmonic mean of precision and recall (F1 score) on the minority water classes, preventing the network from trivially predicting "Dry Land" everywhere.

### Q33: Define Intersection-over-Union (IoU) and Mean IoU (mIoU).
**Answer:**  
For a given class $c$:
$$\text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$$
Mean IoU (mIoU) is the unweighted average across all semantic classes:
$$\text{mIoU} = \frac{1}{K} \sum_{c=0}^{K-1} \text{IoU}_c \quad (\text{where } K = 3)$$

### Q34: What quantitative evaluation metrics did Model B achieve on the official Sen1Floods11 test split?
**Answer:**  
- **Overall Pixel Accuracy:** 95.50%
- **Validation mIoU:** 0.7217 (Best Epoch: 28)
- **Test mIoU:** 0.7198
- **Class 0 (Dry Land) IoU:** 0.949 (Precision: 0.978, Recall: 0.969)
- **Class 1 (Permanent Water) IoU:** 0.684 (Precision: 0.782, Recall: 0.841)
- **Class 2 (Mapped Floodwater) IoU:** 0.526 (Precision: 0.674, Recall: 0.698)

### Q35: Why is the IoU for Class 2 (Floodwater, 0.526) lower than Class 0 (Dry Land, 0.949)?
**Answer:**  
Class 2 represents an extreme minority class (often $< 3\%$ of pixels) characterized by highly irregular, thin, dynamic boundaries. Under the strict mathematical definition of IoU, even a 1-pixel boundary displacement along an extensive flood shoreline drastically penalizes the intersection over union, whereas the massive contiguous background of dry land easily achieves $> 0.94$ IoU. A flood IoU of 0.526 on hand-labeled Sen1Floods11 is competitive with published academic literature.

### Q36: What optimizer and learning rate schedule were used?
**Answer:**  
AdamW optimizer with an initial learning rate of $1 \times 10^{-4}$, weight decay of $1 \times 10^{-5}$, and a `ReduceLROnPlateau` scheduler (decay factor 0.5, patience 3 epochs). Training stopped after 28 epochs via early stopping when validation mIoU reached its peak of 0.7217.

### Q37: What data augmentations were applied during training?
**Answer:**  
Random horizontal flips ($p = 0.5$), random vertical flips ($p = 0.5$), and random 90-degree rotations. Augmentations were applied synchronously across both SAR and optical input tensor pairs as well as the ground truth label raster.

### Q38: How long did training take and on what hardware?
**Answer:**  
Training converged in approximately 48 minutes on an NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM) running PyTorch 2.5.1 with CUDA 12.1.

---

## Domain 6: Scientific Model Provenance: Model A vs Model B

### Q39: What is the exact distinction between Model A and Model B in FloodSense-AI?
**Answer:**  
- **Model B (`best_checkpoint.pth`):** The scientifically verified benchmark model. It has 8.24M parameters, was trained on the official Sen1Floods11 hand-labeled dataset for 28 epochs, and achieved 95.50% test accuracy and 0.7198 test mIoU.
- **Model A (`dual_stream_attention_unet.pth`):** A procedural demonstration baseline model. It has 40.6M parameters and was calibrated on 24 synthetic procedural tiles for 5 epochs to verify pipeline data structures. It has *no independently verified accuracy on real satellite imagery*.

### Q40: Why is it crucial to disclose this distinction during academic review?
**Answer:**  
Scientific integrity demands that researchers never claim benchmark results from a model that was not evaluated on that benchmark. Claiming that the 40.6M model achieved 0.7217 mIoU would be factually incorrect. FloodSense-AI documents the exact provenance, weights, training logs, and evaluations for each model in its code, UI, and reports.

### Q41: Which model is loaded by default in the integrated pipeline?
**Answer:**  
The pipeline configuration explicitly specifies and loads Model B (`backend/outputs/sen1floods11_training/best_checkpoint.pth`) for production inference.

### Q42: What was the Phase 4E.1 statistical reconciliation regarding Bolivia holdout labels?
**Answer:**  
During initial reporting, the narrative stated 376,966 reference water pixels, while the per-chip table summed to 386,588 pixels. Our audit recalculated the Class 2 pixels directly from source GeoTIFFs and identified that chip `Bolivia_291602` (9,622 pixels) was omitted from the narrative aggregation. The verified total of 386,588 pixels was reconciled across all documentation and unit tests.

### Q43: What was the correction regarding the "three-chip percentage"?
**Answer:**  
An earlier document claimed the top 3 chips contained "> 80%" of all floodwater pixels. The exact pixel counts are:
- `Bolivia_129334`: 152,073 pixels
- `Bolivia_314919`: 83,131 pixels
- `Bolivia_432776`: 69,982 pixels  
Sum = 305,186 pixels. Divided by 386,588, the exact percentage is **78.94%**. We corrected the text to state 78.94% and eliminated the erroneous "> 80%" claim.

### Q44: What are the two distinct evaluation pixel domains in Phase 4E?
**Answer:**  
1. **5-Chip Optical Clean Domain:** 1,310,720 valid pixels where cloud-free optical imagery is available for dual-stream comparison.
2. **12-Chip All-Weather SAR Domain:** 3,145,728 valid pixels across all holdout chips including cloudy scenes where SAR operates independently.

---

## Domain 7: GIS Infrastructure Exposure & Geodesic Area Mathematics

### Q45: How does the system query and ingest OpenStreetMap data?
**Answer:**  
The backend executes structured Overpass QL queries over HTTP (or loads pre-cached local OSM GeoJSON files for offline reliability). It queries `way["building"]` and `relation["building"]` for building parcels, and `way["highway"]` for road networks within the bounding box coordinates.

### Q46: How is building exposure categorized?
**Answer:**  
Building polygons are converted to a GeoPandas GeoDataFrame and overlaid onto the 10m classified flood raster:
- **High Exposure (Red):** $> 50\%$ of the building footprint intersects mapped floodwater pixels.
- **Moderate Exposure (Yellow):** $10\%$ to $50\%$ intersection.
- **Low / Nil Exposure (Green):** $< 10\%$ intersection.  
In the Chennai testbed (1,000 buildings sampled), 290 were High, 4 Moderate, and 706 Low.

### Q47: Does high building exposure mean the building has collapsed?
**Answer:**  
No. Our documentation explicitly highlights that the system measures 2D raster flood exposure, not structural engineering damage. It indicates that the ground parcel is submerged. Structural damage depends on construction materials, foundation height, and flood depth, which satellite imagery cannot observe directly.

### Q48: Explain the discrepancy between 21.58 km² and 21.05 km² in the Chennai flood extent.
**Answer:**  
The Chennai demonstration raster (`exec_8f7ef10bcb32_flood_mask.tif`) has 210,497 Class 2 flood pixels:
1. **Legacy Planar Code Formula (21.58 km²):** A legacy approximation calculated area using a flat degree-to-meter constant ($111.320\text{ km/degree}$) in both dimensions, omitting the cosine of latitude ($2,158.39\text{ ha} = 21.584\text{ km}^2$).
2. **True Geodesic WGS84 Calculation (21.05 km²):** In WGS 84, longitudinal distance decreases with latitude:
   $$\Delta x = \Delta\lambda \cdot 111.320 \cdot \cos(\text{lat})$$
   At Chennai's latitude ($13.0039^\circ\text{ N}$), $\cos(13.0039^\circ) \approx 0.9743$.
   Multiplying $21.584 \times 0.9743$ gives **21.049 km²** (row-integrated) and **21.050 km²** (centroid-based).
3. **Projected UTM Zone 44N Metric Area:** **21.025 km²**.  
All project documentation has been reconciled to report the scientifically verified **21.05 km²** geodesic area.

### Q49: What is the total geographic footprint of the Chennai demonstration raster?
**Answer:**  
Raster dimensions: 1,304 columns $\times$ 1,015 rows = 1,323,560 total pixels. At 10m nominal GSD, it covers $13.04\text{ km} \times 10.15\text{ km} \approx 132.35\text{ km}^2$. The 210,497 flood pixels represent 15.90% of the scene area.

### Q50: How are road segments classified for flood hazard?
**Answer:**  
Road linestrings are buffered according to highway hierarchy (motorway: 15m, primary: 12m, secondary: 10m, residential: 6m). The buffered polygon is intersected with the flood raster:
- **Road Hazard (Red):** $> 30\%$ floodwater contamination (504 segments in Chennai testbed).
- **Caution (Yellow):** $10\%$ to $30\%$ contamination (27 segments).
- **Clear (Green):** $< 10\%$ contamination (469 segments).

---

## Domain 8: Road Network Topology, Directed DiGraphs & A* Routing

### Q51: Why was the road network upgraded from an undirected Graph to a directed DiGraph?
**Answer:**  
An undirected graph allows traversal in both directions along any edge. However, real urban road networks have one-way streets. Routing an emergency convoy the wrong way down a one-way arterial road is hazardous and illegal. Upgrading to `networkx.DiGraph` enforces directional connectivity based on OpenStreetMap `oneway` tags.

### Q52: How are OpenStreetMap `oneway` tags parsed by the graph builder?
**Answer:**  
- `oneway=yes` or `oneway=1`: Adds a single directed edge from node $u$ to node $v$ ($u \to v$).
- `oneway=-1` or `oneway=reverse`: Adds a single reversed directed edge from $v$ to $u$ ($v \to u$).
- `oneway=no` or `oneway=0` (or absent): Adds bidirectional paired directed edges ($u \to v$ and $v \to u$).

### Q53: How does the A* routing algorithm incorporate flood hazard?
**Answer:**  
Standard A* minimizes path distance $f(n) = g(n) + h(n)$, where $h(n)$ is the Haversine distance heuristic. FloodSense-AI penalizes edge traversal based on flood inundation:
$$\text{cost}(u, v) = \text{length}(u, v) \cdot [1.0 + W_{\text{flood}} \cdot P_{\text{flood}}(u, v)]$$
where $P_{\text{flood}}(u, v)$ is the segment flood ratio, and $W_{\text{flood}} = 50.0$ is the flood penalty weight. If a road segment is $100\%$ flooded, its effective traversal cost increases by a factor of 51, forcing the search algorithm to explore longer dry detours rather than traversing deep water.

### Q54: How is the Estimated Time of Arrival (ETA) calculated, and what was the 9.4 vs 8.33 minute fix?
**Answer:**  
ETA is calculated using kinematic motion:
$$\text{ETA}_{\text{minutes}} = \frac{\text{distance\_meters} / 1000.0}{\text{speed\_kmh}} \times 60.0$$
For the Chennai route of 3,469.2 meters at an assumed heavy convoy speed of 25 km/h:
$$\text{ETA} = \frac{3.4692}{25} \times 60 = 8.326\text{ minutes} \approx 8.33\text{ minutes}$$
An earlier display reported 9.4 minutes due to an uncalibrated default speed of 22.1 km/h. The implementation and frontend display were corrected to display the exact 8.33 minutes at 25 km/h.

### Q55: What are the candidate route safety disclosures?
**Answer:**  
The system terms generated paths "Candidate Rescue Routes" because satellite remote sensing detects surface inundation but cannot verify:
1. Physical bridge structural integrity or foundation washout.
2. Localized debris blockages, downed electrical lines, or stalled vehicles.
3. Sub-surface sinkholes.  
Emergency convoy commanders are presented with clear operational notices that satellite candidate routes must be confirmed by field scout teams.

### Q56: How is system testing structured, and how many tests pass?
**Answer:**  
The project enforces automated test verification via `pytest`. There are four distinct test suites comprising **61 automated tests**, all executing and passing with 100% success rate in 7.59 seconds:
1. `test_phase6_pipeline.py` (19 tests): Pipeline orchestration, SSE events, manifest generation.
2. `test_phase5_gis_integrity.py` (13 tests): DiGraph oneway enforcement, A* convergence, kinematic ETA.
3. `test_phase4e_integrity.py` (14 tests): Sen1Floods11 Bolivia pixel counts (386,588), IoU checks.
4. `test_flood_change_detection_integrity.py` (15 tests): Temporal backscatter differencing and safeguards.

---

*End of Viva Examination Guide. All questions prepared for final B.E. CSE project defense.*

