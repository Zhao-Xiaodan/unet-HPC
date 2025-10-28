# U-Net Feature Visualization Analysis: 320x_2025-05-15_02-05-00

**Date**: October 28, 2025
**Model**: U-Net (PyTorch)
**Test Image**: 320x_2025-05-15_02-05-00
**Tile Position**: Row 3, Column 4 (center region)
**Analysis Directory**: `unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/`

---

## Executive Summary

This report presents an in-depth analysis of U-Net's internal representations for microbead segmentation using advanced feature visualization techniques. The analysis includes:

1. **Dimension-aware feature inversions** that reconstruct input images from layer activations at their native spatial resolutions (512×512 → 256×256 → 128×128 → 64×64 → 32×32)
2. **Representative feature map visualization** using PCA-based clustering to identify and display the most informative channels
3. **Layer-by-layer analysis** across all 9 encoder-decoder blocks

Key findings reveal how U-Net progressively abstracts microbead features through the encoding pathway and reconstructs spatial details through the decoding pathway.

---

## 1. Methodology

### 1.1 Architecture Overview

The U-Net architecture processes images through 5 spatial resolution levels:

| Layer | Spatial Resolution | Feature Channels | Feature Maps After |
|-------|-------------------|------------------|-------------------|
| **encoder_1** | 512×512 | 32 | conv1, conv2 |
| **encoder_2** | 256×256 | 64 | max_pool + conv1, conv2 |
| **encoder_3** | 128×128 | 128 | max_pool + conv1, conv2 |
| **encoder_4** | 64×64 | 256 | max_pool + conv1, conv2 |
| **bottleneck** | 32×32 | 512 | max_pool + conv1, conv2 |
| **decoder_4** | 64×64 | 256 | up_conv + skip + conv1, conv2 |
| **decoder_3** | 128×128 | 128 | up_conv + skip + conv1, conv2 |
| **decoder_2** | 256×256 | 64 | up_conv + skip + conv1, conv2 |
| **decoder_1** | 512×512 | 32 | up_conv + skip + conv1, conv2 |

### 1.2 Visualization Techniques

#### Feature Inversion
Feature inversion reconstructs what input patterns maximally activate specific layers by:
1. Starting from random noise at the layer's **native spatial resolution**
2. Optimizing the input to match captured activation patterns using gradient descent
3. Applying total variation regularization for smoother, more interpretable reconstructions
4. Upsampling lower-resolution inversions to 512×512 for consistent visualization

**Mathematical formulation**:
```
minimize: ||φ(x) - φ(x_target)||² + λ_TV · TV(x)
```
where φ(x) represents layer activations and TV(x) is the total variation loss.

#### Feature Map Clustering
To reduce redundancy among hundreds of feature channels:
1. Flatten each feature map to a 1D vector
2. Apply PCA dimensionality reduction to 2D space
3. Cluster using K-means (k=8 clusters)
4. Select representative feature maps from each cluster (closest to cluster centroid)

This approach reduces ~32-512 channels to ~8 representative visualizations per layer.

---

## 2. Input and Prediction Analysis

### Figure 1: Input Preprocessing and Model Prediction

![3-panel overview](unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/320x_2025-05-15_02-05-00_3panel.png)

**Figure 1**: Three-panel visualization showing (left) original microscopy image, (center) preprocessed input with percentile normalization, and (right) U-Net segmentation prediction. The tile shows a moderate density of microbeads with clear separation between particles.

**Observations**:
- **Original tile**: Grayscale microscopy image at 320× magnification showing microbeads as dark circular objects against a lighter background
- **Preprocessed input**: Percentile normalization (0.5th to 99.5th) enhances contrast and standardizes intensity distribution for model input
- **Prediction**: Binary segmentation mask successfully identifies individual microbeads with high precision, showing minimal false positives and capturing even touching or overlapping particles

The high-quality prediction demonstrates that the trained U-Net has learned robust feature representations suitable for microbead detection.

---

## 3. Feature Inversion Analysis: Encoder Pathway

Feature inversions reveal what visual patterns each layer has learned to detect by reconstructing inputs that would produce similar activations.

### 3.1 Encoder Layer 1 (512×512, 32 channels)

![Encoder 1 inversion](unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/feature_inversions/feature_inversion_encoder_1_conv2.png)

**Figure 2**: Feature inversion from encoder_1_conv2 at native 512×512 resolution showing low-level edge and texture patterns.

**Analysis**:
- Captures fine-grained spatial details and edge information
- Reconstructed pattern closely resembles the original input, indicating this layer preserves spatial structure
- Individual microbeads remain clearly visible as discrete circular objects
- This layer acts as an **edge detector and texture analyzer**, identifying boundaries and local intensity variations

### 3.2 Encoder Layer 2 (256×256, 64 channels)

![Encoder 2 inversion](unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/feature_inversions/feature_inversion_encoder_2_conv2.png)

**Figure 3**: Feature inversion from encoder_2_conv2 at 256×256 resolution showing intermediate-level pattern detection.

**Analysis**:
- Microbeads remain identifiable but with slightly reduced spatial precision
- Layer begins combining local edge information into **blob-like patterns**
- Activations correspond to regions of interest (microbeads) rather than pixel-level details
- Transition from texture detection to **object-level feature detection**

### 3.3 Encoder Layer 3 (128×128, 128 channels)

![Encoder 3 inversion](unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/feature_inversions/feature_inversion_encoder_3_conv2.png)

**Figure 4**: Feature inversion from encoder_3_conv2 at 128×128 resolution showing abstract shape representations.

**Analysis**:
- Individual microbeads become more abstract but maintain their spatial distribution
- Layer encodes **coarse shape information** and spatial relationships between objects
- Fine textural details are lost; emphasis shifts to overall structure
- This abstraction level is crucial for **position-invariant object detection**

### 3.4 Encoder Layer 4 (64×64, 256 channels)

![Encoder 4 inversion](unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/feature_inversions/feature_inversion_encoder_4_conv2.png)

**Figure 5**: Feature inversion from encoder_4_conv2 at 64×64 resolution showing highly abstract semantic features.

**Analysis**:
- Strong spatial downsampling produces blocky, coarse representations
- Layer captures **semantic information**: "presence of objects" rather than "exact object boundaries"
- Activations indicate **density patterns and global object distribution**
- Individual microbeads merge into regional activation zones
- Critical for encoding **context and global scene understanding**

### 3.5 Bottleneck (32×32, 512 channels)

![Bottleneck inversion](unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/feature_inversions/feature_inversion_bottleneck_conv2.png)

**Figure 6**: Feature inversion from bottleneck_conv2 at 32×32 resolution showing maximum abstraction and semantic encoding.

**Analysis**:
- **Highest level of abstraction**: Individual objects are no longer distinguishable
- Captures **global scene properties**: overall microbead density, spatial distribution patterns
- Blocky, coarse-grained representation encodes "what" and "where" at a semantic level
- This layer serves as the **information bottleneck**, compressing the scene into its most essential features
- Despite extreme compression (32×32 from 512×512 input), sufficient information is retained for accurate reconstruction in the decoder

---

## 4. Feature Inversion Analysis: Decoder Pathway

The decoder pathway progressively reconstructs spatial details while integrating semantic information from the bottleneck.

### 4.1 Decoder Layer 4 (64×64, 256 channels)

![Decoder 4 inversion](unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/feature_inversions/feature_inversion_decoder_4_conv2.png)

**Figure 7**: Feature inversion from decoder_4_conv2 showing initial spatial reconstruction from bottleneck.

**Analysis**:
- Combines upsampled bottleneck features with encoder_4 skip connections
- Begins **re-introducing spatial structure** into abstract semantic representations
- Blocky patterns similar to encoder_4 but now influenced by decoder processing
- Skip connections from encoder_4 provide fine-grained details lost during downsampling

### 4.2 Decoder Layer 3 (128×128, 128 channels)

![Decoder 3 inversion](unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/feature_inversions/feature_inversion_decoder_3_conv2.png)

**Figure 8**: Feature inversion from decoder_3_conv2 showing progressive refinement of object boundaries.

**Analysis**:
- Further spatial refinement; objects become more defined
- Integration of semantic understanding (from bottleneck) with spatial details (from encoder_3 skip)
- Layer begins **recovering object shapes** while maintaining semantic context
- Less blocky than decoder_4; smoother transitions between foreground and background

### 4.3 Decoder Layer 2 (256×256, 64 channels)

![Decoder 2 inversion](unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/feature_inversions/feature_inversion_decoder_2_conv2.png)

**Figure 9**: Feature inversion from decoder_2_conv2 showing high-resolution spatial reconstruction.

**Analysis**:
- Substantial recovery of spatial detail; microbeads appear as distinct, well-defined objects
- Combines semantic "object presence" information with precise spatial "object boundary" information
- Skip connection from encoder_2 reintroduces texture and edge details
- Critical layer for **accurate segmentation boundary localization**

### 4.4 Decoder Layer 1 (512×512, 32 channels)

![Decoder 1 inversion](unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/feature_inversions/feature_inversion_decoder_1_conv2.png)

**Figure 10**: Feature inversion from decoder_1_conv2 showing final high-resolution feature representations before output layer.

**Analysis**:
- Full spatial resolution reconstruction (512×512)
- Near-perfect recovery of input spatial structure with semantic refinement
- Skip connection from encoder_1 provides pixel-level detail
- Layer produces **refined features ready for final binary classification**
- Striking similarity to preprocessed input, demonstrating successful spatial reconstruction

---

## 5. Representative Feature Map Analysis

Feature maps visualize the actual activation patterns across channels. Due to high channel counts (32-512), we use PCA-based clustering to identify representative channels.

### 5.1 Encoder Feature Maps

#### Encoder Layer 1 (32 channels → 8 representatives)

![Encoder 1 features](unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/feature_maps/representative_feature_maps/feature_map_encoder_1_conv2.png)

**Figure 11**: Representative feature maps from encoder_1_conv2 showing diverse edge and texture detection filters.

**Observations**:
- High diversity in activation patterns (green, blue, yellow channels indicate different response magnitudes)
- Some channels respond strongly to **microbead edges** (blue/navy patterns)
- Other channels activate on **background textures** (green patterns)
- Yellow/bright green channels show **strong selective responses** to specific texture orientations
- Each representative captures a distinct aspect of low-level visual features

#### Encoder Layer 2 (64 channels → 8 representatives)

![Encoder 2 features](unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/feature_maps/representative_feature_maps/feature_map_encoder_2_conv2.png)

**Figure 12**: Representative feature maps from encoder_2_conv2 showing intermediate-level pattern detection with increased channel diversity.

**Observations**:
- Increased feature complexity compared to encoder_1
- Purple/dark blue channels show **blob detection** - activation on circular microbead regions
- Green/yellow channels demonstrate **background suppression** with selective foreground responses
- Feature maps show more abstract patterns than encoder_1, focusing on object-level features
- Clear distinction between "object detector" channels and "background filter" channels

#### Encoder Layer 4 (256 channels → 8 representatives)

![Encoder 4 features](unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/feature_maps/representative_feature_maps/feature_map_encoder_4_conv2.png)

**Figure 13**: Representative feature maps from encoder_4_conv2 showing highly abstract semantic feature detection.

**Observations**:
- Substantial abstraction; individual microbeads no longer clearly visible
- Feature maps show **regional activation patterns** rather than object-specific responses
- Diversity in activation intensity (green vs. blue vs. purple) indicates specialized semantic detectors
- Some channels activate broadly (green), others sparsely (purple/navy)
- This layer encodes "scene-level" information about microbead density and distribution

#### Bottleneck (512 channels → 8 representatives)

![Bottleneck features](unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/feature_maps/representative_feature_maps/feature_map_bottleneck_conv2.png)

**Figure 14**: Representative feature maps from bottleneck_conv2 showing maximum channel diversity and abstract semantic encoding.

**Observations**:
- **Highest channel count** (512) with greatest feature diversity
- Extremely coarse spatial resolution (32×32) but rich channel-wise information
- Color diversity (green, blue, purple, navy, yellow) indicates highly specialized semantic detectors
- Each channel encodes a specific aspect of the global scene: overall density, spatial clustering, texture uniformity
- This compressed representation must contain sufficient information to reconstruct the full segmentation mask

### 5.2 Decoder Feature Maps

#### Decoder Layer 1 (32 channels → 8 representatives)

![Decoder 1 features](unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/feature_maps/representative_feature_maps/feature_map_decoder_1_conv2.png)

**Figure 15**: Representative feature maps from decoder_1_conv2 showing refined segmentation features at full resolution.

**Observations**:
- High spatial detail recovery; individual microbeads clearly visible
- Yellow/bright channels show **strong foreground-background separation**
- Green channels provide **boundary refinement** information
- Blue/navy channels may encode **confidence or ambiguity** for difficult regions
- Feature diversity suggests ensemble-like processing where different channels vote for segmentation decisions

**Key Insight**: Comparing decoder_1 to encoder_1 reveals a critical difference: while encoder_1 detects raw edges and textures, decoder_1 produces **semantically-informed spatial features** that combine "what" (object identity from bottleneck) with "where" (precise location from skip connections).

---

## 6. Key Findings

### 6.1 Hierarchical Feature Learning

The U-Net demonstrates clear hierarchical feature abstraction:

1. **Encoder 1-2**: Low-level edge, texture, and boundary detection
2. **Encoder 3-4**: Mid-level object shape and spatial relationship encoding
3. **Bottleneck**: High-level semantic scene understanding (density, distribution)
4. **Decoder 4-3**: Semantic-to-spatial translation, progressive boundary refinement
5. **Decoder 2-1**: Pixel-precise spatial reconstruction with semantic guidance

### 6.2 Information Flow Through the Bottleneck

Despite extreme compression at the bottleneck (32×32×512), sufficient information is preserved for accurate reconstruction:

- **Spatial compression** (512×512 → 32×32 = 256× reduction) is compensated by **channel expansion** (1 → 512 channels)
- The 512 bottleneck channels act as a **distributed semantic encoding** of scene properties
- Skip connections are essential: they bypass the lossy bottleneck path to reintroduce fine spatial details

### 6.3 Feature Map Redundancy and Clustering

PCA-based clustering successfully reduces ~32-512 channels to ~8 representatives per layer:

- High similarity among many channels (hence clustering effectiveness)
- Suggests the network learns **redundant features** for robustness
- Representative features show diverse activation patterns, indicating they capture complementary aspects of the input

### 6.4 Dimension-Aware Feature Inversion Success

Feature inversions at native layer resolutions reveal:

- **Smooth abstraction gradient**: Progressive loss of spatial detail from encoder_1 → bottleneck
- **Successful spatial reconstruction**: Progressive recovery of spatial detail from bottleneck → decoder_1
- **Semantic preservation**: Even abstract layers (encoder_4, bottleneck) retain core scene information (object positions, density patterns)

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. **Missing UMAP Cluster Visualizations**
   - The current results use PCA-based clustering only
   - UMAP (Uniform Manifold Approximation and Projection) was not available on the HPC system during this run
   - **Expected improvement**: UMAP better preserves local structure and may reveal more meaningful feature groupings than linear PCA

2. **Single Threshold Analysis**
   - Clustering used k=8 clusters; alternative k values may reveal different representative features

3. **Single Tile Analysis**
   - This report analyzes one tile from one image; generalizations should be validated across all 8 test images

### 7.2 Code Improvements Made

Following this analysis, the visualization code has been **updated** to generate **both UMAP and PCA analyses simultaneously** for direct comparison:

#### Updated Output Structure:
```
image_name/
├── feature_maps/
│   ├── pca_clusters/                      # PCA scatter plots (NEW)
│   ├── umap_clusters/                     # UMAP scatter plots (NEW)
│   ├── comparison_umap_vs_pca/            # Side-by-side comparison (NEW)
│   ├── representative_feature_maps_pca/   # PCA representatives (UPDATED)
│   └── representative_feature_maps_umap/  # UMAP representatives (NEW)
└── feature_inversions/                    # (unchanged)
```

#### Key Code Changes:
1. **Dual clustering function** (`cluster_feature_maps_dual()`): Runs both UMAP and PCA
2. **Method-agnostic helper** (`_cluster_with_method()`): Handles both dimensionality reduction techniques
3. **Comparison plotting** (`plot_umap_pca_comparison()`): Side-by-side scatter plots
4. **Separate output directories**: PCA and UMAP results stored independently for comparison

### 7.3 Future Analyses

1. **UMAP vs PCA Comparison**
   - Once rerun with updated code, compare UMAP and PCA clustering results
   - Assess whether UMAP identifies different representative features
   - Evaluate which method better captures perceptually meaningful feature groupings

2. **Cross-Image Consistency**
   - Analyze whether similar feature patterns emerge across all 8 test images
   - Identify layer-wise feature universality vs. image-specific adaptations

3. **Attention Mechanism Analysis**
   - Extend analysis to Attention U-Net and Attention ResU-Net architectures
   - Compare how attention mechanisms modify feature representations

4. **Quantitative Feature Metrics**
   - Compute feature activation statistics (sparsity, selectivity)
   - Measure inter-channel similarity to quantify redundancy
   - Correlate feature activations with segmentation accuracy

---

## 8. Conclusions

This visualization analysis reveals the sophisticated hierarchical feature learning within the U-Net model for microbead segmentation:

1. **Progressive Abstraction**: The encoder pathway systematically transforms low-level pixel intensities into high-level semantic scene understanding through spatial compression and channel expansion.

2. **Effective Information Compression**: Despite 256× spatial downsampling at the bottleneck (512×512 → 32×32), the 512-channel representation preserves sufficient information for accurate reconstruction, demonstrating efficient distributed encoding.

3. **Skip Connection Importance**: Direct comparison between encoder and decoder feature inversions confirms that skip connections are critical for recovering fine spatial details lost during aggressive downsampling.

4. **Semantic-Spatial Integration**: Decoder layers successfully combine abstract semantic understanding from the bottleneck with precise spatial information from skip connections, enabling pixel-accurate segmentation.

5. **Feature Redundancy and Diversity**: While PCA clustering reveals redundancy among feature maps (motivating representative selection), the diverse activation patterns of representatives demonstrate that the network learns complementary, specialized detectors.

6. **Dimension-Aware Visualization Success**: Feature inversions at native layer resolutions provide interpretable insights into what each layer encodes, from edge detection (encoder_1) to semantic density patterns (bottleneck) to refined segmentation features (decoder_1).

The updated visualization pipeline with dual UMAP/PCA analysis will enable deeper insights into feature clustering and may reveal non-linear structure in feature spaces that PCA cannot capture. These visualizations provide a foundation for understanding, debugging, and improving neural segmentation models.

---

## References

**Related Files**:
- Visualization script: [`visualize_unet_features_advanced.py`](visualize_unet_features_advanced.py)
- PBS submission script: [`pbs_visualize_unet_features_advanced.sh`](pbs_visualize_unet_features_advanced.sh)
- Results directory: `unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/`
- Tile metadata: [`tile_metadata.json`](unet_visualization_advanced_20251028_000507/320x_2025-05-15_02-05-00/tile_metadata.json)

**Techniques**:
- Feature Inversion: Mahendran & Vedaldi, 2015. "Understanding Deep Image Representations by Inverting Them"
- PCA: Pearson, 1901. "On Lines and Planes of Closest Fit to Systems of Points in Space"
- UMAP: McInnes et al., 2018. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"
- U-Net Architecture: Ronneberger et al., 2015. "U-Net: Convolutional Networks for Biomedical Image Segmentation"

---

**Analysis Date**: October 28, 2025
**Analyst**: Claude (AI Assistant)
**Contact**: User's research group at phyzxi@nus.edu.sg
