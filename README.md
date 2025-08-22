# Object-Counting
This is a project using python and openCV to perform universal object detection and counting
# Template Matching Algorithm Documentation

## Overview

This project implements a comprehensive template matching system using computer vision techniques to detect and count specific patterns or objects within images. The application combines multiple image processing methodologies to achieve robust pattern recognition under various conditions including scale variations, rotations, and different lighting conditions.

## Core Methodologies

### 1. Template Matching Algorithm

**Algorithm Used**: OpenCV's `cv2.matchTemplate()` with Normalized Cross-Correlation (`TM_CCOEFF_NORMED`)

**Methodology**:
- **Sliding Window Approach**: The template is systematically moved across the entire image pixel by pixel
- **Correlation Calculation**: At each position, the algorithm computes the normalized cross-correlation coefficient between the template and the corresponding image region
- **Matching Score**: Returns values between -1 and 1, where 1 indicates a perfect match
- **Threshold-based Detection**: Matches above a user-defined threshold are considered valid detections

**Mathematical Foundation**:
```
R(x,y) = Σ[T'(x',y') × I'(x+x',y+y')] / √[Σ[T'(x',y')²] × Σ[I'(x+x',y+y')²]]
```
Where:
- R(x,y) is the correlation coefficient at position (x,y)
- T'(x',y') is the normalized template
- I'(x+x',y+y') is the normalized image patch

### 2. Multi-Scale Template Matching

**Purpose**: Detect objects of varying sizes when the template and target objects may have different scales.

**Implementation**:
- **Scale Range**: 0.5x to 2.5x (50% to 250% of original template size)
- **Scale Steps**: 30 evenly distributed scale factors using `np.linspace(0.5, 2.5, 30)`
- **Resize Method**: `cv2.INTER_AREA` interpolation for downscaling, providing better quality for size reduction

**Process Flow**:
1. Generate scale factors
2. Resize template for each scale
3. Perform template matching at each scale
4. Collect all matches with their corresponding scale information

### 3. Rotational Template Matching

**Purpose**: Detect objects that may appear at different orientations in the target image.

**Implementation**:
- **Rotation Range**: 0° to 360°
- **Angular Steps**: 24 evenly distributed angles (15° increments)
- **Rotation Method**: `cv2.getRotationMatrix2D()` and `cv2.warpAffine()`

**Process Flow**:
1. For each scale, generate rotation matrices
2. Apply rotation transformation to the scaled template
3. Perform template matching with rotated template
4. Store matches with rotation angle information

### 4. Color-Based Thresholding

**Algorithm**: HSV Color Space Filtering with Morphological Operations

**Methodology**:
- **Color Space Conversion**: BGR to HSV for better color separation
- **Adaptive Thresholding**: Based on template's mean HSV values
- **Range Calculation**: User-adjustable color tolerance range

**HSV Thresholding Logic**:
```python
mean_color = np.mean(hsv_template, axis=(0,1))  # [H, S, V]
lower_bound = [H-range, max(50,S-50), max(50,V-50)]
upper_bound = [H+range, 255, 255]
```

**Morphological Cleanup**:
- **Opening Operation**: Removes small noise pixels
- **Closing Operation**: Fills small holes within objects
- **Kernel**: 3×3 structuring element

### 5. Non-Maximum Suppression (NMS)

**Purpose**: Eliminate overlapping detections and reduce false positives.

**Algorithm**: Intersection over Union (IoU) based filtering

**Implementation**:
```python
IoU = Intersection_Area / (Area1 + Area2 - Intersection_Area)
```

**Process**:
1. Sort matches by confidence score (descending)
2. For each match, calculate IoU with all previously accepted matches
3. Reject matches with IoU > 0.5 (50% overlap threshold)
4. Accept non-overlapping matches

### 6. Image Enhancement Techniques

#### Contrast Limited Adaptive Histogram Equalization (CLAHE)
- **Purpose**: Improve local contrast in grayscale images
- **Parameters**: Clip limit = 2.0, Tile grid = 8×8
- **Application**: Applied to both template and search image when grayscale mode is enabled

#### Blur Filtering Options
1. **Gaussian Blur**: Reduces high-frequency noise
   - Kernel size: 5×5
   - Sigma: Automatically calculated

2. **Median Blur**: Effective against salt-and-pepper noise
   - Kernel size: 5×5

3. **Bilateral Filter**: Edge-preserving smoothing
   - Parameters: d=9, sigmaColor=75, sigmaSpace=75

## Applications of the Algorithms

### 1. Industrial Quality Control
- **Use Case**: Detecting defective components on assembly lines
- **Algorithm Application**: Multi-scale matching handles varying component sizes due to camera distance variations
- **Color Thresholding**: Isolates specific component colors from background
- **Benefits**: Automated inspection reduces human error and increases throughput

### 2. Medical Image Analysis
- **Use Case**: Identifying specific anatomical structures or abnormalities
- **Algorithm Application**: 
  - Rotational matching accommodates patient positioning variations
  - CLAHE enhancement improves visibility of low-contrast features
- **Benefits**: Assists radiologists in diagnosis and reduces examination time

### 3. Surveillance and Security
- **Use Case**: Detecting specific objects or persons in CCTV footage
- **Algorithm Application**:
  - Multi-scale matching handles objects at different distances
  - Color thresholding can isolate objects by clothing color
- **Benefits**: Automated monitoring and alert generation

### 4. Retail and Inventory Management
- **Use Case**: Counting products on shelves or identifying specific items
- **Algorithm Application**:
  - Template matching counts identical products
  - NMS prevents double-counting of overlapping items
- **Benefits**: Automated inventory tracking and stock level monitoring

### 5. Gaming and Automation
- **Use Case**: Bot development for automated gameplay
- **Algorithm Application**:
  - Template matching identifies UI elements or game objects
  - Multi-scale matching handles different screen resolutions
- **Benefits**: Consistent automation across different display configurations

### 6. Document Processing
- **Use Case**: Finding logos, stamps, or specific text patterns
- **Algorithm Application**:
  - Rotational matching handles scanned documents with varying orientations
  - Enhancement techniques improve OCR accuracy
- **Benefits**: Automated document classification and data extraction

### 7. Archaeological and Historical Research
- **Use Case**: Identifying artifacts or patterns in archaeological images
- **Algorithm Application**:
  - Multi-scale matching handles artifacts photographed at different distances
  - Color thresholding isolates artifacts from soil background
- **Benefits**: Systematic cataloging and pattern analysis

## Technical Implementation Details

### Performance Optimizations

1. **Early Termination**: Skip processing if template becomes too small after scaling
2. **Size Validation**: Prevent crashes by ensuring minimum template dimensions (3×3 pixels)
3. **Memory Management**: Process matches in batches to prevent memory overflow
4. **Efficient Data Structures**: Use NumPy arrays for fast mathematical operations

### Error Handling

1. **Template Size Validation**: Ensures template remains valid after transformations
2. **Color Mask Validation**: Warns users if color thresholding results in empty masks
3. **OpenCV Error Catching**: Gracefully handles mathematical errors during matching

### User Interface Integration

1. **Real-time Parameter Adjustment**: Sliders for threshold and color range provide immediate feedback
2. **Visual Debugging**: Display intermediate processing results for troubleshooting
3. **Result Visualization**: Draws rotated bounding boxes around detected matches
4. **Match Counting**: Provides quantitative results for analysis

## Limitations and Considerations

### 1. Computational Complexity
- **Time Complexity**: O(n × m × s × r) where n,m are image dimensions, s is scale count, r is rotation count
- **Memory Usage**: Proportional to image size and number of transformations

### 2. Template Quality Requirements
- Templates should be representative of target objects
- High contrast templates perform better
- Templates should avoid repetitive patterns that may cause false positives

### 3. Environmental Sensitivity
- Lighting variations can affect matching accuracy
- Significant perspective changes may reduce detection performance
- Partial occlusion of target objects reduces matching scores

### 4. Parameter Tuning
- Threshold selection requires domain expertise
- Color range parameters need adjustment based on lighting conditions
- Scale and rotation ranges should be optimized for specific applications

## Conclusion

This template matching system demonstrates the effective combination of multiple computer vision algorithms to create a robust object detection tool. The integration of multi-scale and rotational matching with advanced image enhancement techniques provides flexibility across diverse application domains. The modular design allows users to enable or disable specific features based on their requirements, making it adaptable to various use cases while maintaining high detection accuracy.
