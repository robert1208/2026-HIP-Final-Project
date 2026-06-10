# Human Information Processing Final Project - Decoding Left and Right Hand Motor Imagery from EEG Signals

## 1. Introduction

### Project Title

**Decoding Left and Right Hand Motor Imagery from EEG Signals**

### Introduction & Cognitive Mechanism

This project investigates **motor imagery**, a human information processing function in which a person mentally simulates a movement without physically executing it. Motor imagery is widely used in brain-computer interface (BCI) research because imagined movements can modulate sensorimotor EEG rhythms in a way that can be detected and visualized. In this project, we focus on left- and right-hand motor imagery and examine whether these two mental movement conditions produce distinguishable scalp-level EEG power distributions after preprocessing.

Our core hypothesis is that left- and right-hand motor imagery trials will show different topographic patterns in the 8-30 Hz frequency range. This range includes the mu rhythm, approximately 8-13 Hz, and the beta rhythm, approximately 13-30 Hz. Both rhythms are strongly associated with sensorimotor processing, motor planning, and imagined movement. A common neurophysiological marker of motor imagery is event-related desynchronization (ERD), in which mu/beta power decreases during motor imagery relative to a baseline or resting condition. Therefore, our analysis focuses not only on the absolute bandpower topographies of left and right imagery, but also on action-minus-rest difference maps.

The objective of this project is to implement an end-to-end streaming EEG analysis pipeline in NeuroPype. The pipeline covers signal import, streaming simulation, timestamp correction, channel-location assignment, bandpass filtering, artifact removal, event segmentation, spectral feature extraction, topographic visualization, and action-rest difference maps. By presenting time-series plots, before/after PSD plots, condition-specific topoplots, and action-minus-rest maps, this project demonstrates how NeuroPype can support real-time or near-real-time EEG preprocessing and visualization for motor imagery analysis.

## 2. Data Source & Technical Parameters

### Data Source

This project uses **BCI Competition IV Dataset 1**, an EEG dataset collected for motor imagery-based BCI research. The dataset contains calibration and evaluation recordings from multiple subjects performing left- and right-hand motor imagery. The main NeuroPype demonstration uses subject `ds1e`, which showed stable performance in preliminary binary classification tests.

References and local files:

- BCI Competition IV Dataset 1
- Dataset description: https://www.bbci.de/competition/iv/desc_1.html
- Local files:
  - `BCICIV_1calib_1000Hz_mat/BCICIV_calib_ds1e_1000Hz.mat`
  - `BCICIV_1eval_1000Hz_mat/BCICIV_eval_ds1e_1000Hz.mat`
  - `true_labels/BCICIV_eval_ds1e_1000Hz_true_y.mat`

### Recording Device Information

The original data are EEG recordings with:

- Number of EEG channels: 59
- Original sampling rate: 1000 Hz
- Converted streaming sampling rate: 250 Hz
- Unit: microvolts, obtained by multiplying the original `cnt` values by 0.1
- Experimental labels: `left` and `right`; this project additionally adds a `rest` marker at the end of each trial

The original MATLAB `.mat` files were converted into NeuroPype-friendly CSV files:

- `eeg_with_markers_rest.csv`: continuous EEG data with an embedded `marker` column
- `channels.csv`: channel labels and 2D electrode coordinates
- `markers_with_rest.csv`: trial-onset and rest events
- `metadata.json`: source file, sampling rate, unit, and metadata description

### Original Purpose

This dataset was collected by the Berlin Brain-Computer Interface (Berlin BCI) team, primarily for evaluating classification algorithms on continuous EEG data.

The experiment involved healthy subjects performing motor imagery tasks, which included imagining movements of the left hand, right hand, or foot, without any feedback provided during the process.

The original research objective was to challenge whether algorithms could accurately identify the subjects' continuously changing intentions in an "uncued" and "asynchronous" state, and correctly output a 0 during rest periods when the subjects had no intentional control (no motor imagery).


## 3. Data Preprocessing & Quality Control

### Visual Inspection of Raw Data

![Raw EEG time series](imgs/Raw_Signal_Visual_Inspection.png)
Visual inspection should highlight:

- Continuous EEG waveforms across 59 channels
- Embedded markers for `left`, `right`, and `rest`
- Possible transient artifacts or high-amplitude activity

### Artifact Identification

Several signal-quality issues were identified:

1. **Potential bad channel: CCP2**  
   The standard deviation of `CCP2` was approximately 55.83 uV, whereas the median channel standard deviation was approximately 13.75 uV. This roughly fourfold difference suggests possible electrode noise, bad contact, or a non-neural artifact source. The final pipeline did not manually remove `CCP2`, because the project goal is to demonstrate a streaming NeuroPype workflow rather than to optimize classification performance. Therefore, `CCP2` is reported as a quality-control observation and a limitation in the interpretation.
![CCP2](imgs/CCP2.png)

2. **Transient high-amplitude activity**  
   Some time periods contained sudden peaks or large fluctuations, which may reflect muscle activity, cable movement, or environmental interference.

3. **Baseline drift and non-target spectral components**  
   The raw signal contained low-frequency drift and frequency components outside the motor imagery range, motivating bandpass filtering.

### Preprocessing Pipeline

The NeuroPype preprocessing pipeline is:

```text
Import CSV
-> Stream Data
-> Dejitter Timestamps
-> Assign Channel Locations
-> FIR Filter
-> Artifact Removal
-> Assign Target Values
-> Segmentation
-> Select Instances
-> Power Spectrum (Welch)
-> Power Bands
-> Mean over time/instances
-> Hold Last Packet for rest reference
-> Subtract action-rest maps
-> TopoPlot Viewer
```

### Filter Parameters

FIR Filter configuration:

- Filter along axis: `time`
- Filter mode: `bandpass`
- Frequencies: `[7, 8, 30, 32]`
- Minimum stopband attenuation: `50 dB`
- Filtering direction: forward

This filter preserves the 8-30 Hz range associated with alpha and beta rhythms in motor imagery.

### Before and After Preprocessing
**Raw data**
![raw data before preprocessing](imgs/Org.png)
![raw data after preprocessing](imgs/Processed.png)

**PSD**

![PSD before preprocessing](imgs/Before_PSD.png)
![PSD after preprocessing](imgs/After_PSD.png)


Expected results:

- Reduced power outside the 8-30 Hz target range after bandpass filtering
- Reduced transient and high-amplitude noise after artifact removal
- A cleaner spectral representation of motor imagery-relevant rhythms

## 4. NeuroPype Architecture Implementation

### Pipeline Architecture

The NeuroPype pipeline is designed as a streaming EEG workflow. The input file is `eeg_with_markers_rest.csv`, which contains both continuous EEG signals and an embedded marker column. This design avoids the need to merge separate EEG and marker streams. The Stream Data node simulates continuous data playback, and Dejitter Timestamps corrects timing irregularities. Assign Channel Locations assigns spatial positions to the 59 EEG channels so that the TopoPlot Viewer can display scalp maps. An FIR bandpass filter then preserves the 8-30 Hz motor imagery range. Artifact Removal reduces the influence of transient artifacts. The final pipeline does not manually reject `CCP2`, it is retained and reported as a data-quality limitation.



The Assign Target Values node maps markers to numerical targets:

```python
{'left': -1, 'right': 1, 'rest': 0}
```

Segmentation extracts epochs from 0.5 to 3.5 seconds after each event marker, focusing on the stable motor imagery interval while avoiding the cue onset and trial end. Select Instances separates trials by condition. Each condition is processed through Welch power spectrum estimation and Power Bands extraction. Mean nodes average the spectral features over time and instances, and TopoPlot Viewer displays scalp-level bandpower maps. Since this is a streaming pipeline, a Hold Last Packet node stores the latest available rest topography as a baseline. Subtract nodes then compute `left-rest` and `right-rest` maps in streaming mode.

### Parameter Justification

- **250 Hz sampling rate**: sufficient for 8-30 Hz motor imagery analysis while improving streaming efficiency.
- **8-30 Hz bandpass filter**: targets alpha and beta rhythms, the main EEG indices for motor imagery.
- **Segmentation [0.5, 3.5] seconds**: captures the stable imagery period and avoids early cue-related transients.
- **Welch power spectrum**: provides stable power spectral density estimates for block-based streaming data.
- **Power Bands with average across channels disabled**: preserves channel-level bandpower required for topoplot visualization.
- **Hold Last Packet**: stores the latest valid rest topography because left, right, and rest epochs do not arrive simultaneously in streaming mode.
- **Subtract nodes**: compute `left-rest` and `right-rest` maps to show motor imagery changes relative to the most recent rest baseline.
- **TopoPlot Viewer**: visualizes spatial EEG power distributions across the scalp.

### Visual Pipeline Graph

![pipeline](imgs/pipeline.png)

## 5. End-to-End Demo Video Script (2:00-2:30)

### English Script

Hello, this is our Human Information Processing final project. Our project demonstrates a NeuroPype-based streaming EEG pipeline for visualizing topographic differences between left- and right-hand motor imagery.

We used BCI Competition IV Dataset 1, focusing on subject ds1e for the main demonstration. The original EEG recordings contain 59 channels sampled at 1000 Hz. For streaming efficiency, we converted the data to 250 Hz CSV files and embedded event markers directly into the EEG CSV. The marker labels include left, right, and rest.

In NeuroPype, the data are imported through the Import CSV node and streamed using Stream Data. Dejitter Timestamps corrects timing irregularities, and an FIR bandpass filter preserves the 8-30 Hz range, which includes mu and beta rhythms related to motor imagery. Artifact Removal is then used to reduce the influence of noisy channels or transient artifacts.

Next, Assign Target Values maps left, right, and rest events to -1, 1, and 0. Segmentation extracts epochs from 0.5 to 3.5 seconds after each marker, focusing on a stable motor imagery interval. Select Instances separates the trials by condition.

For feature extraction, we compute Welch power spectra and extract power-band features. Mean nodes average the features over time and instances, and TopoPlot Viewer displays scalp-level maps for left, right, left-rest, and right-rest conditions. Since the pipeline is streaming, a Hold Last Packet node stores the most recent rest map as the subtraction baseline.

The results show different spatial bandpower patterns between left and right motor imagery, especially around sensorimotor regions.
## 6. Analytical Results & Interpretation

### Results Presentation

![Original and processed time series](imgs/result.png)

### Neurophysiological Interpretation

The left and right motor imagery topoplots show different 8-30 Hz bandpower distributions across the scalp. The most relevant regions are central and sensorimotor electrodes such as C3, C4, Cz, CP3, and CP4. These areas are associated with motor planning, sensorimotor integration, and imagined movement. Because the final pipeline did not manually remove `CCP2`, its unusually high variance may influence local color scaling or topographic patterns. Therefore, the results should be interpreted as a streaming visualization of overall spatial trends rather than a channel-rejection-optimized analysis.

In the action-minus-rest maps, negative values indicate that bandpower during motor imagery is lower than during rest. This pattern is consistent with event-related desynchronization (ERD), a common motor imagery phenomenon in which alpha/beta power is suppressed during imagined movement. Therefore, the topographic results support the idea that left- and right-hand motor imagery can be observed through spatial changes in EEG alpha/beta bandpower.

However, topoplots represent scalp-level EEG power distributions and should not be interpreted as precise cortical source localization. The results should therefore be understood as electrode-level spectral differences associated with the motor imagery task.