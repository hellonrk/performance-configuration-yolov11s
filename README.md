# performance-configuration using yolov11s
Performance Configuration Analysis in Portuguese Traditional Music
This repository contains the complete source code, trained models, and resources for the Master's thesis, "Performance Configuration Analysis in Portuguese Traditional Music: A Computational Approach."

The project addresses the challenge of analyzing the vast MPAGDP (A Música Portuguesa A Gostar Dela Própria) archive, which contains over 8,000 field recordings. Manually classifying the number of performers in each video would be a prohibitively time-consuming task. To solve this, we developed a computational system using a fine-tuned YOLOv11s computer vision model to automatically detect and count performers in archival footage.

This automated approach allowed us to classify all 8,122 recordings into four meaningful categories (Solo, Duo, Small Group, Large Group) with a 96% classification accuracy. The resulting data provides new, large-scale quantitative insights into the social and regional dynamics of Portuguese musical practices, confirming, for example, the prevalence of solo performers in northern narrative traditions and large ensembles in Alentejo's collective singing traditions.




# Table of Contents
- Methodology
- Repository Structure
- System Requirements
- Workflow and Usage
- Key Results
- Trained Model
- Citation
- License

# Methodology
The project's success relies on a domain-adapted model tailored to the specific visual characteristics of the MPAGDP archive.

# Performance Classification
A four-part classification system was developed to analyze ensemble size based on musicological rationales:
  - Solo (1 performer): Represents individual expression, crucial for narrative and poetic genres where textual clarity is paramount.
  - Duo (2 performers): The transition to interactive music-making, enabling harmonic complexity while retaining intimacy, common in Fado and competitive traditions.
  - Small Group (3-5 performers): A flexible configuration that allows for greater instrumental diversity and often serves as a vehicle for stylistic innovation.
  - Large Group (5+ performers): Strongly associated with communal identity and ceremonial functions, such as the choral tradition of Cante Alentejano.


# Iterative Fine-Tuning (Active Learning)
A standard YOLO model pre-trained on general datasets would underperform on the unique challenges of field recordings (e.g., variable lighting, occlusions, cultural attire). To achieve high accuracy, we employed an iterative fine-tuning process:
  - An initial model was trained on a small, manually labeled set of 500 frames.
  - This model was used to generate predictions for the entire 40,600-frame dataset.
  - The predictions were analyzed to identify the most challenging cases (i.e., those with the lowest confidence and consistency scores).
  - These difficult frames were then prioritized for manual review and correction using the model's predictions as a starting point (model-assisted labeling).
  - The corrected labels were added back into the training set, and the model was fine-tuned again.

This active learning cycle was repeated five times, allowing the model to progressively learn from its mistakes and adapt to the specific nuances of the MPAGDP archive.

# Repository Structure
This repository is organized into functional directories that reflect the project's workflow.

.
├── CSV files/                # Contains final, categorized analysis results and other data tables.
├── Frame Extraction/         # Scripts to download videos and extract the 5 sample frames per video.
├── YOLOv11s/train8/          # Output from a YOLOv11s training run, including the final model weights and result graphics.
├── categorization/           # Scripts that use the trained model to analyze frames and categorize performances.
├── re-training/              # Tools for the active learning loop, including scripts to filter low-confidence results and assist with labeling.
├── yolo-training/            # An interactive script to launch the YOLO training process with custom parameters.
├── LICENSE                   # The project license.
└── README.md                 # This file.

# System Requirements
To run the scripts in this repository, you'll need Python 3 and the following key libraries:

- ultralytics
- pandas
- opencv-python
- yt-dlp

# Workflow and Usage
The project follows a sequential pipeline. The scripts are designed to be run in the following order:

1. Data Preparation
First, create the image dataset from the source videos.
- The Frame Extraction/download_vimeo_frames.py script is a parallel-processing tool that downloads videos from a list, extracts 5 frames from each with at least 10-second separation, saves them to an output folder, and cleans up the temporary video files.

2. Model Training
With a set of manually labeled frames, train the initial model.
- The yolo-training/run_yolo_training.py script provides an interactive command-line interface to start the YOLO training process. It prompts for the model path, dataset YAML, epochs, and other hyperparameters.

3. Analysis and Inference
Use the trained model to analyze the full dataset of 40,600 frames.
- The categorization/for-process.py script is an interactive tool that loads your trained model, processes all frames, and generates a detailed CSV report (performer_analysis_results.csv) with performer counts, confidence scores, consistency analysis, and a final verdict for each video.

4. Curation and Filtering
Separate the reliable results from those that need human verification.
- The re-training/filter_low_confidence.py script processes the analysis results and creates a separate CSV file (for_manual_review.csv) containing only the videos where the model's average confidence was below 70%.

5. Retraining and Model Improvement
Use the filtered, low-confidence results to improve the model.
- re-training/auto_labelling.py can generate "draft" labels for the low-confidence images to speed up manual correction.
- Once corrected, these challenging examples can be added to the dataset, and the model can be fine-tuned again using the yolo-training script.

# Key Results
The final fine-tuned YOLOv11s model achieved a 96% overall classification accuracy on a random validation set of 500 videos. The final analysis of the 8,122 recordings yielded the following distribution of performance configurations:

Category        Number of Videos        Percentage of Dataset

Solo            3,085                     38% 

Duo            1,037                      13% 

Small Group      964                      12% 

Large Group     1,332                      17% 


# Trained Model
The final trained model weights (best.pt) and all associated training metrics (confusion matrix, PR curve, etc.) are located in the /YOLOv11s/train8/ directory.

# Citation
If you use the code or findings from this project in your research, please cite the following thesis:

Khatri, Nawaraj. (2025). Performance Configuration Analysis in Portuguese Traditional Music: A Computational Approach. Master's Thesis, Faculdade de Engenharia da Universidade do Porto.

# License
This project is licensed under the GPL-3.0 License.
