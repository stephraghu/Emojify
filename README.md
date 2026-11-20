# Facial Expression Recognition & Live Emoji Matcher (FER-2013)

A deep learning computer vision project built in Google Colab (configured for a T4 GPU) that trains a Convolutional Neural Network (CNN) on the **FER-2013** facial expression dataset and matches captured webcam photos to corresponding emotion emojis.

---

## Overview

The notebook walks through an end-to-end workflow:
1. **Environment & Dependencies:** Installs/configures `opencv-python-headless==4.10.0.84` and imports TensorFlow/Keras, NumPy, Matplotlib, Seaborn, PIL, and Scikit-Learn.
2. **Emoji Assets:** Uploads and extracts `emojis.zip` containing emotion reference images (`angry.jpg`, `disgust.jpg`, `fear.jpg`, `happy.jpg`, `neutral.jpg`, `sad.jpg`, `surprise.jpg`, `znone.jpg`).
3. **Dataset Loading & Exploration:** Downloads `msambare/fer2013` via `kagglehub` (28,709 training images and 7,178 test/validation images across 7 emotion classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) and visualizes class distribution.
4. **Preprocessing & Augmentation:** Loads `48x48` grayscale images in batches of `64` with categorical labels, applying random horizontal flips, slight rotations (`0.05`), and slight zooms (`0.05`).
5. **CNN Model Training:** Builds a 3-block `Conv2D` + `BatchNormalization` + `MaxPooling2D` + `Dropout` architecture compiled with `Adam` and `categorical_crossentropy`, trained with `EarlyStopping` and `ReduceLROnPlateau`.
6. **Evaluation:** Generates a classification report and confusion matrix on the validation set, and plots training vs. validation accuracy across epochs.
7. **Webcam Capture & Visualization:** Provides browser-based webcam photo capture (`take_photo`) and side-by-side image/emoji display helpers (`plot_image`).

---

## Requirements

* **Runtime:** Python 3 with GPU acceleration (tested on Google Colab T4 GPU)
* **Key Python Packages:**
  * `tensorflow` / `keras`
  * `opencv-python-headless==4.10.0.84`
  * `numpy`
  * `matplotlib`
  * `seaborn`
  * `pillow`
  * `scikit-learn`
  * `kagglehub`

---

## Project Structure & Usage

* **Setup & Imports:** Run the package setup and import cells first. (Note: the first cell contains a typo in its `pip uninstall` line: `opencv -python` / `opencv-contib-python`.)
* **Emoji Archive:** Upload `emojis.zip` when prompted by `files.upload()`, then unzip into `./emojis`.
* **Dataset & Data Generators:** Downloads FER-2013 via `kagglehub.dataset_download("msambare/fer2013")` and creates `train_gen` and `val_gen`.
* **Class Weights Note:** There are two consecutive `class_weights` definitions in the notebook—the second one sets `0: ()`, which should be a float (such as `1.0`) before running `model.fit()`.
* **Training & Evaluation:** Fits the model for up to 50 epochs with callbacks, plots accuracy curves, and prints `classification_report` and a `confusion_matrix` heatmap.
* **Webcam Capture:** Run `take_photo()` in Colab to capture a `photo.jpg` image from your webcam and display it.

---

## Notes & Known Issues in the Notebook

* **Class Weights Typo:** The second `class_weights` cell assigns an empty tuple `()` to class `0` (`angry`). Update `0: ()` to a numeric weight (e.g., `0: 1.0`) to avoid training errors.
* **Pip Uninstall Command:** The initial `!pip uninstall` command has spacing/spelling typos (`opencv -python`, `opencv-contib-python`) and prints a usage warning before installing `opencv-python-headless==4.10.0.84`.
* **Colab Dependency:** `google.colab.patches.cv2_imshow`, `google.colab.files`, and `google.colab.output.eval_js` are specific to Google Colab environments.
