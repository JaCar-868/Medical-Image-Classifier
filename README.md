# Medical Image Classifier

A TensorFlow-powered command-line tool for binary disease detection from medical images stored as PyTorch `.pt` tensors.  
Built with reproducibility, transfer learning, and a robust `tf.data` pipeline.

---

## Features

- **Configurable CLI** — All major settings (paths, image size, batch size, epochs, learning rate, splits, output paths, random seed, log directory) are exposed via command-line arguments.  
- **Reproducibility** — Seeds set for Python `random`, NumPy, and TensorFlow.  
- **Stratified Splits** — Train / validation / test splits preserve class balance.  
- **On-the-fly Data Loading** — `.pt` files are loaded per-batch via `tf.numpy_function`, avoiding large in-memory arrays.  
- **Preprocessing & Augmentation**  
  - Resize to a fixed square resolution (default 224×224)  
  - Per-image standardization (zero mean, unit variance)  
  - Keras preprocessing layers for random flip, rotation, zoom, and contrast  
- **Transfer Learning** — Frozen ImageNet-pretrained ResNet50 backbone + GlobalAveragePooling + BatchNorm + Dropout  
- **Metrics & Callbacks**  
  - Tracks **accuracy**, **AUC**, **precision**, and **recall**  
  - Uses **EarlyStopping**, **ReduceLROnPlateau**, **ModelCheckpoint**, and **TensorBoard**  
- **Evaluation** — Prints test metrics, detailed classification report, and confusion matrix

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/JaCar-868/Medical-Image-Classifier.git
   cd Medical-Image-Classifier
Create a virtual environment and install dependencies:

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch tensorflow scikit-learn

## Data Preparation
Your data directory must have two subfolders, each containing .pt files:

/path/to/data_dir/
├── Positive_tensors/
│   ├── img_0001.pt
│   ├── img_0002.pt
│   └── …
└── Negative_tensors/
    ├── img_1001.pt
    ├── img_1002.pt
    └── …
Each .pt should load to a tensor of shape C×H×W or H×W×C.

## Usage

python medical_image_classifier.py \
  --data_dir     /path/to/data_dir \
  --img_size     224 \
  --batch_size   32 \
  --epochs       50 \
  --lr           1e-3 \
  --test_size    0.1 \
  --val_size     0.1 \
  --seed         42 \
  --model_out    best_model.h5 \
  --log_dir      logs/
  
Command-Line Arguments
Argument	Type	Default	Description
--data_dir	string	—	Root folder containing Positive_tensors/ and Negative_tensors/.
--img_size	int	224	Height and width to resize each image to (square).
--batch_size	int	32	Number of samples per gradient update.
--epochs	int	50	Maximum number of training epochs.
--lr	float	1e-3	Base learning rate for Adam optimizer.
--test_size	float	0.1	Fraction of the dataset reserved for the final test split.
--val_size	float	0.1	Fraction of the remaining data reserved for validation.
--seed	int	42	Random seed for Python, NumPy, and TensorFlow.
--model_out	string	best_model.h5	Filepath to save the best-performing model weights.
--log_dir	string	logs/	Directory for TensorBoard logs.

## Training & Callbacks
EarlyStopping monitors val_loss (patience = 5)

ReduceLROnPlateau reduces LR by factor 0.5 on val_loss plateau (patience = 3, min_lr = 1e-6)

ModelCheckpoint saves only the best model by val_accuracy

TensorBoard logs are written to <log_dir>/

Launch TensorBoard to monitor training in real time:


tensorboard --logdir logs/

## Evaluation
After training, the script will:

Print test‐set loss and all tracked metrics (accuracy, auc, precision, recall).

Display a full classification report (precision, recall, F1-score) and confusion matrix on the test set.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/JaCar-868/Medical-Image-Classifier/blob/main/LICENSE) file for more details.
