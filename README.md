# CIFAR-10 Classification - Manual Model Creation
This project demonstrates how to manually create and train a neural network from scratch on the CIFAR-10 dataset using PyTorch.

📁 Dataset
The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is automatically downloaded via PyTorch's torchvision.datasets.

🖥️ Case 1: Run Locally
✅ Requirements
Make sure you have the following installed:

python>=3.8
torch
torchvision
matplotlib
You can install the dependencies using:


pip install torch torchvision matplotlib
📦 Setup
Clone the repository or download the code.

Navigate to the project folder.

Run the training script:


python Net_CIFAR10.py
Net_CIFAR10.py is the main script where the model architecture is defined and trained manually.

🔍 Customization
You can modify the architecture inside BT2.py under the model class (e.g., MyCNN).

Hyperparameters like learning rate, batch size, and epochs can be edited at the top of the script.
-b : batch_size
-d : path dataset(if you don't download dataset yet, dataset will be automatically downloaded )
-a : path for pre-trained model
-g : gpu id
-j : number worker(default: 4)
-e : total epochs you want to train (default: 200)
-l : learning-rate (default: 0.1)
-s : schedule
-re : path for model you trained.
-se : start epoch.  

☁️ Case 2: Run on Google Colab
🔗 Setup
Open Google Colab.

git clone this project to google colab 

Ensure you're using a GPU runtime:

Go to Runtime > Change runtime type and select GPU.

▶️ Run the Project
In a Colab cell:

!python Net_CIFAR10.py
Alternatively, copy the code from Net_CIFAR10.py directly into a cell and run.

🧠 Project Highlights
Manual model definition using nn.Module

Training and evaluation loop written from scratch

Option to visualize accuracy and loss curves (if implemented)

📂 File Structure

.
├── train.py           # Main training script
├── README.md          # This file
└── utils.py           # (Optional) Helper functions (e.g., plotting, metrics)
