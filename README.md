Summary:
This project implements a Convolutional Neural Network (CNN) from scratch using only numpy. It includes:
- Weight initialization with He normalization
- Two convolutional layers with ReLU and max-pooling
- Two fully connected layers
- Binary classification with sigmoid output
- Vectorized forward and backwards propagation
- JSON-based weight saving/loading
- Data loading and evaluation using PyTorch's "DataLoader"

Weights:
- Weights initialized using He initialization:
- Biases are initialized to zero
- Weights are saved automatically after each epoch to cnn_weights_copy.json

Model architecture:
Convolutional Layer 1:
- 4 filters
- 3x3 kernel, 1 input channel (grayscale)
- padding = 1, stride = 1

Max Pooling Layer 1
- Pool size 2x2
- Stride = 2


Convolutional Layer 2:
- 8 filters
- 3x3 kernel, 4 input channels
- padding = 1, stride = 1

Max Pooling Layer 2
- Pool size 2x2
- Stride = 2


Fully connected Layer 1:
- Takes flattened feature map
- 64 output units

Fully connected Layer 2:
- 1 output unit (binary classification)


To train:
1. Download 25000 images of cats and dogs:
https://www.microsoft.com/en-us/download/details.aspx?id=54765
- Make sure folder structure is like:
- CatsAndDogs/
- ├── cat/
- │   ├── 0.jpg
- │   ├── ...
- ├── dog/
- │   ├── 0.jpg
- │   ├── ...

2. Split this folder into another folder for evaluation:
- Eval/
- ├── cat/
- │   ├── 10000.jpg
- │   ├── ...
- │   ├── 12499.jpg
- ├── dog/
- │   ├── dog.0.jpg
- │   ├── ...
 -│   ├── 12499.jpg

2. Set the following line in CNN.py to true
- use_image = False

After training, weights are automatically saved to json. Every 5 epochs, accuracy is computed using a separate Eval/ directory (same structure as CatsAndDogs)



