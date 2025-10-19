import sys
import os
import json

from numpy.lib.stride_tricks import as_strided
import numpy as np

# LIterally only for image loading and transforming 
from dataloader import Loader
from torchvision import transforms
from torch.utils.data import DataLoader

class CNN:
    def __init__(self):
        # if weights exist, use weights
        if os.path.exists('cnn_weights.json'):
            self.load_weights('cnn_weights.json')
        else:
            # 2 layers convolutional filters and biases
            self.conv1_filters = np.random.randn(4, 3, 3, 1) * np.sqrt(2 / 9)  # 4 filters of size 3x3, through the one channel
            self.conv2_filters = np.random.randn(8, 3, 3, 4) * np.sqrt(2 / 36) # 8 filters of size 3x3, through the four channels
            self.conv1_bias = np.zeros(4) # bias for the first convolution
            self.conv2_bias = np.zeros(8) # bias for the second convolution

            # third layer connected layers weights and biases
            self.fc1_weights = np.random.randn(2048, 64) * np.sqrt(2 / 2048)  # Flattened to be 2048, points to 64 neurons
            self.fc1_biases = np.zeros(64) # 64 biases

            # fourth layer
            self.fc2_weights = np.random.randn(64, 2) * np.sqrt(2 / 64)  # 64 neurons, points to 2 neurons
            self.fc2_biases = np.zeros(2) # last two biases for output

            # Save the initialized weights
            self.save_weights('cnn_weights.json')
            print("Weights generated")

    def save_weights(self, filepath):
        weights_dict = {
            'conv1_filters': self.conv1_filters.tolist(),
            'conv2_filters': self.conv2_filters.tolist(),
            'conv1_bias' : self.conv1_bias.tolist(),
            'conv2_bias' : self.conv2_bias.tolist(),
            'fc1_weights': self.fc1_weights.tolist(),
            'fc2_weights': self.fc2_weights.tolist(),
            'fc1_biases': self.fc1_biases.tolist(),
            'fc2_biases': self.fc2_biases.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(weights_dict, f)
        print("Weights updated")
    
    def load_weights(self, filepath):
        if filepath:
            with open(filepath, 'r') as f:
                weights_dict = json.load(f)
                self.conv1_filters = np.array(weights_dict['conv1_filters'])
                self.conv2_filters = np.array(weights_dict['conv2_filters'])
                self.conv1_bias = np.array(weights_dict['conv1_bias'])
                self.conv2_bias = np.array(weights_dict['conv2_bias'])
                self.fc1_weights = np.array(weights_dict['fc1_weights'])
                self.fc2_weights = np.array(weights_dict['fc2_weights'])
                self.fc1_biases = np.array(weights_dict['fc1_biases'])
                self.fc2_biases = np.array(weights_dict['fc2_biases'])
        print("loaded weights from json")

    def check_shape(self, matrix):
        print(matrix.shape)
        sys.exit()

# ----------------------------------- Neural Network Helper Functions -----------------------------------
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, Z):
        return Z > 0

    def flatten(self, x):
        return x.reshape(x.shape[0], -1) if x.ndim > 2 else x.flatten()

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    # backprop functions
    def rotate_filters_180(self, filters):
        return np.flip(filters, axis=(1, 2))

    def one_hot(self, Y):
        return np.vstack((1 - Y, Y)).T

# ----------------------------------- Neural Network Functions -----------------------------------
    # works 100%, vectorized 
    def convolve(self, A_prev, filters, bias, padding=1, stride=1, print_data=True):
        if print_data:
            print(A_prev.shape)
            print(filters.shape)

        # get individual shapes for convolution
        batch_size, input_height, input_width, input_channels = A_prev.shape
        num_filters, filter_height, filter_width, filter_channels = filters.shape

        # verify shape
        if input_channels != filter_channels:
            raise ValueError("Input channels must match filter channels.")

        # for final output shape
        output_height = (input_height - filter_height + 2 * padding) // stride + 1
        output_width = (input_width - filter_width + 2 * padding) // stride + 1

        # pad input
        A_prev_pad = np.pad(A_prev, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')

        # vectorize convolution window
        shape = (batch_size, output_height, output_width, filter_height, filter_width, input_channels)
        strides = (
            A_prev_pad.strides[0],           # Batch dimension
            stride * A_prev_pad.strides[1],  # Stride in height
            stride * A_prev_pad.strides[2],  # Stride in width
            A_prev_pad.strides[1],           # Filter height
            A_prev_pad.strides[2],           # Filter width
            A_prev_pad.strides[3]            # Channels remain unchanged
        )
        patches = as_strided(A_prev_pad, shape=shape, strides=strides)

        # reshape it so it can be dot producted
        patches_reshaped = patches.reshape(batch_size * output_height * output_width, -1)
        W_reshaped = filters.reshape(num_filters, -1).T  # Reshape filters for matmul

        conv_out = np.dot(patches_reshaped, W_reshaped)
        conv_out = conv_out.reshape(batch_size, output_height, output_width, num_filters)

        # Adding bias, this is final value
        conv_out += bias.reshape(1, 1, 1, num_filters)

        # info for back propogation
        hparameters = {"pad": padding, "stride": stride}
        cache = (A_prev, filters, bias, hparameters)

        return conv_out, cache

    # works 100%, vectorized
    def pool(self, x, pool_size=2, stride=2):
        batch_size, height, width, channels = x.shape
        out_height = (height - pool_size) // stride + 1
        out_width = (width - pool_size) // stride + 1

        # creating batch window
        shape = (batch_size, out_height, out_width, pool_size, pool_size, channels)
        strides = (
            x.strides[0],          # batch dimension
            stride * x.strides[1], # vertical step
            stride * x.strides[2], # horizontal step
            x.strides[1],          # pooling region height
            x.strides[2],          # pooling region width
            x.strides[3]           # channels remain unchanged
        )
        x_reshaped = as_strided(x, shape=shape, strides=strides)


        pooled = x_reshaped.max(axis=(3, 4))   # Max over pooling region

        # cache for back propogation
        cache = (x, {"f": pool_size, "stride": stride})
        return pooled, cache

    # works 100%, vectorized
    def reversepool(self, dA, cache):
        # make sure argmax is stored as some value, error will occur if you remove this
        if len(cache) == 2:
            A_prev, hparameters = cache
            argmax = None  # No argmax stored
        else:
            A_prev, hparameters, argmax = cache

        # Necessary information for access
        f = hparameters["f"]
        stride = hparameters["stride"]
        m, _, _, n_C = A_prev.shape
        m, n_H, n_W, n_C = dA.shape

        # Output as zeros
        dA_prev = np.zeros_like(A_prev)
        
        # get all pooling windows
        shape = (m, n_H, n_W, f, f, n_C)
        strides = (A_prev.strides[0],
                stride * A_prev.strides[1],
                stride * A_prev.strides[2],
                A_prev.strides[1],
                A_prev.strides[2],
                A_prev.strides[3])
        windows = np.lib.stride_tricks.as_strided(A_prev, shape=shape, strides=strides)

        # reshaping windows where each pooling window is flattened
        windows_reshaped = windows.reshape(m, n_H, n_W, f * f, n_C)
        
        # compute argmax dynamically if it wasn't stored in cache
        if argmax is None:
            argmax = np.argmax(windows_reshaped, axis=3)  # shape: (m, n_H, n_W, n_C)

        # Convert flat indices to 2D indices (offsets within the window)
        offset_i = argmax // f  # row offsets
        offset_j = argmax % f   # column offsets
        
        # Create grid indices for the top-left corners of each pooling window
        grid_i = np.arange(n_H).reshape(1, n_H, 1, 1) * stride
        grid_j = np.arange(n_W).reshape(1, 1, n_W, 1) * stride
        
        # Compute the exact indices in A_prev where the max was taken
        row_indices = grid_i + offset_i  # shape: (m, n_H, n_W, n_C)
        col_indices = grid_j + offset_j  # shape: (m, n_H, n_W, n_C)
        
        # Build index arrays for the batch and channels
        m_idx = np.arange(m).reshape(m, 1, 1, 1)
        c_idx = np.arange(n_C).reshape(1, 1, 1, n_C)
        
        # Scatter the gradient dA to the corresponding positions in dA_prev
        np.add.at(dA_prev, (m_idx, row_indices, col_indices, c_idx), dA)
            
        
        assert dA_prev.shape == A_prev.shape
        return dA_prev

    # only used in backprop, vectorized
    def convolve_filter_grad(self, x, dz, kernel_size, stride=1, pad=1):
        N, _, _, C_in = x.shape
        N, H_out, W_out, _ = dz.shape

        # pad x
        x_padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')
        
        # this will create patches of shape (N, H_out, W_out, kernel_size, kernel_size, C_in)
        shape = (N, H_out, W_out, kernel_size, kernel_size, C_in)
        
        # strides for sliding window
        strides = (
            x_padded.strides[0],
            stride * x_padded.strides[1],
            stride * x_padded.strides[2],
            x_padded.strides[1],
            x_padded.strides[2],
            x_padded.strides[3]
        )
        
        x_cols = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
        # x_cols has shape: (N, H_out, W_out, kernel_size, kernel_size, C_in)
        
        # compute gradient
        dW = np.tensordot(x_cols, dz, axes=([0, 1, 2], [0, 1, 2]))
        # shape = (kernel_size, kernel_size, C_in, C_out)
        
        # transpose into correct shape
        dW = np.transpose(dW, (3, 0, 1, 2))
        # shape = (C_out, kernel_size, kernel_size, C_in)

        # Average over the batch and return
        dW /= (N * H_out * W_out)
        
        return dW

    #also for backprop, vectorized
    def conv2d_backward_data(self, dout, filters, stride=1, pad=1):
        N, H_out, W_out, C_out = dout.shape
        _, kH, kW, _ = filters.shape

        # get dimensions for stride
        H_in = (H_out - 1) * stride + kH - 2 * pad
        W_in = (W_out - 1) * stride + kW - 2 * pad

        # pad dout
        dout_padded = np.pad(dout, ((0,0), (pad, pad), (pad, pad), (0,0)), mode='constant', constant_values=0)

        # get patches from dout_padded
        # shape = (N, H_in, W_in, kH, kW, C_out)
        shape = (N, H_in, W_in, kH, kW, C_out)
        strides = (
            dout_padded.strides[0],
            stride * dout_padded.strides[1],
            stride * dout_padded.strides[2],
            dout_padded.strides[1],
            dout_padded.strides[2],
            dout_padded.strides[3]
        )
        patches = np.lib.stride_tricks.as_strided(dout_padded, shape=shape, strides=strides)

        dX = np.tensordot(patches, filters, axes=([3, 4, 5], [1, 2, 0]))
        # dX now has shape (N, H_in, W_in, C_in)

        return dX

# ----------------------------------- Neural network passes -----------------------------------
    # Forward pass through the network
    def forward(self, x, print_shapes = False, super_debug = False):
        # Layer1
        conv1, cache = self.convolve(x, self.conv1_filters, self.conv1_bias, print_data = print_shapes)
        relu1 = self.relu(conv1)
        pooled1, poolcache = self.pool(relu1)

        # Layer2
        conv2, cache2 = self.convolve(pooled1, self.conv2_filters, self.conv2_bias, print_data=print_shapes)
        relu2 = self.relu(conv2)
        pooled2, poolcache2 = self.pool(relu2)

        # Flatten to 1d
        flat = self.flatten(pooled2)

        # FFNN layer3
        fc1 = np.dot(flat, self.fc1_weights) + self.fc1_biases
        relu3 = self.relu(fc1)
        
        # FFNN layer4
        fc2 = np.dot(relu3, self.fc2_weights) + self.fc2_biases

        # output
        probabilities = self.softmax(fc2)

        if print_shapes:
            # layer1
            print(f"Convolved shape = conv1 = {conv1.shape}")
            print(f"Pooled shape = pooled1 = {pooled1.shape}")
            print()
            # layer2
            print(f"Convolved shape = conv2 = {conv2.shape}")
            print(f"Pooled shape = pooled2 = {pooled2.shape}")
            print()
            # flatten
            print(f"Flattened shape = {flat.shape}")
            print()
            # layer 3
            print(f"fc1 shape = {fc1.shape}, relu3 shape = {relu3.shape}")
            # layer 4
            print(f"fc2 shape = {fc2.shape}")
            # output
            print(f"probabilities shape = {probabilities.shape}")

        if super_debug:
            print("conv1 pre-activation stats: min =", conv1.min(), "max =", conv1.max(), "mean =", conv1.mean())
            print("conv2 stats: min =", conv2.min(), "max =", conv2.max(), "mean =", conv2.mean())
            print("relu2 stats: min =", relu2.min(), "max =", relu2.max(), "mean =", relu2.mean())
            print("pooled2 stats: min =", pooled2.min(), "max =", pooled2.max(), "mean =", pooled2.mean())
            print("flat stats: min =", flat.min(), "max =", flat.max(), "mean =", flat.mean())
            print("pooled1 stats: min =", pooled1.min(), "max =", pooled1.max(), "mean =", pooled1.mean())
            #print("conv2_filters:", self.conv2_filters)
            print("conv1_bias:", self.conv1_bias)
            print("conv2_bias:", self.conv2_bias)



        return conv1, relu1, pooled1, poolcache, conv2, relu2, pooled2, poolcache2, flat, fc1, relu3, probabilities

    # Backwards pass through the network
    def backpropogation(self, x_train, y_train, learning_rate = 0.02):
        # x_train is tuple of (n pictures, nxn, 1), y train is the answer as a 1d np array
        # [1,0] or 0 for cat | [0,1] or 1 for dog

        # Forward pass
        conv1, relu1, pooled1, poolcache, conv2, relu2, pooled2, poolcache2, flat, fc1, relu3, predictions = self.forward(x_train, print_shapes = False)
        # fc1 = preactivation layer 1 || ? = relu(Z1) || Z2 = preactivation layer 2 || predictions is softmaxed probability

        m = x_train.shape[0]
        # ---------- 4th layer ----------
        dz4 = (predictions - self.one_hot(y_train))     # (N, 2)
        # Gradients for W2 and b2
        dw2 = (1/m) * (np.dot(relu3.T, dz4))            # (64, 2)
        db2 = (1/m) * (np.sum(dz4, axis = 0))           # (2,)

        # ---------- 3rd layer ----------
        dz3 = np.dot(dz4, self.fc2_weights.T) * self.relu_derivative(fc1) # (N, 64)
        # Gradients for W1 and b1
        dw1 = (1/m) * (np.dot(flat.T, dz3))             # (2048, 64)
        db1 = (1/m) * (np.sum(dz3, axis = 0))           # (64,)


        # Loss with respect to flatten
        df2 = np.dot(dz3, self.fc1_weights.T)           # (N, 64) * (2048, 64).T = (N, 2048)


        # ---------- Layer 2 ----------
        # Loss with respect to pool2
        dp2 = df2.reshape(pooled2.shape)                # (N, 16, 16, 8)
        # Unpooling
        dc2 = self.reversepool(dp2, poolcache2)              # (N, 32, 32, 8)
        # 2nd layer loss
        dz2 = dc2 * self.relu_derivative(conv2)         # (N, 32, 32, 8)
        # Gradients for Conv2 filter, Conv2 Bias        # (8, 3, 3, 4) 

        dwf2 = self.convolve_filter_grad(pooled1, dz2, kernel_size = self.conv2_filters.shape[1])
        dbf2 = np.sum(dz2, axis=(0,1,2))                # (8,)


        # ---------- Layer 1 ----------
        # Loss with respect to pooled1                  # (N, 32, 32, 4)
        dp1 = self.conv2d_backward_data(dz2, self.rotate_filters_180(self.conv2_filters))

        # Unpooling
        dc1 = self.reversepool(dp1, poolcache)              # (N, 64, 64, 4)

        # Loss with respect to input filter
        dz1 = dc1 * self.relu_derivative(conv1)         # (N, 64, 64, 4)

        # Gradients for Conv1 filter, Conv1 Bias        # (4, 3, 3, 1)
        dwf1 = self.convolve_filter_grad(x_train, dz1, kernel_size = self.conv1_filters.shape[1])
        dbf1 = np.sum(dz1, axis=(0,1,2))                # (4,)


        # Update all weights
        self.fc2_weights -= learning_rate * dw2
        self.fc2_biases -= learning_rate * db2

        self.fc1_weights -= learning_rate * dw1
        self.fc1_biases -= learning_rate * db1

        self.conv1_filters -= learning_rate * dwf1
        self.conv1_bias -= learning_rate * dbf1

        self.conv2_filters -= learning_rate * dwf2
        self.conv2_bias -= learning_rate * dbf2


# ----------------------------------- Image data processing | only use for torch -----------------------------------
    def load_image_data(self, batch_size = 64, epoch = 10):
        from tqdm import tqdm
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        train_dataset = Loader(root_dir = r'D:\CNN\GSDogsAndCats', transform = transform)

        # Create a DataLoader from the dataset.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        

        batch_counter = 0
        for i in range(epoch):
            #gradient descent time
            for images, labels in tqdm(train_loader, desc=f"Epoch {i+1}"):

                # convert torch tensor to numpy tensor
                images = images.permute(0, 2, 3, 1).numpy() # Size of (N, 64, 64 1)
                labels = labels.numpy()                     # Size of (N)

                # start backpropagation
                self.backpropogation(images, labels)

            self.save_weights("cnn_weights.json")
            batch_counter += 1
            print(f"Processed batch {batch_counter}")

            # Test accuracy every 5 epoch
            if ((i+1) % 2) == 0:
                accuracy = self.accuracy_check()
                print(f"Test Accuracy: {accuracy * 100:.2f}%")

    def accuracy_check(self, batch_size=10):
        eval_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        eval_transform = transforms.ToTensor()  # Converts image to tensor with shape (1, 64, 64)
        eval_dataset = Loader(root_dir=r'D:\CNN\Eval', transform=eval_transform)
        
        # Create a DataLoader to load evaluation data in batches.
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        
        total_correct = 0
        total_samples = 0
        
        # Iterate over evaluation batches.
        for images, labels in eval_loader:
            # images comes as a tensor with shape: (batch, channels, height, width) = (B, 1, 64, 64)
            # Convert to numpy array with shape (B, 64, 64, 1)
            images = images.permute(0, 2, 3, 1).numpy()

            *_, predictions = self.forward(images, print_shapes=False)
            
            # predictions assumed to have shape (B, num_classes). Get predicted class labels.
            predicted_labels = np.argmax(predictions, axis=1)
            
            # Compare with ground truth
            total_correct += np.sum(predicted_labels == labels.numpy())
            total_samples += labels.size(0)
        
        return total_correct / total_samples




if __name__ == '__main__':
    size = 64
    # Create network
    # 64x64 grayscale images
    cnn = CNN()

    #this starts training, set false if testing something
    use_image = True

    if use_image:
        cnn.load_image_data()

    else:
        test_bp = True

        amount_images = 10
        dummy_input = np.random.randn(amount_images, size,size, 1) #test 20 images of 64x64
        #dummy_input = np.ones((4, 64, 64, 1), dtype=np.float32)
        # Forward pass with a dummy input 
        if test_bp:
            x_train = dummy_input
            y_train = np.random.choice([0, 1], size=amount_images)

            cnn.backpropogation(dummy_input, y_train)
        else:
            conv1, relu1, pooled1, poolcache, conv2, relu2, pooled2, poolcache2, flat, fc1, relu3, probabilities = cnn.forward(dummy_input, print_shapes = False)
            print("conv1_filters std:", np.std(cnn.conv1_filters))



