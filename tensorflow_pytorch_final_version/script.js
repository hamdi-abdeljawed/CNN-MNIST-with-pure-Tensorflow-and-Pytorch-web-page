// Initialize everything when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Prism.js
    Prism.highlightAll();

    // Initialize code tabs
    const defaultTab = document.querySelector('.tab-button');
    if (defaultTab) {
        defaultTab.click();
    }

    // Initialize Charts
    initializeCharts();

    // Initialize smooth scrolling
    initializeSmoothScroll();

    // Load code into containers
    const tensorflowCode = document.getElementById('tensorflow-code');
    const pytorchCode = document.getElementById('pytorch-code');

    // TensorFlow code content
    const tfCodeContent = `import tensorflow as tf
import numpy as np

# 1. Load and Prepare MNIST Data
def load_mnist():
    """Loads and preprocesses the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Reshape to (batch_size, height, width, channels)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    # Convert labels to one-hot encoding
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)
    
    return x_train, y_train, x_test, y_test

# 2. Define Model Parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 10
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10

# 3. Initialize Weights and Biases
def initialize_weights_and_biases():
    W_conv1 = tf.Variable(tf.random.normal([3, 3, 1, 32], stddev=0.1))
    b_conv1 = tf.Variable(tf.random.normal([32], stddev=0.1))
    
    W_conv2 = tf.Variable(tf.random.normal([3, 3, 32, 64], stddev=0.1))
    b_conv2 = tf.Variable(tf.random.normal([64], stddev=0.1))
    
    W_fc1 = tf.Variable(tf.random.normal([7 * 7 * 64, 128], stddev=0.1))
    b_fc1 = tf.Variable(tf.random.normal([128], stddev=0.1))
    
    W_fc2 = tf.Variable(tf.random.normal([128, NUM_CLASSES], stddev=0.1))
    b_fc2 = tf.Variable(tf.random.normal([NUM_CLASSES], stddev=0.1))
    
    return W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2

# 4. CNN Model Architecture
def cnn_model(x, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2):
    # First Convolutional Layer
    conv1 = tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding="SAME")
    conv1 = tf.nn.bias_add(conv1, b_conv1)
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Second Convolutional Layer
    conv2 = tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1], padding="SAME")
    conv2 = tf.nn.bias_add(conv2, b_conv2)
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Flatten and Fully Connected Layers
    flat_out = tf.reshape(pool2, [-1, 7 * 7 * 64])
    fc1 = tf.matmul(flat_out, W_fc1) + b_fc1
    fc1 = tf.nn.relu(fc1)
    out = tf.matmul(fc1, W_fc2) + b_fc2

    return out

# 5. Training Loop
def train(x_train, y_train, x_test, y_test, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2):
    num_batches = len(x_train) // BATCH_SIZE
    
    for epoch in range(EPOCHS):
        for batch in range(num_batches):
            batch_x = x_train[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
            batch_y = y_train[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
            
            with tf.GradientTape() as tape:
                logits = cnn_model(batch_x, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_y))
            
            # Apply gradients
            grads = tape.gradient(loss, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2])
            tf.optimizers.Adam(learning_rate=LEARNING_RATE).apply_gradients(
                zip(grads, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]))`;

    // PyTorch code content
    const ptCodeContent = `import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from torchsummary import summary

# 1. Setup & Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
SEED = 42
IMAGE_SIZE = 28

# Device and reproducibility
device = torch.device("cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

# 2. Data Loading & Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 3. CNN Model Definition
class CNN(nn.Module):
    def __init__(self, image_size):
        super(CNN, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # MaxPooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size
        self.fc_input_size = int((image_size/2/2) * (image_size/2/2) * 64)
        
        # Fully connected Layers
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 4. Model Training
model = CNN(IMAGE_SIZE).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        correct_test = 0
        total_test = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()`;

    if (tensorflowCode) {
        tensorflowCode.textContent = tfCodeContent;
        Prism.highlightElement(tensorflowCode);
    }

    if (pytorchCode) {
        pytorchCode.textContent = ptCodeContent;
        Prism.highlightElement(pytorchCode);
    }
});

// Code Tab Switching
function showTab(tabId) {
    // Hide all containers
    document.querySelectorAll('.code-container-wrapper').forEach(container => {
        container.classList.remove('active');
    });
    
    // Show selected container
    const selectedContainer = document.getElementById(tabId);
    if (selectedContainer) {
        selectedContainer.classList.add('active');
    }
    
    // Update button states
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
        if (button.textContent.toLowerCase().includes(tabId)) {
            button.classList.add('active');
        }
    });

    // Re-highlight code in the newly visible container
    Prism.highlightAll();
}

// Initialize Charts
function initializeCharts() {
    // Training Speed Chart
    const speedCtx = document.getElementById('speedChart');
    if (speedCtx) {
        new Chart(speedCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['TensorFlow', 'PyTorch'],
                datasets: [{
                    label: 'Time per Epoch (seconds)',
                    data: [12.5, 11.8],
                    backgroundColor: [
                        'rgba(255, 111, 0, 0.8)',
                        'rgba(238, 76, 44, 0.8)'
                    ],
                    borderColor: [
                        'rgba(255, 111, 0, 1)',
                        'rgba(238, 76, 44, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // Memory Usage Chart
    const memoryCtx = document.getElementById('memoryChart');
    if (memoryCtx) {
        new Chart(memoryCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: ['Start', 'Epoch 5', 'Epoch 10', 'Epoch 15', 'End'],
                datasets: [{
                    label: 'TensorFlow Memory (MB)',
                    data: [512, 768, 825, 842, 856],
                    borderColor: 'rgba(255, 111, 0, 1)',
                    fill: false
                }, {
                    label: 'PyTorch Memory (MB)',
                    data: [525, 785, 845, 868, 880],
                    borderColor: 'rgba(238, 76, 44, 1)',
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

// Initialize smooth scrolling
function initializeSmoothScroll() {
    document.querySelectorAll('nav a').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Initialize particles.js with enhanced programming-themed animation
particlesJS('particles-js', {
    particles: {
        number: {
            value: 100,
            density: {
                enable: true,
                value_area: 800
            }
        },
        color: {
            value: ['#ff6f00', '#ee4c2c'] // TensorFlow and PyTorch colors
        },
        shape: {
            type: ['circle', 'triangle'],
            stroke: {
                width: 0,
                color: '#000000'
            },
            polygon: {
                nb_sides: 5
            }
        },
        opacity: {
            value: 0.6,
            random: true,
            anim: {
                enable: true,
                speed: 1,
                opacity_min: 0.1,
                sync: false
            }
        },
        size: {
            value: 3,
            random: true,
            anim: {
                enable: true,
                speed: 2,
                size_min: 0.1,
                sync: false
            }
        },
        line_linked: {
            enable: true,
            distance: 150,
            color: '#808080',
            opacity: 0.4,
            width: 1
        },
        move: {
            enable: true,
            speed: 3,
            direction: 'none',
            random: false,
            straight: false,
            out_mode: 'out',
            bounce: false,
            attract: {
                enable: true,
                rotateX: 600,
                rotateY: 1200
            }
        }
    },
    interactivity: {
        detect_on: 'canvas',
        events: {
            onhover: {
                enable: true,
                mode: ['grab', 'bubble']
            },
            onclick: {
                enable: true,
                mode: 'push'
            },
            resize: true
        },
        modes: {
            grab: {
                distance: 140,
                line_linked: {
                    opacity: 1
                }
            },
            bubble: {
                distance: 200,
                size: 6,
                duration: 2,
                opacity: 0.8,
                speed: 3
            },
            push: {
                particles_nb: 4
            }
        }
    },
    retina_detect: true
});
