import numpy as np
import matplotlib.pyplot as plt

class InputLayer:
    def __init__(self, size):
        self.size = size
        self.activations = np.zeros(size)
    
    def set_activation(self, indices):
        self.activations = np.zeros(self.size)
        self.activations[indices] = 1
class CA3:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
        self.state = np.zeros(size)
    
    def train(self, patterns):
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)
        self.weights /= self.size
    
    def recall(self, input_state, steps=5):
        self.state = input_state.copy()
        for _ in range(steps):
            self.state = np.sign(self.weights @ self.state)
            self.state[self.state == 0] = 1
        return self.state
class CA1:
    def __init__(self, size, output_size):
        self.size = size
        self.output_size = output_size
        # Basit rastgele ağırlıklar
        self.weights = np.random.randn(output_size, size)
    
    def forward(self, ca3_state):
        return self.weights @ ca3_state
class Hippocampus:
    def __init__(self, input_size, ca3_size, ca1_output_size):
        self.input_layer = InputLayer(input_size)
        self.ca3 = CA3(ca3_size)
        self.ca1 = CA1(ca3_size, ca1_output_size)
    
    def train_ca3(self, patterns):
        self.ca3.train(patterns)
    
    def recall(self, input_indices):
        self.input_layer.set_activation(input_indices)
        ca3_input = self.input_layer.activations
        ca3_output = self.ca3.recall(ca3_input)
        ca1_output = self.ca1.forward(ca3_output)
        return ca1_output
# Girdi boyutları
input_size = 100
ca3_size = 100
ca1_output_size = 50

# Hipokampüs modelini oluşturma
hippocampus = Hippocampus(input_size, ca3_size, ca1_output_size)

# Eğitim için örnek desenler oluşturma
pattern1 = np.zeros(input_size)
pattern1[10:20] = 1
pattern2 = np.zeros(input_size)
pattern2[40:50] = 1
patterns = [pattern1, pattern2]

# CA3'ü eğitme
hippocampus.train_ca3(patterns)

# Hatırlama testi
test_input = [12, 18]  # Pattern1'e benzer girdi
output = hippocampus.recall(test_input)

print("CA1 Çıkışı:", output)

# CA3 çıkışını görselleştirme
ca3_output = hippocampus.ca3.state
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Girdi Katmanı")
plt.imshow(hippocampus.input_layer.activations.reshape(10,10), cmap='gray')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("CA3 Çıkışı")
plt.imshow(ca3_output.reshape(10,10), cmap='gray')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("CA1 Çıkışı")
plt.imshow(output.reshape(5,10), cmap='gray')
plt.colorbar()

plt.show()
