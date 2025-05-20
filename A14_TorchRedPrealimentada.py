#====================================
# Red Prealimentada (feed forward)
#====================================

#====================================
# Módulos Necesarios
#====================================
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#====================================
# Configuración del GPU
#====================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#====================================
# Hiper-parámetros
#====================================
input_size  = 784   # imagen 28x28
hidden_size = 500   # neuronas ocultas
num_classes = 10    # clasificaciones
num_epochs  = 2     # iteraciones sobre los datos
batch_size  = 100   # tamaño de conjuntos de datos
learning_rate = 0.001 # tasa de aprendisaje (para que vaya con calma)

#====================================
# MNIST base de Datos
#====================================
train_dataset = torchvision.datasets.MNIST(root='./Datos/data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./Datos/data',
                                          train=False,
                                          transform=transforms.ToTensor())

#====================================
# Carga de Datos
#====================================
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

examples = iter(test_loader)                    # iterable
example_data, example_targets = next(examples)  # siguiente elemento

#====================================
# Mostrar Datos en una Imagen
#====================================
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()

#============================================================
# Red Neuronal Completamente Conectada con una Capa Oculta
#============================================================
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # sin activation y sin softmax al final
        # porque la aplica crossentropyloss
        return out
    
#====================================
# Correr Modelo en el GPU
#====================================
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

#====================================
# Optimización y cálculo de error
#====================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#====================================
# entrenar el modelo
#====================================
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # dimensiones originales: [100, 1, 28, 28]
        # nuevas dimensiones: [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # evaluación
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Cálculo del Gradiente y Optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # diagnóstico
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')