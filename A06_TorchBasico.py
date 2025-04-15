#===================================
# Modulos necesariode pytorch
#===================================
import torch
#===================================
# En pytorch todo está basado en operaciones tensoriales
#===================================
# Un tensor vive en Rn x Rm x Ro x Rp ... etc
#===================================

#===================================
# Escalar vacío (trae basura)
#===================================
x = torch.empty(1) # scalar
print(x)
#===================================
# Vector en R3
#===================================
x = torch.empty(3)
print(x)
#===================================
# Escalar en R2XR3
#===================================
x = torch.empty(2,3)
print(x)
#===================================
# Escalar en R2xR2XR3
#===================================
x = torch.empty(2,2,3)
print(x)
#===================================
# torch.rand(size): números aleatorios [0,1]
#===================================
# Tensor de números aleatorios de R5xR3
#===================================
x = torch.rand(5,3)
print(x)
#===================================
# Checar tamaño (lista de dimensiones)
#===================================
print(x.size())
#===================================
# Checar tamaño (default es float32)
#===================================
print(x.dtype)
#===================================
# Específicando tipo de datos
#===================================
x = torch.zeros(5, 3, dtype=torch.float16)
print(x)
print(x.dtype)
#====================================
# Construir vector de datos
#====================================
x = torch.tensor([5.5, 3])
print(x.size())
#====================================
# Vector optimizable (variables del gradiente)
#====================================
x = torch.tensor([5.5, 3], requires_grad=True)
#====================================
# Suma de tensores
#====================================
y = torch.rand(2, 2)
x = torch.rand(2, 2)
z = x + y
z = torch.add(x,y)
print(z)
y.add_(x)
print(y)
#====================================
# Resta de tensores
#====================================
z = x - y
z = torch.sub(x, y)
print(z)
#====================================
# Multiplicación
#====================================
z = x * y
z = torch.mul(x, y)
print(z)
#====================================
# División
#====================================
z = x / y
z = torch.div(x, y)
print(z)
#====================================
# Rebanadas
#====================================
x = torch.rand(5, 3)
print(x)
print(x[:, 0])  # todos los renglones, columna 0
print(x[1, :])  # renglón 1, todas las columnas
print(x[1, 1])  # elemento (1, 1)
#====================================
# Valor del elemento en (1,1)
#====================================
print(x[1, 1].item())
#====================================
# Cambiar forma conb torch.view()
#====================================
x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)   # -1: se infiere de las otras dimensiones
# si -1 pytorch determinara automáticamente el tamaño necesario
print(x.size(), y.size(), z.size())
#====================================
# Convertir un tensor en arreglo de numpy y viceversa
#====================================
a = torch.ones(5)
b = a.numpy()
print(b)
print(type(b))
#====================================
# Le suma 1 a todas las entradas
#====================================
a.add_(1)
print(a)
print(b)
#====================================
# De numpy a torch
#====================================
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)
#====================================
# Le suma 1 a todas las entradas de a
#====================================
a += 1
print(a)
print(b)
#====================================
# De CPU a Gpu (si hay CUDA)
#====================================
if torch.cuda.is_available():
    device = torch.device("cuda")   # la tarjeta de video con CUDA
    print("Tengo GPU " + str(device))
    y_d = torch.ones_like(x, device=device) # crear tensor en el GPU
    x_d = x.to(device)                      # copiar a GPU o usar ''.to("cuda")''
    z_d = x_d + y_d
    #====================================
    # z = z_d.numpy() (numpy no maneja tensores en el GPU)
    #====================================
    # de vuelta al CPU
    #====================================
    x = z_d.to("cpu")
    z = z.numpy()
    print(z)