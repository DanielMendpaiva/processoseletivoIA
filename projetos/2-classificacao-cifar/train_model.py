import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ---------------------------------------------------------------------------
# Projeto 2 — Classificação CIFAR-10
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset CIFAR-10 via tf.keras.datasets.cifar10
#   2. Normalizar as imagens para [0, 1] (shape (32, 32, 3))
#   3. Separar um conjunto de validação
#   4. Incluir data augmentation (ex: layers.RandomFlip, RandomRotation, RandomZoom)
#      aplicada ao conjunto de treino
#   5. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   6. Treinar com EarlyStopping monitorando a perda de validação
#   7. Exibir a acurácia de validação final no terminal
#   8. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui

#1. Carregando o dataset CIFAR-10
print("Carregando o dataset CIFAR-10...")
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

#2. Normalizando as imagens para a faixa [0, 1]
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#3. Divisão de treino / Validação (uso de 20% do treino para validação)
val_split = 0.2
val_size = int(len(x_train_full) * val_split)

x_val = x_train_full[:val_size]
y_val = y_train_full[:val_size]

x_train = x_train_full[val_size:]
y_train = y_train_full[val_size:]

print(f"Dados de Treino: {x_train.shape[0]} amostras")
print(f"Dados de Validação: {x_val.shape[0]} amostras")
print(f"Dados de Teste: {x_test.shape[0]} amostras")

#4. Definição de Data Augmentation usando camadas do Keras
#Ajuda a evitar Overfitting em imagens coloridas complexas

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

#5. Construção da CNN
#CNN com 3 blocos convolucionais, Conv2D + BatchNormalization + MaxPooling2D + Dropout

model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),

    #Camada de Data Augmentation acoplada no modelo
    data_augmentation,

    #Bloco Convolucional 1
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    #Bloco Convolucional 2
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),

    #Bloco Convolucional 3
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.4),

    # Classificador Final (Densa)
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

#Compilando o Modelo

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

model.summary()

#6. Configurando o Early Stopping baseado na perda de validação(val_loss)
#Se a perda na validação parar de melhorar por 5 épocas, o treino para -
# e recupera os melhores pesos 

early_stopping = keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = 5,
    restore_best_weights= True
)

#7. Treinamento do Modelo
#Usando 15 épocas, considerando o limite de CPU
EPOCHS = 15
BATCH_SIZE = 64

print("\nIniciando treinamento na CPU")

history = model.fit(
    x_train, y_train,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    validation_data = (x_val, y_val),
    callbacks = [early_stopping]
)

#8. Exibição da Acurácia de Validação Final 

val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
print(f"\nAcurácia de Validação Final: {val_acc:.4}% ({val_acc * 100: .2f}%)")


#9. Salvamento do modelo original 
print("Salvando o modelo treinado em MODEL.H5")
model.save("model.h5")
print("Model salvo com sucesso!")