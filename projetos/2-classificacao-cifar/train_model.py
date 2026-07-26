import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import h5py
import json

# ---------------------------------------------------------------------------
# Projeto 2 — Classificação CIFAR-10
# ---------------------------------------------------------------------------

# 1. Carregando o dataset CIFAR-10
print("Carregando o dataset CIFAR-10...")
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

# 2. Normalizando as imagens para a faixa [0, 1]
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Divisão de treino / Validação (uso de 20% do treino para validação)
val_split = 0.2
val_size = int(len(x_train_full) * val_split)

x_val = x_train_full[:val_size]
y_val = y_train_full[:val_size]

x_train = x_train_full[val_size:]
y_train = y_train_full[val_size:]

print(f"Dados de Treino: {x_train.shape[0]} amostras")
print(f"Dados de Validação: {x_val.shape[0]} amostras")
print(f"Dados de Teste: {x_test.shape[0]} amostras")

# 4. Definição de Data Augmentation usando camadas do Keras
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# 5. Construção da CNN
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),

    # Camada de Data Augmentation acoplada no modelo
    data_augmentation,

    # Bloco Convolucional 1
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    # Bloco Convolucional 2
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),

    # Bloco Convolucional 3
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

# Compilando o Modelo
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. Configurando o Early Stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# 7. Treinamento do Modelo
EPOCHS = 15
BATCH_SIZE = 64

print("\nIniciando treinamento na CPU...")

history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping]
)

# 8. Exibição da Acurácia de Validação Final 
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
print(f"\nAcurácia de Validação Final: {val_acc:.4f} ({val_acc * 100:.2f}%)")

# 9. Salvamento do modelo original
print("Salvando o modelo treinado em 'model.h5'...")
model.save("model.h5")

# Ajuste de compatibilidade de desserialização Keras 3 -> Keras 2 / tf.keras no GitHub Actions
try:
    with h5py.File("model.h5", "r+") as f:
        if "model_config" in f.attrs:
            cfg_str = f.attrs["model_config"]
            if isinstance(cfg_str, bytes):
                cfg_str = cfg_str.decode("utf-8")
            cfg_dict = json.loads(cfg_str)
            
            def clean_dict(d):
                if isinstance(d, dict):
                    # Remove chaves do Keras 3 incompatíveis com tf.keras do GitHub Actions
                    d.pop("input_axes", None)
                    d.pop("output_axes", None)
                    d.pop("synchronized", None)
                    d.pop("registered_name", None)
                    d.pop("renorm", None)
                    d.pop("renorm_clipping", None)
                    d.pop("renorm_momentum", None)
                    for v in d.values():
                        clean_dict(v)
                elif isinstance(d, list):
                    for item in d:
                        clean_dict(item)

            clean_dict(cfg_dict)
            new_str = json.dumps(cfg_dict)
            f.attrs["model_config"] = new_str.encode("utf-8") if isinstance(f.attrs["model_config"], bytes) else new_str
    print("Compatibilidade HDF5 ajustada com sucesso!")
except Exception as e:
    print(f"Aviso ao ajustar HDF5: {e}")

print("Modelo salvo com sucesso!")