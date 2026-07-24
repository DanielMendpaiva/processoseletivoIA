import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 2 — Otimização do Modelo (CIFAR-10)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui

MODEL_H5_PATH = "model.h5"
MODEL_TFLITE_PATH = "model.tflite"

#1. Verifica se o modelo original já treinado existe
if not os.path.exists(MODEL_H5_PATH):
    raise FileNotFoundError(
        f"Arquivo '{MODEL_H5_PATH}' não encontrado" 
        "Execute 'python train_model.py'"
    )

print(f"Carregando o modelo original de '{MODEL_H5_PATH}'")
model = tf.keras.models.load_model(MODEL_H5_PATH)

#2. Criar o conversor para TensorFlow lite
print("Iniciando a conversão para TensorFlow Lite")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

#3. Aplicar a Otimização: Dynamic Range Quantization 
#Reduz o tamanho dos pesos de Float32 para Int8, reduzindo o arquivo

converter.optimizations = [tf.lite.Optimize.DEFAULT]

#4. Executa a conversão

tflite_model = converter.convert()

#5. Salvar o modelo otimizado em .tflite
print(f"Salvando modelo otimizado em '{MODEL_TFLITE_PATH}'")
with open(MODEL_TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

#6. Comparar o tamanhos dos arquivos (Demonstração)
h5_size = os.path.getsize(MODEL_H5_PATH) / (1024*1024)
tflite_size = os.path.getsize(MODEL_TFLITE_PATH) / (1024*1024)

print("\n Comparativo de Otimização para (Edge AI)")
print(f"Tamanho do modelo original (.h5): {h5_size:.2f} MB")
print(f"Tamanho do modelo otimizado (.tflite): {tflite_size:.2f} MB")
print(f"Redução de tamanho: {((h5_size - tflite_size) / h5_size) * 100:.1f}%")
print("Otimização concluída com sucesso!")