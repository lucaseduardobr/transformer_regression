# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 15:20:17 2025

@author: lucas
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Dados sintéticos
np.random.seed(42)
X = np.random.rand(1000, 10)  # 10 parâmetros como entrada
y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 1000)  # Saída é soma com ruído (problema simples de regressão)


# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Transformer Block simplificado para regressão
def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
    print(attn_output.shape)
    
    x = Dropout(dropout)(attn_output) + inputs
    x = LayerNormalization(epsilon=1e-6)(x)
    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    return Dropout(dropout)(x_ff) + x

# Criar modelo simples
inputs = Input(shape=(X_train.shape[1],))
x = Dense(32, activation='relu')(inputs)
x = tf.expand_dims(x, axis=1)  # Adiciona uma dimensão de sequência fictícia
x = transformer_block(x, head_size=8, num_heads=2, ff_dim=32)
x = tf.squeeze(x, axis=1)  # Remove a dimensão fictícia
outputs = Dense(1)(x)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Treinar modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Avaliação
loss, mae = model.evaluate(X_test, y_test)
print(f"Loss: {loss:.4f}, MAE: {mae:.4f}")

# Visualizar resultados
preds = model.predict(X_test)

plt.scatter(y_test, preds, c=y_test, cmap='viridis', alpha=0.7)
plt.xlabel("Valores reais")
plt.ylabel("Previsões")
plt.title("Comparação valores reais vs previsões com Transformer")
plt.colorbar(label="Valor real")
plt.grid(True)
plt.show()


