import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. RASTGELELİK ÇEKİRDEĞİ (SEED) AYARI ---
# Okul numaranın son iki hanesi: 11
SEED = 11

def set_seed(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# --- 2. VERİ SETİNİN OLUŞTURULMASI ---
# Giriş: -2 <= p <= 2
# Hedef fonksiyon: y = 1 + sin(pi * p / 4)
set_seed()
p_train = np.linspace(-2, 2, 200).reshape(-1, 1)
y_train = 1 + np.sin(np.pi * p_train / 4)

# Test / Değerlendirme için daha sık örnekleme
p_test = np.linspace(-2, 2, 400).reshape(-1, 1)
y_test = 1 + np.sin(np.pi * p_test / 4)

# --- 3. MODEL EĞİTİM YARDIMCI FONKSİYONU ---
def build_and_train_model(
    activation="relu",
    learning_rate=0.1,
    optimizer_name="sgd",
    init_distribution="normal",
    hidden_layers_count=1,
    neurons_per_layer=2,
    epochs=300
):
    set_seed()
    
    # Ağırlık başlangıç dağılımı (Initializer)
    if init_distribution == "normal":
        initializer = keras.initializers.RandomNormal(seed=SEED)
    else:
        initializer = keras.initializers.RandomUniform(seed=SEED)

    # Model Mimarisi
    model = keras.Sequential()
    model.add(layers.Input(shape=(1,)))
    
    for _ in range(hidden_layers_count):
        if activation.lower() == "leakyrelu":
            model.add(layers.Dense(neurons_per_layer, kernel_initializer=initializer))
            model.add(layers.LeakyReLU(alpha=0.2))
        elif activation.lower() == "silu":
            model.add(layers.Dense(neurons_per_layer, activation="swish", kernel_initializer=initializer))
        elif activation.lower() == "lineer":
            model.add(layers.Dense(neurons_per_layer, activation="linear", kernel_initializer=initializer))
        elif activation.lower() == "hiperbolik tanjant":
            model.add(layers.Dense(neurons_per_layer, activation="tanh", kernel_initializer=initializer))
        else:
            model.add(layers.Dense(neurons_per_layer, activation=activation.lower(), kernel_initializer=initializer))
            
    # Çıkış katmanı: 1 lineer nöron
    model.add(layers.Dense(1, activation="linear", kernel_initializer=initializer))

    # Optimizasyon algoritması
    if optimizer_name.lower() == "sgd":
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name.lower() == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name.lower() == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss="mse")

    # Süre ölçümü ve eğitim
    start_time = time.time()
    history = model.fit(p_train, y_train, epochs=epochs, verbose=0, batch_size=32)
    elapsed_time = time.time() - start_time

    # Tahmin ve Metrikler
    y_pred = model.predict(p_test, verbose=0)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "history": history,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "time": elapsed_time,
        "y_pred": y_pred
    }

# --- 4. DENEYLERİN ÇALIŞTIRILMASI ---
# Varsayılan temel parametreler:
# LR: 0.1, Optimizer: SGD, Init: normal, Katman: 1, Nöron: 2

results = {}

# 1. Deney: Aktivasyon Fonksiyonları
print("\n--- TABLO 1: AKTİVASYON FONKSİYONLARI ---")
activations = ["Lineer", "Sigmoid", "Hiperbolik Tanjant", "ReLU", "LeakyReLU", "ELU", "SiLU"]
results["activation"] = {}
for act in activations:
    res = build_and_train_model(activation=act, learning_rate=0.1, optimizer_name="sgd",
                                init_distribution="normal", hidden_layers_count=1, neurons_per_layer=2)
    results["activation"][act] = res
    print(f"{act:<20} | MSE: {res['mse']:.5f} | RMSE: {res['rmse']:.5f} | R2: {res['r2']:.4f} | Süre: {res['time']:.2f}s")

# 2. Deney: Öğrenme Oranı (ReLU ile)
print("\n--- TABLO 2: ÖĞRENME ORANI ---")
lrs = [0.1, 0.01, 0.001]
results["lr"] = {}
for lr in lrs:
    res = build_and_train_model(activation="relu", learning_rate=lr, optimizer_name="sgd",
                                init_distribution="normal", hidden_layers_count=1, neurons_per_layer=2)
    results["lr"][lr] = res
    print(f"LR {lr:<17} | MSE: {res['mse']:.5f} | RMSE: {res['rmse']:.5f} | R2: {res['r2']:.4f} | Süre: {res['time']:.2f}s")

# 3. Deney: Optimizasyon Yöntemleri
print("\n--- TABLO 3: OPTİMİZASYON YÖNTEMLERİ ---")
opts = ["SGD", "RMSProp", "ADAM"]
results["optimizer"] = {}
for opt in opts:
    res = build_and_train_model(activation="relu", learning_rate=0.01, optimizer_name=opt,
                                init_distribution="normal", hidden_layers_count=1, neurons_per_layer=2)
    results["optimizer"][opt] = res
    print(f"{opt:<20} | MSE: {res['mse']:.5f} | RMSE: {res['rmse']:.5f} | R2: {res['r2']:.4f} | Süre: {res['time']:.2f}s")

# 4. Deney: Başlangıç Ağırlık Dağılımları
print("\n--- TABLO 4: BAŞLANGIÇ DAĞILIMI ---")
inits = ["normal", "uniform"]
results["init"] = {}
for ini in inits:
    res = build_and_train_model(activation="relu", learning_rate=0.01, optimizer_name="adam",
                                init_distribution=ini, hidden_layers_count=1, neurons_per_layer=2)
    results["init"][ini] = res
    print(f"{ini:<20} | MSE: {res['mse']:.5f} | RMSE: {res['rmse']:.5f} | R2: {res['r2']:.4f} | Süre: {res['time']:.2f}s")

# 5. Deney: Gizli Katman Sayısı
print("\n--- TABLO 5: GİZLİ KATMAN SAYISI ---")
layers_counts = [1, 2, 5, 10, 20]
results["layers"] = {}
for lc in layers_counts:
    res = build_and_train_model(activation="relu", learning_rate=0.01, optimizer_name="adam",
                                init_distribution="normal", hidden_layers_count=lc, neurons_per_layer=10)
    results["layers"][lc] = res
    print(f"Katman {lc:<13} | MSE: {res['mse']:.5f} | RMSE: {res['rmse']:.5f} | R2: {res['r2']:.4f} | Süre: {res['time']:.2f}s")

# 6. Deney: Nöron Sayısı
print("\n--- TABLO 6: NÖRON SAYISI ---")
neurons_counts = [2, 5, 10, 20, 50]
results["neurons"] = {}
for nc in neurons_counts:
    res = build_and_train_model(activation="relu", learning_rate=0.01, optimizer_name="adam",
                                init_distribution="normal", hidden_layers_count=2, neurons_per_layer=nc)
    results["neurons"][nc] = res
    print(f"Nöron {nc:<14} | MSE: {res['mse']:.5f} | RMSE: {res['rmse']:.5f} | R2: {res['r2']:.4f} | Süre: {res['time']:.2f}s")

# --- 5. EN İYİ MODELİN BELİRLENMESİ VE GRAFİKLERİN ÇİZİLMESİ ---
all_experiments = []
for group in results.values():
    for name, data in group.items():
        all_experiments.append(data)

# R2 skoru en yüksek ve MSE en düşük olan en iyi modeli seç
best_exp = max(all_experiments, key=lambda x: x["r2"])
print(f"\nEn Başarılı Model Metrikleri -> MSE: {best_exp['mse']:.5f}, R2: {best_exp['r2']:.4f}")

# Grafikler
plt.figure(figsize=(15, 5))

# Grafik 1: Epoch / Kayıp (Doğrusal ve Logaritmik)
plt.subplot(1, 3, 1)
loss_history = best_exp["history"].history["loss"]
plt.plot(loss_history, color="firebrick", label="MSE Kayıp")
plt.title("Epoch / Kayıp (Doğrusal Eksen)")
plt.xlabel("Epoch")
plt.ylabel("Kayıp (MSE)")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 3, 2)
plt.semilogy(loss_history, color="firebrick", label="MSE Kayıp (Log)")
plt.title("Epoch / Kayıp (Logaritmik Eksen)")
plt.xlabel("Epoch")
plt.ylabel("log(MSE)")
plt.grid(True, linestyle="--", alpha=0.6)

# Grafik 2: Hedef - Çıkış Grafiği
plt.subplot(1, 3, 3)
plt.plot(p_test, y_test, label="Hedef (Gerçek)", color="navy", linewidth=2)
plt.plot(p_test, best_exp["y_pred"], label="Ağ Çıkışı (Tahmin)", color="crimson", linestyle="--", linewidth=2)
plt.title(f"Hedef - Ağ Çıkışı (R²: {best_exp['r2']:.4f})")
plt.xlabel("Giriş (p)")
plt.ylabel("Çıkış (y)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("en_iyi_model_grafigi.png", dpi=300)
plt.show()
