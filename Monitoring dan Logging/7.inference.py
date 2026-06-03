import mlflow
import mlflow.pyfunc

# Tracking URI MLflow
mlflow.set_tracking_uri(
    "http://localhost:5000"
)

# Run ID model terbaik
RUN_ID = "63d7989023bb49fba4f58d9374aec766"

# Load model dari MLflow
model = mlflow.pyfunc.load_model(
    f"runs:/{RUN_ID}/model"
)

# Contoh data inference
samples = [
    # Positif
    "Aplikasi DANA sangat membantu, proses transaksi cepat dan fiturnya sangat mudah digunakan.",

    # Positif
    "Saya sangat puas menggunakan DANA, transfer dan pembayaran selalu lancar tanpa kendala.",

    # Negatif
    "Aplikasi DANA sering error, saldo saya terpotong tetapi transaksi gagal.",

    # Negatif
    "Customer service sangat lambat merespons dan masalah saya tidak kunjung selesai.",

    # Netral
    "Saya menggunakan aplikasi DANA untuk melakukan pembayaran tagihan setiap bulan.",

    # Netral
    "Aplikasi DANA menyediakan fitur transfer, pembayaran, dan pembelian pulsa."
]

# Prediksi
predictions = model.predict(samples)

# Tampilkan hasil
print("=" * 70)
print("HASIL INFERENCE MODEL SENTIMENT ANALYSIS DANA")
print("=" * 70)

for i, (text, pred) in enumerate(zip(samples, predictions), start=1):
    print(f"\nData #{i}")
    print(f"Text       : {text}")
    print(f"Prediction : {pred}")

print("\n" + "=" * 70)
print("Inference selesai")
print("=" * 70)