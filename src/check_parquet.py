import polars as pl
from pathlib import Path

# Lista de los 13 activos fallidos
assets = [
    "BMTUSDT", "CARVUSDT", "COTIUSDT", "CUSDT", "DENTUSDT",
    "MDTUSDT", "PHBUSDT", "PORT3USDT", "PROMUSDT", "SQDUSDT",
    "SXTUSDT", "VIDTUSDT", "XNYUSDT"
]

# Directorio de datos (ajusta si es necesario)
data_dir = Path("C:/Users/kterz/PycharmProjects/Correlation(EDA)/data")

for asset in assets:
    try:
        file_path = data_dir / asset / f"{asset}.parquet"
        if not file_path.exists():
            print(f"⚠️ {asset}: archivo no existe")
            continue
        df = pl.read_parquet(file_path)
        print(f"📋 Columnas en {asset}: {df.columns}")
        print(f"Primeras 5 filas de {asset}:\n{df.head(5)}\n")
    except Exception as e:
        print(f"❌ Error al leer {asset}: {str(e)[:100]}")