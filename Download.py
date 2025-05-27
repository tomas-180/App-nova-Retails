import requests
import os

def download_file(url, destination):
    print(f"Downloading {destination}...")
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download {destination}: HTTP {response.status_code}")

    content_type = response.headers.get('Content-Type', '')
    if 'text/html' in content_type or response.content.startswith(b'<!DOCTYPE html>'):
        raise Exception(f"Received HTML instead of a binary file for {destination}")

    # Assegura que a pasta destino existe
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded {destination} ({os.path.getsize(destination)} bytes)")

files = {
    "df_historico.pkl": "https://huggingface.co/datasets/tomas180/df_historico/resolve/main/df_historico.pkl",
    "model_A_random_forest_tuned.joblib": "https://huggingface.co/tomas180/ModelArf/resolve/main/model_A_random_forest_tuned.joblib?download=true",
    "model_B_random_forest_tuned.joblib": "https://huggingface.co/tomas180/ModelArf/resolve/main/model_B_random_forest_tuned.joblib?download=true"
}

for filename, url in files.items():
    try:
        download_file(url, f"./models/{filename}")
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")

