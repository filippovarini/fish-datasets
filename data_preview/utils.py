import requests
import zipfile


def download_file(url, save_path):
    print(f"Downloading {url} to {save_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Download complete: {save_path}")


def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete: {zip_path}")
    
    
def download_and_extract_zip(data_dir, data_url, dataset_shortname):
    data_path = data_dir / f"{dataset_shortname}.zip"
    if data_dir.exists() and len(list(data_dir.glob("*"))) > 0:
        print("Data already downloaded and extracted")
    else:
        print("Downloading data...")
        download_file(data_url, data_path)
        print("Extracting data...")
        extract_zip(data_path, data_dir)
        print("Removing Zipped files...")
        data_path.unlink()
    return data_dir
