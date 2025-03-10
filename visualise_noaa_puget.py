# %% [markdown]
# ## Download the data
#
# This section downloads and extracts the dataset using Python libraries instead of shell commands.

# %%
dataset_shortname = "noaa_puget"
data_dir = Path("/mnt/data/tmp/") / dataset_shortname
data_dir.mkdir(exist_ok=True, parents=True)

# %%
import requests
import zipfile
import os

# URLs for the data
data_url = "https://storage.googleapis.com/public-datasets-lila/noaa-psnf/noaa_estuary_fish-images.zip"
annotations_url = "https://storage.googleapis.com/public-datasets-lila/noaa-psnf/noaa_estuary_fish-annotations-2023.08.19.zip"

data_path = data_dir / "images.zip"
annotations_path = data_dir / "annotations.zip"


# Download the data files
def download_file(url, save_path):
    print(f"Downloading {url} to {save_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Download complete: {save_path}")


# Extract zip files
def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete: {zip_path}")


# Download and extract data
download_file(data_url, data_path)
download_file(annotations_url, annotations_path)

extract_zip(data_path, data_dir)
extract_zip(annotations_path, data_dir)

# Remove the zip files after extraction
os.remove(data_path)
os.remove(annotations_path)
print("Zip files removed")
