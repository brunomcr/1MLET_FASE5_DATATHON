import gdown

class Downloader:
    def download_file(self, url: str, output_file: str):
        print(f"Downloading file from {url} to {output_file}...")
        gdown.download(url, output_file, quiet=False)
        print("Download completed!")
