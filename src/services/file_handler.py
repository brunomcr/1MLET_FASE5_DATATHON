import zipfile
import os

class FileHandler:
    def unzip_and_delete(self, file_path: str, extract_to: str):

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"File unzipped to: {extract_to}")

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f".zip file deleted: {file_path}")
        else:
            print("The .zip file was not found for deletion.")

