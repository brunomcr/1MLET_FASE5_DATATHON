class Config:
    def __init__(self):
        # Usar caminhos relativos ao container
        self.bronze_path = "/app/datalake/bronze"
        self.silver_path_treino = "/app/datalake/silver/treino"
        self.silver_path_itens = "/app/datalake/silver/itens"

        self.file_id = "13rvnyK5PJADJQgYe-VbdXb7PpLPj7lPr"
        self.download_url = f"https://drive.google.com/uc?id={self.file_id}"
        self.output_file = f"{self.bronze_path}/challenge-webmedia-e-globo-2023.zip"
