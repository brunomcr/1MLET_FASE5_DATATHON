class Config:
    def __init__(self):

        self.bronze_path = "/datalake/bronze"
        self.silver_path_treino = "/datalake/silver/treino/"
        self.silver_path_itens = "/datalake/silver/itens/"

        self.file_id = "13rvnyK5PJADJQgYe-VbdXb7PpLPj7lPr"
        self.download_url = f"https://drive.google.com/uc?id={self.file_id}"
        self.output_file = f"{self.bronze_path}/challenge-webmedia-e-globo-2023.zip"
