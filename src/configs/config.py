class Config:
    def __init__(self):
        # Caminhos base
        self.datalake_path = "/app/datalake"
        self.bronze_path = f"{self.datalake_path}/bronze"
        self.silver_path = f"{self.datalake_path}/silver"
        self.gold_path = f"{self.datalake_path}/gold"
        self.models_path = "/app/models"

        # Caminhos Silver
        self.silver_path_treino = f"{self.silver_path}/treino"
        self.silver_path_itens = f"{self.silver_path}/itens"
        self.silver_path_treino_normalized = f"{self.silver_path}/treino_normalized"
        self.silver_path_itens_normalized = f"{self.silver_path}/itens_normalized"
        self.silver_path_itens_embeddings = f"{self.silver_path}/itens_embeddings"

        # Caminhos Gold
        self.gold_path_lightfm_interactions = f"{self.gold_path}/lightfm_interactions"
        self.gold_path_lightfm_user_features = f"{self.gold_path}/lightfm_user_features"
        self.gold_path_lightfm_item_features = f"{self.gold_path}/lightfm_item_features"

        # Arquivo de download
        self.output_file = f"{self.bronze_path}/challenge-webmedia-e-globo-2023.zip"
        self.download_url = "https://drive.google.com/uc?id=13rvnyK5PJADJQgYe-VbdXb7PpLPj7lPr"
