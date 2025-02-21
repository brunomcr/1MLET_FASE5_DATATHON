class Config:
    def __init__(self):
        # Caminhos base
        self.datalake_path = "/app/datalake"
        self.bronze_path = f"{self.datalake_path}/bronze"
        self.silver_path = f"{self.datalake_path}/silver"
        self.gold_path = f"{self.datalake_path}/gold"

        # Caminhos Silver
        self.silver_path_treino = f"{self.silver_path}/treino"
        self.silver_path_itens = f"{self.silver_path}/itens"
        self.silver_path_treino_normalized = f"{self.silver_path}/treino_normalized"
        self.silver_path_itens_normalized = f"{self.silver_path}/itens_normalized"
        self.silver_path_itens_embeddings = f"{self.silver_path}/itens_embeddings"
        self.silver_path_itens_tfidf = f"{self.silver_path}/itens/tfidf"

        # Caminhos Gold
        #self.gold_path_matrices = f"{self.gold_path}/matrices"
        self.gold_path_interactions = f"{self.gold_path}/interactions"
        self.gold_path_item_features = f"{self.gold_path}/item_features"
        self.gold_path_models = f"{self.gold_path}/models"

        # Caminhos específicos para treino
        self.train_interactions_path = f"{self.gold_path_interactions}/year=2022/month=7/day=1"

        # Arquivo de download
        self.output_file = f"{self.bronze_path}/challenge-webmedia-e-globo-2023.zip"
        self.download_url = "https://drive.google.com/uc?id=13rvnyK5PJADJQgYe-VbdXb7PpLPj7lPr"
