# Datathon - News Recommendation System

This project was developed as part of Datathon Phase 5 and aims to build a news recommendation system using data from the G1 portal, predicting which news articles a user is likely to read next.

**Challenge**

Content recommendation is essential for media platforms due to the vast amount of information generated daily. In the case of news, challenges such as recency and cold-start must be considered:

- Cold-start: How to recommend news for new users or when there is limited data available?
- Recency: How to ensure that recommendations are relevant, considering that older news articles may no longer be useful?

**Project Objectives**

The recommendation system should:

1. Train a model to predict which news articles a user will consume.
2. Save the trained model for future use.
3. Create an API to serve recommendations.
4. Package the solution with Docker for easy deployment.
5. Test and validate the API's predictions.
6. Deploy locally or in the cloud (optional).

**About the Data**

The dataset is divided into training and validation sets.

**Training Data**

The treino_parte_X.csv files contain information about user interactions with news articles, including:

- userId: User identification.
- history: List of previously read news articles.
- TimestampHistory: Time of reading.
- scrollPercentageHistory: Percentage of the page viewed.
- pageVisitsCountHistory: Number of visits to the same article.

Additionally, there is a subfolder containing details about the news articles:

- Page (Article ID), Title, Body, Issued (publication date), Modified (last modification).

**Validation Data**

Captures interactions from a period after the training set, including:

- userId
- userType (logged-in or anonymous)
- history (news articles to be recommended)

The goal is to predict the user's next interactions based on their browsing history.

## Overview

The pipeline:
1. Downloads data from a specified source
2. Processes it through bronze and silver layers
3. Provides a Jupyter environment for data exploration

## Project Structure

```
├── datalake/              # Data storage
│   ├── bronze/            # Raw data storage
│   ├── silver/            # Processed and cleaned data
│   └── gold/              # Final modeling data
├── src/                   # Source code
│   ├── configs/           # Configuration files
│   ├── services/          # Core services (ETL, transformations)
│   ├── utils/             # Utility functions
│   ├── main.py            # Main application entry point
│   ├── pipeline_etl.py    # ETL pipeline implementation
│   ├── pipeline_model.py  # Model training pipeline
│   └── run.py             # Container execution script
├── notebooks/             # Jupyter notebooks for analysis
├── models/                # Trained models and monitoring
├── streamlit/             # Streamlit web application
│   └── app.py             # Main Streamlit application
├── docker-compose.yml     # Container orchestration
├── Dockerfile.etl         # ETL service container
├── Dockerfile.jupyter     # Jupyter service container
├── Dockerfile.model       # Model training container
├── Dockerfile.streamlit   # Streamlit app container
├── pyproject.toml         # Poetry project dependencies
├── LICENSE                # MIT License
└── README.md              # Project documentation
```

## Prerequisites

- Docker Desktop
- Python 3.9+

## Technologies Used

- Python 3.9+
- Docker
- Poetry
- Jupyter Lab
- Streamlit
- Spark

## Setup and Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <project-directory>
```

2. Install dependencies using Poetry:

```bash
poetry install
```

3. Build Docker containers:

```bash
docker-compose build
```

## Running the Project

### ETL Process

To run the ETL process, execute:

```bash
python src/main.py --service etl
```

### Model Training

To train the model, execute:

```bash
python src/main.py --mode full --sample_size <percentage> --epochs <number>
```

### Jupyter and Streamlit

To start the Jupyter and Streamlit services, execute:

```bash
python src/main.py --service jupyter
python src/main.py --service streamlit
```

Access Jupyter Lab at `http://localhost:8888` and the Spark UI at `http://localhost:4040`.

## Data Processing

1. **Bronze Layer**
   - Raw data storage
   - Minimal transformations
   - Preserves original data

2. **Silver Layer**
   - Cleaned and transformed data
   - Business logic applied
   - Optimized for analysis

## Development and Testing

### Adding Dependencies

Dependencies are managed with Poetry inside containers. To add new packages:

1. Update `pyproject.toml`:

```toml
[tool.poetry.dependencies]
python = ">=3.9"
new-package = "^1.0.0"
```

2. Rebuild containers:

```bash
docker-compose build --no-cache
```

## Monitoring and Troubleshooting

- ETL logs available in container output
- Spark UI for performance monitoring
- Jupyter Lab for interactive development

### Troubleshooting

1. **Container Issues**

Reset Docker environment:

```bash
docker-compose down
docker system prune -a --volumes
```

2. **Performance Issues**
- Check resource allocation in `docker-compose.yml`
- Monitor Spark UI
- Optimize Spark configurations

3. **Data Access Issues**
- Verify volume mappings
- Check file permissions
- Validate data paths

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License and Contributors

MIT License

## References

- [Docker Documentation](https://docs.docker.com/)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [Spark Documentation](https://spark.apache.org/docs/latest/)

## Contributors (Group 43)
- Bruno Machado Corte Real
- Pedro Henrique Romaoli Garcia
- Rodrigo Santili Sgarioni
