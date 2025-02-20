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
5. Test and validate the API’s predictions.
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

The goal is to predict the user’s next interactions based on their browsing history.

## Overview

The pipeline:
1. Downloads data from a specified source
2. Processes it through bronze and silver layers
3. Provides a Jupyter environment for data exploration

## Project Structure

```
├── datalake/ # Data storage
│ ├── bronze/ # Raw data
│ └── silver/ # Processed data
├── src/
│ ├── configs/ # Configuration files
│ ├── services/ # Core services (ETL, transformations)
│ ├── utils/ # Utility functions
│ ├── main.py # ETL entry point
│ └── run.py # Script to run containers
├── notebooks/ # Jupyter notebooks
├── docker-compose.yml # Container orchestration
├── Dockerfile.etl # ETL container configuration
├── Dockerfile.jupyter # Jupyter container configuration
├── pyproject.toml # Poetry dependencies
└── README.md
```
## Prerequisites

- Docker Desktop
- Python 3.9+

## Quick Start

1. Clone the repository:

```bash
git clone <repository-url>
cd <project-directory>
```

2. Run the ETL process:

```bash
python src/run.py --service etl
```

3. Start Jupyter environment:

```bash
python src/run.py --service jupyter
```


4. Access Jupyter Lab:
- Open browser at `http://localhost:8888`
- Spark UI available at `http://localhost:4040`

## Container Details

### ETL Container
- Processes raw data
- Runs automatically and exits when complete
- Resource limits: 16GB RAM, 12 CPUs

### Jupyter Container
- Interactive data analysis environment
- Persistent session
- Resource limits: 16GB RAM, 8 CPUs

## Data Processing

1. **Bronze Layer**
   - Raw data storage
   - Minimal transformations
   - Preserves original data

2. **Silver Layer**
   - Cleaned and transformed data
   - Business logic applied
   - Optimized for analysis

## Development

### Adding Dependencies

Dependencies are managed with Poetry inside containers. To add new packages:

1. Update `pyproject.toml`:

toml
[tool.poetry.dependencies]
python = ">=3.9"
new-package = "^1.0.0"


2. Rebuild containers:

```bash
docker-compose build --no-cache
```

### Project Configuration

Key configurations in `docker-compose.yml`:
- Volume mappings
- Resource limits
- Environment variables
- Port mappings

## Monitoring

- ETL logs available in container output
- Spark UI for performance monitoring
- Jupyter Lab for interactive development

## Best Practices

1. **Data Management**
   - Use appropriate data layers (bronze/silver)
   - Implement data validation
   - Monitor data quality

2. **Resource Management**
   - Configure Spark memory appropriately
   - Monitor resource usage
   - Use caching strategically

3. **Development**
   - Follow PEP 8 style guide
   - Document code changes
   - Use version control

## Troubleshooting

1. **Container Issues**

Reset Docker environment

```bash
docker-compose down
docker system prune -a --volumes
```

Reset Docker environment
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

## License

MIT License

## Contributors (Group 43)
- Bruno Machado Corte Real
- Pedro Henrique Romaoli Garcia
- Rodrigo Santili Sgarioni
