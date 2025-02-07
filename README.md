# 1MLET_FASE5_DATATHON


# Data Analysis Pipeline with PySpark

This project implements an ETL (Extract, Transform, Load) pipeline and data analysis environment using PySpark, containerized with Docker.

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
