# Beacon Fullstack Challenge - Server Application

## Overview
This is the server-side application for the MLET Datathon, built with FastAPI.

## Technologies Used
- **Python 3.9+** - Programming language
- **FastAPI 0.111.0** - Modern web framework for building APIs
- **Pydantic 2.7.1** - Data validation
- **Dependency Injector** - Dependency injection container
- **PySpark** - Distributed computing framework
- **pytest** - Testing framework

## Prerequisites
- Python 3.9 or higher
- Java 8 or higher (required for PySpark)
- Hadoop 3.x (for distributed computing)
- pip (Python package manager)

## Getting Started

### Installation

1. Clone the repository
2. Navigate to the server directory
3. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows
```

4. Install dependencies:

- Basic installation:
```bash
pip install -e .
```

- Development installation:
```bash
pip install -e ".[dev]"
```

### Environment Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your configuration values

### Development

Run the development server:
```bash
uvicorn app.main:app --host localhost --port 8000 --reload
```

The API will be available at `http://localhost:8000`
API documentation will be at `http://localhost:8000/docs`

### Testing

Run tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest tests/ --cov=app/ --cov-report=term-missing
```

### API Testing with Bruno

The project includes a Bruno collection for API testing. Bruno is a fast and git-friendly API client, similar to Postman but with local collections.

To use the Bruno collection:
1. Install Bruno from https://www.usebruno.com/
2. Open Bruno and navigate to the `.bruno` directory in the server folder
3. The collection includes all available API endpoints with example requests

### Production Build

The application can be containerized using Docker. The setup includes all necessary dependencies (Python, Java, Hadoop) and configurations.

#### Prerequisites
- Docker installed on your system
- Model files in `../models/`
- LightFM Features files in `../datalake/gold/lightfm_item_features/` and `../datalake/gold/lightfm_user_features/`

#### Build and Run

1. From the root directory of the project, build the Docker image:
```bash
docker build -t recommendation-server -f Dockerfile.server .
```

2. Run the container:
```bash
docker run -p 8000:8000 --env-file server/.env.docker recommendation-server
```

The API will be available at `http://localhost:8000` and the Swagger documentation at `http://localhost:8000/docs`

#### Environment Configuration

The application uses different environment files for different contexts:
- `.env` - Local development
- `.env.docker` - Docker container configuration

The Docker environment file (`.env.docker`) is configured with absolute paths that match the container's filesystem structure:
- Models path: `/app/models/`
- User features: `/app/datalake/gold/lightfm_user_features/`
- Item features: `/app/datalake/gold/lightfm_item_features/`

#### Docker-specific Troubleshooting

- **File Access Issues**: Ensure the model and data files exist in the correct locations relative to the Dockerfile
- **Memory Issues**: Adjust Spark memory settings in `.env.docker`
- **Connection Issues**: The container exposes port 8000, ensure it's not in use by other services
- **Java/Hadoop Issues**: The container uses OpenJDK 17 and Hadoop 3.3.6, check logs for compatibility issues

## Troubleshooting

### Java/PySpark Issues
- Ensure JAVA_HOME is set correctly in your environment
- Ensure HADOOP_HOME is set correctly in your environment
- Ensure PYSPARK_PYTHON is set correctly in your environment
- Ensure PATH is set correctly in your environment (should include JAVA_HOME/bin and HADOOP_HOME/bin)
- Verify Java version compatibility with PySpark
- Check Hadoop configuration if using distributed mode

### Common Problems
- If you get PySpark initialization errors, verify Java installation
- For memory issues, adjust Spark memory settings in .env file
- Path-related errors might require checking the MODEL_PATH and feature paths in .env