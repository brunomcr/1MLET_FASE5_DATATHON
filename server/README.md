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
```bash
pip install -r requirements.txt
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

Build the Docker image:
```bash
docker build -t mlet-datathon-server .
```

Run the container:
```bash
docker run -p 8000:8000 mlet-datathon-server
```

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