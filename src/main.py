import platform
import subprocess
import os
import argparse
import time


def run_docker_compose(service=None, mode=None, sample_size=None, epochs=None):
    """Run docker-compose in a system-agnostic way"""
    try:
        # Create necessary directories
        os.makedirs("datalake/bronze", exist_ok=True)
        os.makedirs("datalake/silver", exist_ok=True)
        os.makedirs("datalake/gold", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        if service:
            # Run a specific service
            subprocess.run(["docker-compose", "up", "--build", "-d", service], check=True)
        elif mode == "full":
            # Run ETL
            print("Starting ETL process...")
            subprocess.run(["docker-compose", "run", "etl", "python", "src/pipeline_etl.py"], check=True)

            # Run model training after ETL completes
            print("Starting model training...")
            model_cmd = ["docker-compose", "run", "model", "python", "src/pipeline_model.py"]
            
            # Add sample size parameter if provided
            if sample_size:
                print(f"Training using {sample_size}% of the dataset...")
                model_cmd.extend(["--sample_size", str(sample_size)])
            else:
                print("Training using 100% of the dataset...")
                
            # Add epochs parameter if provided
            if epochs:
                print(f"Training for {epochs} epochs...")
                model_cmd.extend(["--epochs", str(epochs)])
                
            # Execute the model training command
            subprocess.run(model_cmd, check=True)

            # Start other services
            print("Starting Jupyter and Streamlit...")
            subprocess.run(["docker-compose", "up", "--build", "-d", "jupyter"], check=True)
            subprocess.run(["docker-compose", "up", "--build", "-d", "streamlit"], check=True)
        else:
            # Run all services in detached mode
            subprocess.run(["docker-compose", "up", "--build", "etl"], check=True)
            
            # Run model with parameters
            model_cmd = ["docker-compose", "run", "-d", "model", "python", "src/pipeline_model.py"]
            
            # Add sample size parameter if provided
            if sample_size:
                print(f"Training using {sample_size}% of the dataset...")
                model_cmd.extend(["--sample_size", str(sample_size)])
            else:
                print("Training using 100% of the dataset...")
                
            # Add epochs parameter if provided
            if epochs:
                print(f"Training for {epochs} epochs...")
                model_cmd.extend(["--epochs", str(epochs)])
                
            # Execute the model training command
            subprocess.run(model_cmd, check=True)
            
            subprocess.run(["docker-compose", "up", "--build", "-d", "jupyter"], check=True)
            subprocess.run(["docker-compose", "up", "--build", "-d", "streamlit"], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error running docker-compose: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run project services')
    parser.add_argument('--service', choices=['etl', 'jupyter', 'streamlit', 'model'],
                        help='Specific service to run')
    parser.add_argument('--mode', choices=['full'],
                        help='Run mode: full = ETL + Training + Services')
    parser.add_argument('--sample_size', type=float,
                        help='Percentage of data to use for model training (e.g., 1.0 for 1%, 10.0 for 10%)')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs for the model')

    args = parser.parse_args()

    print(f"Detected Operating System: {platform.system()}")
    run_docker_compose(args.service, args.mode, args.sample_size, args.epochs)
