import platform
import subprocess
import os
import argparse

def run_docker_compose(service=None):
    """Executes docker-compose in an OS-agnostic way"""
    try:
        # Create necessary directories
        os.makedirs("datalake/bronze", exist_ok=True)
        os.makedirs("datalake/silver", exist_ok=True)

        if service:
            # Run specific service
            subprocess.run(["docker-compose", "up", "--build", service], check=True)
        else:
            # Run ETL then Jupyter
            subprocess.run(["docker-compose", "up", "--build", "etl"], check=True)
            subprocess.run(["docker-compose", "up", "--build", "jupyter"], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing docker-compose: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute project services')
    parser.add_argument('--service', choices=['etl', 'jupyter'], help='Specific service to run')
    
    args = parser.parse_args()
    
    print(f"Detected Operating System: {platform.system()}")
    run_docker_compose(args.service) 