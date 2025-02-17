import platform
import subprocess
import os
import argparse


def run_docker_compose(service=None):
    """Run docker-compose in a system-agnostic way"""
    try:
        # Create necessary directories
        os.makedirs("datalake/bronze", exist_ok=True)
        os.makedirs("datalake/silver", exist_ok=True)

        if service:
            # Run a specific service
            subprocess.run(["docker-compose", "up", "--build", "-d", service], check=True)
        else:
            # Run all services in detached mode
            subprocess.run(["docker-compose", "up", "--build", "etl"], check=True)
            subprocess.run(["docker-compose", "up", "--build", "-d", "jupyter"], check=True)
            subprocess.run(["docker-compose", "up", "--build", "-d", "streamlit"], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running docker-compose: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run project services')
    parser.add_argument('--service', choices=['etl', 'jupyter', 'streamlit'], help='Specific service to run')
    
    args = parser.parse_args()
    
    print(f"Detected Operating System: {platform.system()}")
    run_docker_compose(args.service)
