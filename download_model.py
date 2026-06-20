import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import subprocess

def main():
    model_id = "gemma4:12b"
    logger.info(f"Ollama integration active. Pulling model '{model_id}' via Ollama...")
    try:
        # Run ollama pull gemma4:12b
        process = subprocess.Popen(["ollama", "pull", model_id], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if process.returncode == 0:
            logger.info(f"Successfully pulled model '{model_id}' via Ollama.")
        else:
            logger.error(f"Failed to pull model '{model_id}' (exit code: {process.returncode}).")
            sys.exit(process.returncode)
    except FileNotFoundError:
        logger.error("Ollama CLI not found. Please ensure Ollama is installed and running.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error pulling model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
