import logging
import sys
import time
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    model_id = "google/gemma-3-4b-it"
    logger.info(f"Starting download of model {model_id}...")
    
    retries = 5
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{retries}...")
            # We download the model. snapshot_download handles resumes by default.
            path = snapshot_download(
                repo_id=model_id,
                resume_download=True,
                max_workers=4
            )
            logger.info(f"Successfully downloaded model to: {path}")
            break
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            if attempt < retries:
                logger.info("Waiting 10 seconds before retrying...")
                time.sleep(10)
            else:
                logger.error("All download retries exhausted.")
                sys.exit(1)

if __name__ == "__main__":
    main()
