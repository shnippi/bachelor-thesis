import os, sys
import torch
from dotenv import load_dotenv
load_dotenv()

device = os.environ.get('DEVICE') if torch.cuda.is_available() else "cpu"


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
