import os
from dotenv import load_dotenv

load_dotenv('../utils/.env')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
MODEL = "gpt-5-mini"   # pick an appropriate model for cost/throughput; gpt-5-mini/o4-mini are examples
MODEL_LOW = "gpt-5-nano"   # for low-effort tasks like metadata extraction
WORDS_PER_CHUNK = 5_000  # adjust based on model context length and expected response size
TEMPERATURE = 1.0 # 0.0 - unsupported by new reasoning models
MAX_RETRIES = 1

# Cost per million tokens (in USD): Flex mode
pricing = {
    "gpt-5-nano": (0.025, 0.2),
    "gpt-5-mini": (0.125, 1),
    'gpt-5.2': (0.625, 5),
    'o4-mini': (0.55, 2.2),
}
INPUT_TOKEN_COST = pricing[MODEL][0]/1e6
OUTPUT_TOKEN_COST = pricing[MODEL][1]/1e6
