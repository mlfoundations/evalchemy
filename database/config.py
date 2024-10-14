import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# DB_PASSWORD = os.getenv("DB_PASSWORD")
# DB_HOST = os.getenv("DB_HOST")
# DB_PORT = os.getenv("DB_PORT")
# DB_NAME = os.getenv("DB_NAME")
# DB_USER = os.getenv("DB_USER")

# DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

password = "t}LQ7ZL]3$x~I8ye"
ALLOY_DB_IP = "35.225.163.235"
port = "5432"
DATABASE_URL = f"postgresql://postgres:{password}@{ALLOY_DB_IP}:{port}/postgres"
