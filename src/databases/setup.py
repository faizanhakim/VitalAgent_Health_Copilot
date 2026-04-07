import pandas as pd
import sqlite3
import os
import structlog

logger = structlog.get_logger()

CSV_PATH = "data/raw/merged_data.csv"
DB_PATH = "data/health.db"

def setup_database():
    if not os.path.exists(CSV_PATH):
        logger.error("Missing CSV data", path=CSV_PATH)
        print(f"Error: Missing CSV data at path: {CSV_PATH}")
        return
    
    logger.info("Loading CSV")
    df = pd.read_csv(CSV_PATH)

    df.columns = ['user_id', 'date', 'resting_heart_rate', 'total_sleep_minutes', 'calories']

    logger.info("Connecting to SQL lite", db_path=DB_PATH)
    conn = sqlite3.connect(DB_PATH)

    df.to_sql('daily_biometrics', conn, if_exists='replace', index=False)

    conn.close()
    logger.info("Datebase setup completed")
    print(f"Successfully loaded data into database at {DB_PATH}")

if __name__ == "__main__":
    setup_database()

