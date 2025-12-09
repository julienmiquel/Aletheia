import sqlite3
import hashlib
import json
import os
from ia_detector import config

class ResultCache:
    def __init__(self, db_path=config.CACHE_DB_PATH):
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, value TEXT)")
        
    def get(self, text, metric_name):
        """Retrieves a cached result if it exists."""
        key = self._make_key(text, metric_name)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT value FROM results WHERE key=?", (key,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            print(f"Cache Read Error: {e}")
        return None
        
    def set(self, text, metric_name, value):
        """Stores a result in the cache."""
        key = self._make_key(text, metric_name)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO results (key, value) VALUES (?, ?)", 
                             (key, json.dumps(value)))
                conn.commit()
        except Exception as e:
            print(f"Cache Write Error: {e}")
        
    def _make_key(self, text, metric_name):
        # Hash text to avoid huge keys
        h = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"{metric_name}:{h}"
