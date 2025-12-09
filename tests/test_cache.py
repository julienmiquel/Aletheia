import unittest
from ia_detector.cache import ResultCache
import tempfile
import shutil
import os

class TestResultCache(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_result_cache(self):
        # Integration: Real SQLite DB creation
        db_path = os.path.join(self.test_dir, "test_cache.db")
        cache = ResultCache(db_path=db_path)
        
        text = "cache_int_test"
        metric = "test"
        value = {"score": 99}
        
        # Write
        cache.set(text, metric, value)
        
        # Read Same Context
        retrieved = cache.get(text, metric)
        self.assertEqual(retrieved, value)
        
        # Read New Context (Simulate persistence)
        cache2 = ResultCache(db_path=db_path)
        retrieved2 = cache2.get(text, metric)
        self.assertEqual(retrieved2, value)

if __name__ == '__main__':
    unittest.main()
