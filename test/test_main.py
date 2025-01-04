import unittest

def main():
    print("hello")

class TestMain(unittest.TestCase):
    def test_main(self):
        with self.assertLogs(level='INFO') as caplog:
            main()
            self.assertIn('hello', caplog.output)

if __name__ == '__main__':
    unittest.main()
