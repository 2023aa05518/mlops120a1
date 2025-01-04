import unittest
from unittest.mock import patch
import io
from main import main


class TestMainFunction(unittest.TestCase):

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_main_output(self, stdout):
        main()
        self.assertEqual(stdout.getvalue().strip(), "Hello")


if __name__ == '__main__':
    unittest.main()
