import unittest
from utils import filename_from_string


class TestUtils(unittest.TestCase):

    def test_filename_from_string(self):
        label = '355-FE/DE-109 GEBLÄSE_#1 FLOW-RATE  (m^3/hr)'
        self.assertEqual(
            filename_from_string(label), 
            '355-FE_DE-109-GEBLASE_1-FLOW-RATE-m3_hr'
        )
        self.assertEqual(
            filename_from_string(label, sub=''), 
            '355-FEDE-109-GEBLASE_1-FLOW-RATE-m3hr'
        )
        self.assertEqual(
            filename_from_string(label, sub='^'), 
            '355-FEDE-109-GEBLASE_1-FLOW-RATE-m_3hr'
        )
        self.assertEqual(
            filename_from_string(label, sub=r'^\/*&+'), 
            '355-FE_DE-109-GEBLASE_1-FLOW-RATE-m_3_hr'
        )
        self.assertEqual(
            filename_from_string(label, lowercase=True), 
            '355-fe_de-109-geblase_1-flow-rate-m3_hr'
        )
        self.assertEqual(
            filename_from_string(label, allow_unicode=True), 
            '355-FE_DE-109-GEBLÄSE_1-FLOW-RATE-m3_hr'
        )


if __name__ == '__main__':
    unittest.main()