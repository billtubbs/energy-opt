import unittest
from utils import (
    filename_from_string, 
    calculate_3_phase_power,
    calculate_running_status,
    drop_values_when_not_running,
    split_into_groups_of_values,
    groups_of_true_values
)
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


class TestUtils(unittest.TestCase):

    def test_calculate_3_phase_power(self):

        voltage, current = 415, 0.5
        self.assertEqual(
            calculate_3_phase_power(voltage, current),
            voltage * current * np.sqrt(3) * 0.95
        )

    
    def test_calculate_running_status(self):

        data = pd.DataFrame({
            'Pump 1 Amps': [ 0., 86.4, 0., 62.3, 91.5],
            'Pump 2 Amps': [60.6, 63.,  0.,  0., 95.4],
        })
        running_status = calculate_running_status(data, min_value=10)
        assert_frame_equal(
            running_status,
            pd.DataFrame([
                [False, True],
                [True, True],
                [False, False],
                [True, False],
                [True, True]
            ], columns=data.columns)
        )


    def test_drop_values_when_not_running(self):

        data = pd.DataFrame({
            'Pump 1 Amps': [ 0., 86.4, 0., 62.3, 91.5],
            'Pump 2 Amps': [60.6, 63.,  0.,  0., 95.4],
        })
        running_status = calculate_running_status(data, min_value=10)
        amps_when_running = drop_values_when_not_running(data, running_status)
        assert_frame_equal(
            amps_when_running,
            pd.DataFrame([
                [np.nan, 60.6],
                [86.4, 63.0],
                [np.nan, np.nan],
                [62.3, np.nan],
                [91.5, 95.4]
            ], columns=data.columns)
        )


    def test_split_into_groups_of_values(self):

        running_status = pd.Series(
            ['Fully loaded', 'Fully loaded', 'Idling', 'Fully loaded',
             'Fully loaded', 'Idling', 'Idling', 'Idling']
        )
        running_status_counts = split_into_groups_of_values(running_status)
        assert_frame_equal(
            running_status_counts,
            pd.DataFrame([
                [1, 0],
                [1, 0],
                [0, 1],
                [2, 0],
                [2, 0],
                [0, 2],
                [0, 2],
                [0, 2]
            ], columns=['Fully loaded', 'Idling'])
        )


    def test_groups_of_true_values(x):
        x = [True, False, True, True, False]
        assert np.array_equal(
            groups_of_true_values(x),
            [1, 0, 2, 2, 0]
        )


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