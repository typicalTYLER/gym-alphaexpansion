import unittest
import gym_alphaexpansion.utils as utils
import numpy as np


class UtilsTest(unittest.TestCase):
    def test_negative_allowing_log_10(self):
        #self.assertEquals(utils.negative_allowing_log_10(np.asarray([10, -10, 10, -20])))
        print(utils.abs_max_scaling(utils.negative_allowing_log_10(np.asarray([10, -10, 9999999, -20, 0]))))
        print(utils.abs_max_scaling(utils.negative_allowing_log_10(np.asarray([0, 0, 0, 0, 0]))))
        print(utils.abs_max_scaling(utils.negative_allowing_log_10(np.asarray([-8.0, 0.0, 1.0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))))


if __name__ == '__main__':
    unittest.main()
