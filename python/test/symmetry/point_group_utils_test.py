#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

import unittest

import numpy as np

import sys
from os.path import dirname
sys.path.append(dirname("/home/azelenskiy/Documents/"\
                      + "frusa_symmetry/python/src/symmetry/"))

from point_group_utils import operator_C

class TestPointGroupUtils(unittest.TestCase):
    
    def test_x_rotation(self):
        c2_x = np.array([ [  1,  0,  0 ],
                          [  0, -1,  0 ],
                          [  0,  0, -1 ] ])
        
        c2_x_module = operator_C([1,0,0],2)
        
        diff = np.round(np.max(np.abs(c2_x - c2_x_module)),10)

        self.assertEqual(diff,0.0)
    
    def test_y_rotation(self):
        c2_y = np.array([ [ -1,  0,  0 ],
                          [  0,  1,  0 ],
                          [  0,  0, -1 ] ])
        
        c2_y_module = operator_C([0,1,0],2)
        
        diff = np.round(np.max(np.abs(c2_y - c2_y_module)),10)

        self.assertEqual(diff,0.0)
    
    def test_z_rotation(self):
        c2_z = np.array([ [ -1,  0,  0 ],
                          [  0, -1,  0 ],
                          [  0,  0,  1 ] ])
        
        c2_z_module = operator_C([0,0,1],2)
        
        diff = np.round(np.max(np.abs(c2_z - c2_z_module)),10)

        self.assertEqual(diff,0.0)

if __name__ == '__main__':
    unittest.main()
