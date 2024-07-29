#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

import unittest

import numpy as np

import sys
from os.path import dirname
sys.path.append(dirname("../../src/symmetry/"))

from point_group_utils import operator_C, point_group

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

    
    # Generate 1000 axial point groups with random n
    def test_cn_group(self):
        for i in range(1000):
            n = np.random.randint(1,100)
            pg_symbol = 'C' + str(n)
            pg = point_group(pg_symbol)

    
    def test_cnv_group(self):
        for i in range(1000):
            n = np.random.randint(2,100)
            pg_symbol = 'C' + str(n) + 'v'
            pg = point_group(pg_symbol)

    
    def test_cnh_group(self):
        for i in range(1000):
            n = np.random.randint(1,100)
            pg_symbol = 'C' + str(n) + 'h'
            pg = point_group(pg_symbol)

    
    def test_sn_group(self):
        for i in range(1000):
            n = 2*np.random.randint(1,50)
            pg_symbol = 'S' + str(n)
            pg = point_group(pg_symbol)

    
    def test_dn_group(self):
        for i in range(1000):
            n = np.random.randint(2,100)
            pg_symbol = 'D' + str(n)
            pg = point_group(pg_symbol)

    
    def test_dnd_group(self):
        for i in range(1000):
            n = np.random.randint(2,100)
            pg_symbol = 'D' + str(n) + 'd'
            pg = point_group(pg_symbol)

    
    def test_dnh_group(self):
        for i in range(1000):
            n = np.random.randint(2,100)
            pg_symbol = 'D' + str(n) + 'h'
            pg = point_group(pg_symbol)


if __name__ == '__main__':
    unittest.main()
