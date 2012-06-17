import numpy as np
#from itertools import zip
from Quaternion import Quat
from nose.tools import *

ra = 10.
dec = 20.
roll = 30.
q0 = Quat([ra,dec,roll])
q0m = Quat(np.array([[ra, dec, roll],
                     [ra, dec, roll],
                     [ra, dec, roll],
                     [ra, dec, roll]]))

def test_from_eq():
    q = Quat([ra, dec, roll])
    print 'q.equatorial=', q.equatorial
    assert_almost_equal(q.q[0], 0.26853582)
    assert_almost_equal(q.q[1], -0.14487813)
    assert_almost_equal(q.q[2],  0.12767944)
    assert_almost_equal(q.q[3],  0.94371436)

def test_multi_from_eq():
    q = q0m
    print 'q.equatorial=', q.equatorial
    [assert_almost_equal(qterm, 0.26853582) for qterm in q.q[:, 0]]
    [assert_almost_equal(qterm, -0.14487813)for qterm in q.q[:, 1]]
    [assert_almost_equal(qterm, 0.12767944) for qterm in q.q[:, 2]]
    [assert_almost_equal(qterm, 0.94371436) for qterm in q.q[:, 3]]


def test_from_transform():
    """Initialize from inverse of q0 via transform matrix"""
    q = Quat(q0.transform.transpose())
    assert_almost_equal(q.q[0], -0.26853582)
    assert_almost_equal(q.q[1], 0.14487813)
    assert_almost_equal(q.q[2], -0.12767944)
    assert_almost_equal(q.q[3],  0.94371436)

def test_multi_from_transform():
    q = Quat(q0m.transform.transpose(0,2,1))
    """Initialize from inverse of q0 via transform matrix
    (though since N is the first axis, swap axis 1 and 2 for
    for the transpose)
    """
    [assert_almost_equal(qterm, -0.26853582) for qterm in q.q[:, 0]]
    [assert_almost_equal(qterm, 0.14487813)  for qterm in q.q[:, 1]]
    [assert_almost_equal(qterm, -0.12767944) for qterm in q.q[:, 2]]
    [assert_almost_equal(qterm,  0.94371436) for qterm in q.q[:, 3]]


def test_inv_eq():
    q = Quat(q0.equatorial)
    t = q.transform
    tinv = q.inv().transform
    t_tinv = np.dot(t, tinv)
    for v1, v2 in zip(t_tinv.flatten(), [1,0,0,0,1,0,0,0,1]):
        assert_almost_equal(v1, v2)

def test_multi_inv_eq():
    q = Quat(q0m.equatorial)
    t = q.transform
    tinv = q.inv().transform
    # skipping the matrix multiplication because it is 
    # convoluted
    t_tinv = np.array([np.dot(a, b) for a, b in zip(t, tinv)])
    for t_tinv_single in t_tinv:
        for v1, v2 in zip(t_tinv_single.flatten(), [1,0,0,0,1,0,0,0,1]):
            assert_almost_equal(v1, v2)

def test_inv_q():
    q = Quat(q0.q)
    t = q.transform
    tinv = q.inv().transform
    t_tinv = np.dot(t, tinv)
    for v1, v2 in zip(t_tinv.flatten(), [1,0,0,0,1,0,0,0,1]):
        assert_almost_equal(v1, v2)


def test_multi_inv_q():
    q = Quat(q0m.q)
    t = q.transform
    tinv = q.inv().transform
    # skipping the matrix multiplication because it is 
    # convoluted
    t_tinv = np.array([np.dot(a, b) for a, b in zip(t, tinv)])
    for t_tinv_single in t_tinv:
        for v1, v2 in zip(t_tinv_single.flatten(), [1,0,0,0,1,0,0,0,1]):
            assert_almost_equal(v1, v2)


