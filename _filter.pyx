from __future__ import division

import cython
from cython.parallel import prange
from cython.parallel import parallel
import numpy

cimport numpy


ctypedef unsigned char uint8_t


cdef extern from "math.h" nogil:
    float expf(float)
    float fabs(float)


cdef inline float sqr(float x) nogil:
    return x * x


@cython.boundscheck(False)
@cython.wraparound(False)
def _l_bilateral_solver(uint8_t[:, :] src, int d, s_space, s_color):
    cdef int x, y, i, j, y_r, x_r
    cdef int r = d // 2
    cdef int h = src.shape[0]
    cdef int w = src.shape[1]
    cdef uint8_t f, p
    cdef uint8_t[:, :] pad_src = numpy.zeros((h + (2 * r), w + (2 * r)), dtype=numpy.uint8)
    cdef float wt, sum = 0
    cdef float inv_ss = -0.5 / (s_space * s_space)
    cdef float inv_sc = -0.5 / (s_color * s_color)
    cdef float[:, :] ws = numpy.zeros((d, d), dtype=numpy.float32)
    cdef float[:, :] dst = numpy.zeros((h, w), dtype=numpy.float32)

    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            ws[i+r, j+r] = (sqr(i) + sqr(j)) * inv_ss

    pad_src = numpy.pad(src, ((r, r), (r, r)), 'edge')

    with nogil, parallel():
        for y in prange(r, h + r, schedule='guided'):
            y_r = y - r
            for x in range(r, w + r):
                sum = 0
                x_r = x - r
                p = pad_src[y, x]
                for i in range(d):
                    for j in range(d):
                        f = pad_src[y_r + i, x_r + j]
                        wt = expf(ws[i, j] + sqr(p - f) * inv_sc)
                        dst[y_r, x_r] += f * wt
                        sum += wt
                dst[y_r, x_r] /= sum

    return numpy.round(dst).astype(numpy.uint8)


@cython.boundscheck(False)
@cython.wraparound(False)
def _rgb_bilateral_solver(uint8_t[:, :, :] src, int d, s_space, s_color):
    cdef int x, y, i, j, y_r, x_r
    cdef int r = d // 2
    cdef int h = src.shape[0]
    cdef int w = src.shape[1]
    cdef int c = src.shape[2]
    cdef uint8_t f_r, f_g, f_b
    cdef uint8_t p_r, p_g, p_b
    cdef uint8_t[:, :, :] pad_src = numpy.zeros((h + (2 * r), w + (2 * r), c), dtype=numpy.uint8)
    cdef float wt, sum = 0
    cdef float inv_ss = -0.5 / (s_space * s_space)
    cdef float inv_sc = -0.5 / (s_color * s_color)
    cdef float[:, :] ws = numpy.zeros((d, d), dtype=numpy.float32)
    cdef float[:, :, :] dst = numpy.zeros((h, w, c), dtype=numpy.float32)

    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            ws[i+r, j+r] = (sqr(i) + sqr(j)) * inv_ss

    pad_src = numpy.pad(src, ((r, r), (r, r), (0, 0)), 'edge')

    with nogil, parallel():
        for y in prange(r, h + r, schedule='guided'):
            y_r = y - r
            for x in range(r, w + r):
                sum = 0
                x_r = x - r
                p_r = pad_src[y, x, 0]
                p_g = pad_src[y, x, 1]
                p_b = pad_src[y, x, 2]
                for i in range(d):
                    for j in range(d):
                        f_r = pad_src[y_r + i, x_r + j, 0]
                        f_g = pad_src[y_r + i, x_r + j, 1]
                        f_b = pad_src[y_r + i, x_r + j, 2]
                        wt = expf(ws[i, j] + (sqr(p_r - f_r) + sqr(p_g - f_g) + sqr(p_b - f_b)) * inv_sc)
                        dst[y_r, x_r, 0] += f_r * wt
                        dst[y_r, x_r, 1] += f_g * wt
                        dst[y_r, x_r, 2] += f_b * wt
                        sum += wt
                dst[y_r, x_r, 0] /= sum
                dst[y_r, x_r, 1] /= sum
                dst[y_r, x_r, 2] /= sum

    return numpy.round(dst).astype(numpy.uint8)
