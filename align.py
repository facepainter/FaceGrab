'''Alignment and extraction for detected faces'''
import numpy as np
from cv2 import warpAffine

class DetectedFace(object):
    '''Image wrapper for align transforms'''
    def __init__(self, image, landmarks):
        self.image = image
        self.landmarks = landmarks

    def transform(self, size, padding=0):
        '''Warps the face image based on a hard-coded matrix'''
        mat = _umeyama(np.array(self.__landmarks_xy()[17:]), _LANDMARKS_2D)[0:2]
        mat = mat * (size - 2 * padding)
        mat[:, 2] += padding
        return warpAffine(self.image, mat, (size, size))

    def __landmarks_xy(self):
        '''Raw landmarks to Cartesian'''
        return [(p.x, p.y) for p in self.landmarks.parts()]

def _umeyama(X, Y):
    '''
    N-D similarity transform with scaling.
    Adatped from:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_geometric.py
    '''
    N, m = X.shape
    mx = X.mean(axis=0)
    my = Y.mean(axis=0)
    dx = X - mx # N x m
    dy = Y - my # N x m
    A = np.dot(dy.T, dx) / N
    d = np.ones((m,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[m - 1] = -1
    T = np.eye(m + 1, dtype=np.double)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A) #covariance
    if rank == 0:
        return np.nan * T
    elif rank == m - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:m, :m] = np.dot(U, V)
        else:
            s = d[m - 1]
            d[m - 1] = -1
            T[:m, :m] = np.dot(U, np.dot(np.diag(d), V))
            d[m - 1] = s
    else:
        T[:m, :m] = np.dot(U, np.dot(np.diag(d), V))
    scale = 1.0 / dx.var(axis=0).sum() * np.dot(S, d)
    T[:m, m] = my - scale * np.dot(T[:m, :m], mx.T)
    T[:m, :m] *= scale
    return T

_MEAN_FACE_X = np.array([
    0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
    0.553364, 0.490127, 0.42689])

_MEAN_FACE_Y = np.array([
    0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
    0.784792, 0.824182, 0.831803, 0.824182])

_LANDMARKS_2D = np.stack([_MEAN_FACE_X, _MEAN_FACE_Y], axis=1)
