#!/usr/bin/env python

"""
@brief   This module provides functions to perform OT-based domain adaptation.
@author  Luis Carlos Garcia-Peraza Herrera (luiscarlos.gph@gmail.com).
@date    1 Jun 2020.
"""

import numpy as np
import cv2 
import random
import ot


def otda(source_im: np.ndarray, target_im: np.ndarray, method: str = 'linear',
         nsamples:int = 1000):
    """
    @brief Adapt the source image to the domain of the target image.

    @param[in]  source_im  OpenCV/Numpy source BGR image in range [0, 255], 
                           dtype = np.uint8.
    @param[in]  target_im  Target domain BGR image in range [0, 255], 
                           dtype = np.uint8.
    @param[in]  method     Domain adaptation method. The options are:
                           'linear', 'linear_fourier', 'gaussian', 'sinkhorn',
                           'emd'.
    @param[in]  nsamples   Number of pixels used to estimate the transport.

    @returns the domain adapted image.
    """

    # Flatten the images
    source_flat = source_im.reshape((source_im.shape[0] * source_im.shape[1], source_im.shape[2]))
    target_flat = target_im.reshape((target_im.shape[0] * target_im.shape[1], target_im.shape[2]))

    # Normalise to range [0, 1]
    source_norm = source_flat.astype(np.float64) / 255.
    target_norm = target_flat.astype(np.float64) / 255.

    # Create and fit the model 
    adapted_im = None 
    if method == 'linear':
        mapping = ot.da.LinearTransport()
        mapping.fit(Xs=source_norm, Xt=target_norm)
        adapted = np.clip(mapping.transform(Xs=source_norm), 0, 1)
        adapted_im = np.round(adapted * 255.).astype(np.uint8).reshape(source_im.shape)

    elif method == 'linear_fourier': 
        mapping = ot.da.LinearTransport()

        adapted_im = np.empty_like(source_im)
        for k in range(3):
            amp_s, phase_s = fourier.fft_amp_phase(source_im[:, :, k])
            amp_t, phase_t = fourier.fft_amp_phase(target_im[:, :, k])

            amp_s = amp_s.flatten() 
            amp_t = amp_t.flatten() 
            amp_s = amp_s.reshape((amp_s.shape[0], 1))
            amp_t = amp_t.reshape((amp_t.shape[0], 1))

            mapping.fit(Xs=amp_s, Xt=amp_t)
            amp_s_adapted = mapping.transform(Xs=amp_s).reshape((adapted_im.shape[0], adapted_im.shape[1]))
            
            adapted_im[:, :, k] = fourier.ifft_amp_phase(amp_s_adapted, phase_s)

    elif method == 'gaussian':
        mapping = ot.da.MappingTransport(mu=1e0, eta=1e-2, sigma=1, bias=False, max_iter=10)
        source_idx = np.random.randint(source_norm.shape[0], size=(nsamples,))
        target_idx = np.random.randint(target_norm.shape[0], size=(nsamples,))
        mapping.fit(Xs=source_norm[source_idx, :], Xt=target_norm[target_idx, :])
        adapted = np.clip(mapping.transform(Xs=source_norm), 0, 1)
        adapted_im = np.round(adapted * 255.).astype(np.uint8).reshape(source_im.shape)

    elif method == 'sinkhorn':
        mapping = ot.da.SinkhornTransport(reg_e=1e-1)
        source_idx = np.random.randint(source_flat.shape[0], size=(nsamples,))
        target_idx = np.random.randint(target_flat.shape[0], size=(nsamples,))
        mapping.fit(Xs=source_norm[source_idx, :], Xt=target_norm[target_idx, :])
        adapted = np.clip(mapping.transform(Xs=source_norm), 0, 1)
        adapted_im = np.round(adapted * 255.).astype(np.uint8).reshape(source_im.shape)

    elif method == 'emd':
        mapping = ot.da.EMDTransport()
        source_idx = np.random.randint(source_flat.shape[0], size=(nsamples,))
        target_idx = np.random.randint(target_flat.shape[0], size=(nsamples,))
        mapping.fit(Xs=source_norm[source_idx, :], Xt=target_norm[target_idx, :])
        adapted = np.clip(mapping.transform(Xs=source_norm), 0, 1)
        adapted_im = np.round(adapted * 255.).astype(np.uint8).reshape(source_im.shape)

    else:
        raise ValueError('[ERROR] Optimal transport method not recognised.')

    return adapted_im


if __name__ == '__main__':
    raise RuntimeError('[ERROR] This Python module is not meant to be run as a script.')
