from matplotlib.pyplot import axis
import numpy as np
from basetool import *
from CFG import *
from beamformer import *

def wpe(Y_FTM=None, L=10, delay=1, n_iter=3, return_filter=False, ref_pick=0):
    """
    implementation of wighted prediction error(WPE) method for audio signal dereverberation
    parameters
    ------------------------------
    input
        Y_FTM:  reverberant multichannel signal in time-frequency domain
        L:      the order of the prediction filter
        delay:  the delay in frame
        n_iter: number of iterations
    output
        YHat:   dereverberated signal
        h_F_ML:  prediction filter of length M*L, where M is the number of channels
    Reference   
        [1] T Yoshioka, “Speech enhancement in reverberant environments”, Ph.D.
            dissertation, Kyoto University, 2010
    """
    eps = 1e-6
    n_freq, n_frame, n_channel = Y_FTM.shape

    # reshape tensor signal into matrix form 
    Y_FP = np.squeeze(Y_FTM.reshape((n_freq, 1, n_frame*n_channel)))
    K = L*n_channel
    Y_buffer_FK = np.zeros((n_freq, K)).astype(Y_FTM.dtype)
    Y_buffer_FKT = np.zeros((n_freq, K, n_frame-delay-1)).astype(Y_FTM.dtype)
    eyes = np.tile(np.eye(K), [n_freq, 1, 1]).astype(Y_FTM.dtype)
    # calculate 
    # The shift-in scheme is stupid and can be effcacy improved with shape and reshape
    for t_index in range(delay+1, n_frame):
        # shift signal in 
        Y_buffer_FK = np.roll(Y_buffer_FK, n_channel, axis=1)
        Y_buffer_FK[:, :n_channel] = Y_FP[:, (t_index-delay-1)*n_channel:(t_index-delay)*n_channel]
        Y_buffer_FKT[..., t_index-delay-1] = Y_buffer_FK

    # pick reference signal
    # the DOA estimation and Beamformer Module are not uploaded, you can just pick any microphone signal
    # as reference, i.e, ref_pick=0~M-1 or use average
    if ref_pick == "average":
        ref = (np.mean(Y_FTM, axis=2, keepdims=False))
    elif ref_pick == "MPDR":
        theta = DOA_MUSIC_estimation(Y_FTM, weight="PHAT", n_src=1)
        steerVecSOI_FM1 = steerVec_constrcut(theta=theta)
        covY_FMM = CovMat_estimation(Y_FTM, Roubust=True)
        MPDR_F1M = MPDR_construct(steerVecSOI_FM1=steerVecSOI_FM1, CovMat_FMM=covY_FMM)
        ref = np.squeeze(MPDR_F1M @ Y_FTM.swapaxes(1,2))
    elif ref_pick == "DS":
        theta = DOA_MUSIC_estimation(Y_FTM, weight="PHAT", n_src=1)
        steerVecSOI_FM1 = steerVec_constrcut(theta=theta)
        covY_FMM = CovMat_estimation(Y_FTM, Roubust=True)
        DS_F1M = DS_construct(steerVecSOI_FM1=steerVecSOI_FM1)
        ref = np.squeeze(DS_F1M @ Y_FTM.swapaxes(1,2))
    elif ref_pick == "GEVD":
        covY_FMM = CovMat_estimation(Y_FTM, Roubust=True)
        GEV_F1M = GEV_constrcut(Cov_FMM=covY_FMM)
        ref = np.squeeze(GEV_F1M @ Y_FTM.swapaxes(1,2))
    else:
        ref = (np.squeeze(Y_FTM[..., int(ref_pick)]))
    ref_FT = ref[:, delay+1:].copy()

    # implement wighted prediction error
    h_FK1 = np.zeros((n_freq, K, 1)).astype(Y_FTM.dtype) 
    Y_dereved_FT = np.zeros((n_freq, n_frame-delay-1)).astype(Y_FTM.dtype)
    def dereverberation():
        Y_dereved_FT[:, :] = ref_FT - np.squeeze(np.conj(h_FK1.swapaxes(1,2)) @ Y_buffer_FKT)

    def wpe_update():
        # dereverberation
        dereverberation()
        lambda_FT = np.abs(Y_dereved_FT) ** 2.0
        lambda_FT[lambda_FT<eps] = eps
        lambda_inv_FT = 1.0 / lambda_FT
        # update dereverberation matrix
        R_FKK = (Y_buffer_FKT * lambda_inv_FT[:, None, :]) @ np.conj(Y_buffer_FKT.swapaxes(1,2)) + eps*eyes
        p_FK1 = np.sum((Y_buffer_FKT* lambda_inv_FT[:, None, :]) * np.conj(ref_FT[:, None, :]), axis=2, keepdims=True)
        h_FK1[:, :, :] = np.linalg.inv(R_FKK) @ p_FK1

    for epoch in range(n_iter):
        wpe_update()
    dereverberation()
    YHat = np.hstack((ref[:, :delay+1], Y_dereved_FT))
    return YHat

    
    


    