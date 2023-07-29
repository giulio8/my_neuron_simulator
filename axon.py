import numpy as np

def getModifiedSpikeTrain(delta_train, train, t_refractory_s, v_rest, noise, time_step):
    jitter = noise["jitter"]
    b_delay = noise["base_delay"]
    b_amp = noise["base_amp"]
    amp_CV = noise["amp_CV"]

    # Trasliamo tutto di v_rest per applicare lo scaling alla forma d'onda
    train = train - v_rest

    N = delta_train.size
    N_changes = N//int(t_refractory_s/4/time_step)
    tf = train[:(N-N%N_changes)].reshape((N_changes, N//N_changes))
    tf_delta = delta_train[:(N-N%N_changes)].reshape((N_changes, N//N_changes))
    conv_p_time = int(3*jitter/time_step)
    y = np.zeros(tf.size+conv_p_time)
    S = np.zeros(tf.size+conv_p_time)

    size_piece = tf.shape[1]

    interrupted = False

    checkpoint = 0
    delays = np.zeros(N_changes)
    amp = b_amp
    for i in range(N_changes):
        dirac = np.zeros(conv_p_time)
        delays[i] = np.abs(np.random.normal(b_delay/time_step, jitter/time_step)) if not interrupted else delays[i-1]
        if (tf[i, -1] != 0):
            interrupted = True
        else:
            interrupted = False
        amp = np.random.normal(b_amp, amp_CV) if not interrupted else amp
        idx = int(delays[i]) if int(delays[i]) < dirac.size else dirac.size - 1
        dirac[idx] = 1
        y_i = amp*np.convolve(dirac, tf[i, :])
        S_i = np.convolve(dirac, tf_delta[i, :])
        y_i = np.concatenate((np.zeros(checkpoint), y_i))
        S_i = np.concatenate((np.zeros(checkpoint), S_i))
        if (y.size > y_i.size):
            y += np.concatenate((y_i, np.zeros(y.size-y_i.size)))
            S += np.concatenate((S_i, np.zeros(S.size-S_i.size)))
        else:
            y += y_i[:y.size]
            S += S_i[:S.size]
        checkpoint+= size_piece

    if (y.size < N):
        y = np.concatenate((y, np.zeros(N-y.size)))
        S = np.concatenate((S, np.zeros(N-S.size)))

    #riapplichiamo lo scaling
    y = y + v_rest
        
    return S[:N], y[:N]