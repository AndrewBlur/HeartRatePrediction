import tensorflow.keras.backend as K

def coeff_determination(y_true, y_pred):
    # Custom R2 Score metric for Keras
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

def rmse_metric(y_true, y_pred):
    # Custom RMSE metric for Keras
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
