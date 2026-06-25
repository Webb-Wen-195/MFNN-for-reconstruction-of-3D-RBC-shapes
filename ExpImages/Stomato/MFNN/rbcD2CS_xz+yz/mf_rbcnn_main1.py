import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt
from datapre import DataSet
from NN import rbc_mfNN
import time
np.random.seed(1117)
tf.random.set_seed(1117)


@tf.function(reduce_retracing=True)
def train_step1(model, x_lf, x_hf, u_lf, u_hf, optimizer):
    with tf.GradientTape() as tape:
        loss, loss_lf, loss_av_lf, loss_hf, loss_av_hf, Ae, Ve = model.loss_rbc(x_lf, x_hf, u_lf, u_hf)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    return loss, loss_lf, loss_av_lf, loss_hf, loss_av_hf, Ae, Ve 

@tf.function(reduce_retracing=True)
def train_step2(model, x_lf, x_hf, u_lf, u_hf, optimizer):
    with tf.GradientTape() as tape:
        loss, loss_lf, loss_av_lf, loss_hf, loss_av_hf, Ae, Ve = model.loss_rbc(x_lf, x_hf, u_lf, u_hf)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    return loss, loss_lf, loss_av_lf, loss_hf, loss_av_hf, Ae, Ve 

def main():
    batchsize = 128
    data = DataSet(batchsize)

    x_lf_all = data.x_lf_all
    x_hf_all = data.x_hf_all
    id_tsr = tf.convert_to_tensor(x_lf_all[:, :1], dtype=tf.float32)
    x_lf_all_tsr = tf.convert_to_tensor(x_lf_all[:, 1:], dtype=tf.float32)
    x_hf_all_tsr = tf.convert_to_tensor(x_hf_all[:, 1:], dtype=tf.float32)
    all_x = [x_lf_all_tsr, x_hf_all_tsr]

    tri_tsr = tf.convert_to_tensor(data.tri_archive, dtype=tf.float32)
    lf_av_tsr = tf.convert_to_tensor(data.AV_lf, dtype=tf.float32)
    hf_av_tsr = tf.convert_to_tensor(data.AV_hf, dtype=tf.float32)
    rbcinfo = [id_tsr, tri_tsr, lf_av_tsr, hf_av_tsr]

    mean_tsr = tf.convert_to_tensor(data.u_mean, dtype=tf.float32)
    std_tsr = tf.convert_to_tensor(data.u_std, dtype=tf.float32)
    decod = [mean_tsr, std_tsr]

    x_lf_test_tsr = tf.convert_to_tensor(data.x_lf_test, dtype=tf.float32)
    u_lf_test_tsr = tf.convert_to_tensor(data.u_lf_test, dtype=tf.float32)

    x_hf_train_tsr = tf.convert_to_tensor(data.x_hf_train, dtype=tf.float32)
    u_hf_train_tsr = tf.convert_to_tensor(data.u_hf_train, dtype=tf.float32)
    x_hf_test_tsr = tf.convert_to_tensor(data.x_hf_test, dtype=tf.float32)
    u_hf_test_tsr = tf.convert_to_tensor(data.u_hf_test, dtype=tf.float32)

    lfc_layers = [3] + [32] + [64] + [128] + [256]
    lf_layers = [256] + [100]*3 + [3]
    hf_nl_layers = [6] + [100]*3 + [3]
    hf_l_layers = [6] + [3]
    all_layers = [lfc_layers, lf_layers, hf_nl_layers, hf_l_layers]

    rbcnn = rbc_mfNN(all_layers, all_x, rbcinfo, decod, batch_size = batchsize)

    start_time = time.time()
    loss = 100
    epoch = 0


    with open("./output_record_mfS_1.txt", "w") as file:
      
        optimizer1 = tf.keras.optimizers.Adam(learning_rate=1e-4) 
        for epoch in range(200*1000):
            x_lf_train, u_lf_train = data.minibatch()

            x_lf_train = tf.convert_to_tensor(x_lf_train, dtype=tf.float32)
            u_lf_train = tf.convert_to_tensor(u_lf_train, dtype=tf.float32)

            x_hf_train = x_hf_train_tsr
            u_hf_train = u_hf_train_tsr


            loss, loss_lf, loss_av_lf, loss_hf, loss_av_hf, Ae, Ve = train_step1(rbcnn, x_lf_train, x_hf_train, u_lf_train, u_hf_train, optimizer1)

            # Print something every N epochs
            if (epoch+1) % 1000 == 0:
                
                u_lf_pred, u_hf_pred, _, _ = rbcnn(x_lf_test_tsr, x_hf_test_tsr)
                u_lf_pred = data.decode(u_lf_pred)
                u_hf_pred = data.decode(u_hf_pred)

                relative_error_lf = tf.norm(u_lf_pred - u_lf_test_tsr) / tf.norm(u_lf_test_tsr)
                relative_error_hf = tf.norm(u_hf_pred - u_hf_test_tsr) / tf.norm(u_hf_test_tsr)

                mid_time = time.time()
                elapsed_time = mid_time - start_time

                print(f"Epoch {epoch+1}:\n  Train-loss[total = {loss.numpy():.4e}, "
                        f"C_lf = {loss_lf.numpy():.4e}, "
                        f"A/V_lf = {loss_av_lf.numpy():.4e}, "
                        f"C_hf = {loss_hf.numpy():.4e}, "
                        f"A/V_hf = {loss_av_hf.numpy():.4e}],\n"
                        f"  Test-error[C_lf = {relative_error_lf.numpy():.4e}, "
                        f"C_hf = {relative_error_hf.numpy():.4e}], "
                        f"A_hf = {Ae.numpy():.4e}, "
                        f"V_hf = {Ve.numpy():.4e}, "
                        f"time: {elapsed_time:.2f}s\n")
                file.write(f"Epoch {epoch+1}:\n  Train-loss[total = {loss.numpy():.4e}, "
                        f"C_lf = {loss_lf.numpy():.4e}, "
                        f"A/V_lf = {loss_av_lf.numpy():.4e}, "
                        f"C_hf = {loss_hf.numpy():.4e}, "
                        f"A/V_hf = {loss_av_hf.numpy():.4e}],\n"
                        f"  Test-error[C_lf = {relative_error_lf.numpy():.4e}, "
                        f"C_hf = {relative_error_hf.numpy():.4e}], "
                        f"A_hf = {Ae.numpy():.4e}, "
                        f"V_hf = {Ve.numpy():.4e}, "
                        f"time: {elapsed_time:.2f}s\n")
                file.flush()
        
        optimizer2 = tf.keras.optimizers.SGD(learning_rate = 1e-5,
                                             momentum=0.9,
                                             nesterov = True # Optional: enabling Nesterov momentum may help with convergence
                                             )
        
        for epoch in range(200*1000,400*1000):
            x_lf_train, u_lf_train = data.minibatch()

            x_lf_train = tf.convert_to_tensor(x_lf_train, dtype=tf.float32)
            u_lf_train = tf.convert_to_tensor(u_lf_train, dtype=tf.float32)

            x_hf_train = x_hf_train_tsr
            u_hf_train = u_hf_train_tsr


            loss, loss_lf, loss_av_lf, loss_hf, loss_av_hf, Ae, Ve = train_step2(rbcnn, x_lf_train, x_hf_train, u_lf_train, u_hf_train, optimizer2)

            # Print something every N epochs
            if (epoch+1) % 1000 == 0:
                
                u_lf_pred, u_hf_pred, _, _ = rbcnn(x_lf_test_tsr, x_hf_test_tsr)
                u_lf_pred = data.decode(u_lf_pred)
                u_hf_pred = data.decode(u_hf_pred)

                relative_error_lf = tf.norm(u_lf_pred - u_lf_test_tsr) / tf.norm(u_lf_test_tsr)
                relative_error_hf = tf.norm(u_hf_pred - u_hf_test_tsr) / tf.norm(u_hf_test_tsr)

                mid_time = time.time()
                elapsed_time = mid_time - start_time

                print(f"Epoch {epoch+1}:\n  Train-loss[total = {loss.numpy():.4e}, "
                        f"C_lf = {loss_lf.numpy():.4e}, "
                        f"A/V_lf = {loss_av_lf.numpy():.4e}, "
                        f"C_hf = {loss_hf.numpy():.4e}, "
                        f"A/V_hf = {loss_av_hf.numpy():.4e}],\n"
                        f"  Test-error[C_lf = {relative_error_lf.numpy():.4e}, "
                        f"C_hf = {relative_error_hf.numpy():.4e}], "
                        f"A_hf = {Ae.numpy():.4e}, "
                        f"V_hf = {Ve.numpy():.4e}, "
                        f"time: {elapsed_time:.2f}s\n")
                file.write(f"Epoch {epoch+1}:\n  Train-loss[total = {loss.numpy():.4e}, "
                        f"C_lf = {loss_lf.numpy():.4e}, "
                        f"A/V_lf = {loss_av_lf.numpy():.4e}, "
                        f"C_hf = {loss_hf.numpy():.4e}, "
                        f"A/V_hf = {loss_av_hf.numpy():.4e}],\n"
                        f"  Test-error[C_lf = {relative_error_lf.numpy():.4e}, "
                        f"C_hf = {relative_error_hf.numpy():.4e}], "
                        f"A_hf = {Ae.numpy():.4e}, "
                        f"V_hf = {Ve.numpy():.4e}, "
                        f"time: {elapsed_time:.2f}s\n")
                file.flush()
        file.close()
    
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print('======================================================')
    print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{epoch+1} epochs in total: Train-loss[total = {loss.numpy():.4e}, "
                f"coordinate_lf = {loss_lf.numpy():.4e}, "
                f"area/volume_lf = {loss_av_lf.numpy():.4e}, "
                f"coordinate_hf = {loss_hf.numpy():.4e}, "
                f"area/volume_hf = {loss_av_hf.numpy():.4e}], "
                f"Test-error[coordinate_lf = {relative_error_lf.numpy():.4e}, "
                f"coordinate_hf = {relative_error_hf.numpy():.4e}, "
                f"A_hf = {Ae.numpy():.4e}, "
                f"V_hf = {Ve.numpy():.4e}], "
                f"time: {elapsed_time:.2f}s")
    u_lf_pred_all, u_hf_pred_all, _, _ = rbcnn(x_lf_all_tsr, x_hf_all_tsr)
    u_lf_pred_all = data.decode(u_lf_pred_all)
    u_hf_pred_all = data.decode(u_hf_pred_all)
    np.savez('./result/mfS_predict.npz', predict_rbc_lf = u_lf_pred_all , predict_rbc_hf = u_hf_pred_all)
    

if __name__ == '__main__':
    main()