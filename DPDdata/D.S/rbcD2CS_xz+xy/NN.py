import tensorflow as tf
import numpy as np

class rbc_mfNN(tf.keras.Model):
    def __init__(self, all_layers, all_x, rbcinfo, decd, batch_size = 64, name = None):
        super().__init__(name = name)

        # rbc paras
        self.x_lf_all = all_x[0]
        self.x_hf_all = all_x[1]

        self.id = rbcinfo[0]
        self.tri = rbcinfo[1]
        self.av_lf = rbcinfo[2]
        self.av_hf = rbcinfo[3]
        
        self.d_mean = decd[0]
        self.d_std = decd[1]

        # NN paras
        self.layers_c = all_layers[0]
        self.layers_lf = all_layers[1]
        self.layers_hf_nl = all_layers[2]
        self.layers_hf_l = all_layers[3]

        self.batch_size = batch_size


        self.conv_layers = []
        for i in range(len(self.layers_c) - 1):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(filters=self.layers_c[i + 1], kernel_size=3, 
                                       strides=1, padding="same", use_bias=True, activation=None, 
                                       kernel_initializer= 'glorot_normal',
                                       name=f"ConvF{i+1}")
            )
        self.pooling_layer = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')

        self.W_lf=[]
        layers_s = self.layers_lf
        for i in range(len(layers_s) - 1):
            in_dim = layers_s[i]
            out_dim = layers_s[i + 1]
            
            # Xavier initialization std
            std_dv = np.sqrt(2.0 / (in_dim + out_dim))
            
            w_init = tf.random.normal([in_dim, out_dim], 
                                      mean=0.0, stddev=std_dv, dtype=tf.float32)
            w_var = tf.Variable(w_init, trainable=True, name=f'Wlf{i+1}')
            
            b_init = tf.zeros([out_dim], dtype=tf.float32)
            b_var = tf.Variable(b_init, trainable=True, name=f'blf{i+1}')
            
            self.W_lf.append(w_var)
            self.W_lf.append(b_var)

        self.W_hf_nl=[]
        layers_s = self.layers_hf_nl
        for i in range(len(layers_s) - 1):
            in_dim = layers_s[i]
            out_dim = layers_s[i + 1]
            
            # Xavier initialization std
            std_dv = np.sqrt(2.0 / (in_dim + out_dim))
            
            w_init = tf.random.normal([in_dim, out_dim], 
                                      mean=0.0, stddev=std_dv, dtype=tf.float32)
            w_var = tf.Variable(w_init, trainable=True, name=f'Whfnl{i+1}')
            
            b_init = tf.zeros([out_dim], dtype=tf.float32)
            b_var = tf.Variable(b_init, trainable=True, name=f'bhfnl{i+1}')
            
            self.W_hf_nl.append(w_var)
            self.W_hf_nl.append(b_var)
        
        self.W_hf_l=[]
        layers_s = self.layers_hf_l
        for i in range(len(layers_s) - 1):
            in_dim = layers_s[i]
            out_dim = layers_s[i + 1]
            
            # Xavier initialization std
            std_dv = np.sqrt(2.0 / (in_dim + out_dim))
            
            w_init = tf.random.normal([in_dim, out_dim], 
                                      mean=0.0, stddev=std_dv, dtype=tf.float32)
            w_var = tf.Variable(w_init, trainable=True, name=f'Whfl{i+1}')
            
            b_init = tf.zeros([out_dim], dtype=tf.float32)
            b_var = tf.Variable(b_init, trainable=True, name=f'bhfl{i+1}')
            
            self.W_hf_l.append(w_var)
            self.W_hf_l.append(b_var)


    @property
    def trainable_variables(self):
        return (self.W_lf + self.W_hf_nl + self.W_hf_l+[v for layer in self.conv_layers for v in layer.trainable_variables])
    

    def call(self, x_lf, x_hf):
        # lf
        A = tf.expand_dims(x_lf, axis=1)  # Shape: (batch_size, 1, 3)
        for i, conv_layer in enumerate(self.conv_layers):
            A = conv_layer(A)  # Apply convolution
            A = tf.keras.layers.LeakyReLU(negative_slope=0.01)(A)  # Use LeakyReLU 
        A = self.pooling_layer(A)  # Pooling applied **once**
        u1 = tf.keras.layers.Flatten()(A) 

        ul_1 = u1
        for i in range(len(self.layers_lf)-2):    
            z = tf.add(tf.matmul(ul_1, self.W_lf[2*i]), self.W_lf[2*i+1])
            ul_1 = tf.nn.tanh(z)  
        u_lf = tf.add(tf.matmul(ul_1, self.W_lf[-2]), self.W_lf[-1]) 


        A = tf.expand_dims(x_hf, axis=1)  # Shape: (batch_size, 1, 3)
        for i, conv_layer in enumerate(self.conv_layers):
            A = conv_layer(A)  # Apply convolution
            A = tf.keras.layers.LeakyReLU(negative_slope=0.01)(A)  # Use LeakyReLU 
        A = self.pooling_layer(A)  # Pooling applied **once**
        u2 = tf.keras.layers.Flatten()(A) 

        ul_2 = u2
        for i in range(len(self.layers_lf)-2):    
            z = tf.add(tf.matmul(ul_2, self.W_lf[2*i]), self.W_lf[2*i+1])
            ul_2 = tf.nn.tanh(z)  
        u_lf_hfx = tf.add(tf.matmul(ul_2, self.W_lf[-2]), self.W_lf[-1]) 

        # hf
        xhf = tf.concat([x_hf, u_lf_hfx], axis=-1)

        uh_1 = xhf
        for i in range(len(self.layers_lf)-2):    
            z = tf.add(tf.matmul(uh_1, self.W_hf_nl[2*i]), self.W_hf_nl[2*i+1])
            uh_1 = tf.nn.tanh(z)  
        u_hf_nl = tf.add(tf.matmul(uh_1, self.W_hf_nl[-2]), self.W_hf_nl[-1])

        uh_2 = xhf
        u_hf_l = tf.add(tf.matmul(uh_2, self.W_hf_l[-2]), self.W_hf_l[-1])
        
        u_hf = u_hf_nl + u_hf_l

        return u_lf, u_hf, u_hf_nl, u_hf_l
    

    def compute_av_tf(self, coord_array, triangle_array):
        # Extract IDs and XYZ coordinates
        coord_ids = tf.cast(coord_array[:, 0], tf.int32)  # Shape (N,)
        coords = coord_array[:, 1:]  # Shape (N, 3)
        # Create an ID-to-index mapping
        id_to_index = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(coord_ids, tf.range(tf.shape(coords)[0])),
            default_value=-1
        )
        # Get indices of triangle vertices
        tri_indices = id_to_index.lookup(tf.cast(triangle_array[:, 1:], tf.int32))  # Shape (M, 3)
        # Gather points based on triangle indices
        p1 = tf.gather(coords, tri_indices[:, 0])  # Shape (M, 3)
        p2 = tf.gather(coords, tri_indices[:, 1])  # Shape (M, 3)
        p3 = tf.gather(coords, tri_indices[:, 2])  # Shape (M, 3)
        # Compute vectors v1 and v2
        v1 = p2 - p1
        v2 = p3 - p1
        # Compute normal vectors using cross product
        normals = tf.linalg.cross(v1, v2)  # Shape (M, 3)
        # Compute triangle areas
        triangle_areas = 0.5 * tf.norm(normals, axis=1)
        total_area = tf.reduce_sum(triangle_areas)
        # Compute volume using dot and cross products
        tetra_volumes = tf.reduce_sum(p1 * tf.linalg.cross(p2, p3), axis=1) / 6.0
        total_volume = tf.abs(tf.reduce_sum(tetra_volumes))
        return total_area, total_volume
    

    def loss_rbc(self, x_lf, x_hf, u_lf, u_hf):
        u_lf_pred, u_hf_pred, _, u_hf_l_pred  = self(x_lf, x_hf)

        loss_lf = tf.reduce_mean(tf.norm(u_lf_pred - u_lf, axis = 1)/(tf.norm(u_lf, axis = 1) + 1e-8))
        loss_hf = tf.reduce_mean(tf.norm(u_hf_pred - u_hf, axis = 1)/(tf.norm(u_hf, axis = 1) + 1e-8))
        loss_hf_l = tf.reduce_mean(tf.norm(u_hf_l_pred - u_hf, axis = 1)/(tf.norm(u_hf, axis = 1) + 1e-8))

        u_lf_all_pred, u_hf_all_pred, _, _ = self(self.x_lf_all, self.x_hf_all)
        
        u_lf_all_ = u_lf_all_pred*(self.d_std + 1.0e-6) + self.d_mean
        u_lf_all = tf.concat([self.id, u_lf_all_], axis=1)
        At1, Vt1 = self.compute_av_tf(u_lf_all, self.tri)
        A_1, V_1 = self.av_lf[0], self.av_lf[1]
        loss_av_lf = tf.square(At1 - A_1)/(A_1 + 1e-8) + tf.square(Vt1 - V_1)/(V_1 + 1e-8)
        
        u_hf_all_ = u_hf_all_pred*(self.d_std + 1.0e-6) + self.d_mean
        u_hf_all = tf.concat([self.id, u_hf_all_], axis=1)
        At2, Vt2 = self.compute_av_tf(u_hf_all, self.tri)
        A_2, V_2 = self.av_hf[0], self.av_hf[1]
        loss_av_hf = tf.square(At2 - A_2)/(A_2 + 1e-8) + tf.square(Vt2 - V_2)/(V_2 + 1e-8)

        A_err = (At2 - A_2)/A_2 
        V_err = (Vt2 - V_2)/V_2 

        regularizer = tf.keras.regularizers.L2(0.001)
        loss_l2 = tf.add_n([regularizer(w_) for w_ in self.W_hf_nl]) # L2-loss for non-linear hf NN added, to prevent overfitting

        loss = loss_lf + 0.01*loss_av_lf + 0.1*(loss_hf + 0.00*loss_av_hf + loss_l2 + loss_hf_l)
        
        return loss, loss_lf, loss_av_lf, loss_hf, loss_av_hf, tf.abs(A_err), tf.abs(V_err)