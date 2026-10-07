import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class DynamicNumPyModel:
    def __init__(self, npz_path):
        data = np.load(npz_path)
        
        # Sort keys numerically (arr_0, arr_1, ..., arr_11)
        sorted_keys = sorted(data.files, key=lambda k: int(k.split('_')[1]))
        
        # Pair weights and biases sequentially: [(w0, b0), (w1, b1), ...]
        self.layers = []
        for i in range(0, len(sorted_keys), 2):
            w = data[sorted_keys[i]]
            b = data[sorted_keys[i+1]]
            self.layers.append((w, b))

    def forward(self, scores, mask):
        mask = np.atleast_2d(mask)
        scores = np.atleast_2d(scores)

        batch_size = mask.shape[0]

        # 1. Dynamic Input Prep (Reshape & Multiply)
        m = np.expand_dims(mask, axis=-1)
        s = np.expand_dims(scores, axis=-1)
        p = m * s
        feat = np.concatenate([m, s, p], axis=-1)

        # 2. Process Layer 0: EinsumDense (identified dynamically by 3D weights)
        w_einsum, b_einsum = self.layers[0]
        h = np.einsum("bcf,cfh->bch", feat, w_einsum) + b_einsum
        h = relu(h)

        # 3. Flatten for standard Dense layers
        x = h.reshape(batch_size, -1)
        
        # 4. Iterate dynamically through remaining Dense layers
        num_dense_layers = len(self.layers) - 1
        
        for i, (w, b) in enumerate(self.layers[1:]):
            #print(np.shape(w))
            # Compute linear pass
            x = x@w + b
            
            # Apply activation based on layer position
            is_last_layer = (i == num_dense_layers - 1)
            if is_last_layer:
                x = softmax(x, axis=-1)
            else:
                x = relu(x)
                
        return x

    def plot_evaluate_model(self, eval_data, num_samples):

        x_data = np.tile(np.linspace(0, 374, 375), (num_samples, 1))
        predicted_data = model.forward(*eval_data)

        for i in range(num_samples):
            # print(plot_input)
            # print(plot_input['mask'][i])
            # print(plot_input['scores'][i])
            #print(calculate_win_prob(np.array([predicted_data[i], predicted_data[i + 1]])))
            # plt.plot(plot_x_prob[i], plot_y_prob[i], '.k')
            # plt.plot(plot_x_prob[i+1], plot_y_prob[i+1], '.b')
            plt.plot(x_data[i], predicted_data[i], 'o')
            #plt.plot(x_data[i], predicted_data_tf[i], '.')
            # plt.plot(x_data[i+1], predicted_data[i+1], '-b')
            plt.show()

    def calculate_win_prob(P):
        n, nb = P.shape
        P = P / np.sum(P, axis=1, keepdims=True)  # normalize pdf
        cdf = np.cumsum(P, axis=1)  # calculate cdf with ties
        cdf_strict = np.pad(cdf[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=0)
        out = np.zeros(n)
        for i in range(len(P)):
            # others = np.delete(cdf, i, axis=0) # remove current player
            others_strict = np.delete(cdf_strict, i, axis=0)  # remove current player

            # prod_others = np.prod(others, axis=0) # cdf of all players scoring lower than x
            prod_others_strict = np.prod(others_strict, axis=0)  # cdf of all players scoring lower than x

            wins_and_ties = P[i] * prod_others_strict
            out[i] = np.sum(wins_and_ties)
        out = out / np.sum(out)  # normalize output
        return out

def calculate_win_prob(P):
    n, nb = P.shape
    P = P / np.sum(P, axis=1, keepdims=True) #normalize pdf
    cdf = np.cumsum(P, axis=1) #calculate cdf with ties
    cdf_strict = np.pad(cdf[:,:-1], ((0,0),(1,0)), mode='constant', constant_values=0)
    out = np.zeros(n)
    for i in range(len(P)):
        #others = np.delete(cdf, i, axis=0) # remove current player
        others_strict = np.delete(cdf_strict, i, axis=0) # remove current player
          
        #prod_others = np.prod(others, axis=0) # cdf of all players scoring lower than x
        prod_others_strict = np.prod(others_strict, axis=0) # cdf of all players scoring lower than x

        wins_and_ties = P[i] * prod_others_strict
        out[i] = np.sum(wins_and_ties)

    out = out/np.sum(out) #normalize output
    return out

if __name__ == "__main__":
    model = DynamicNumPyModel("App\\NN_model_weights.npz")
    eval_data = [np.expand_dims([0, 0, 0, 0, 25, 30, 0, 0, 0, 0, 0, 0, 0, ], axis=-1), np.expand_dims([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1], axis=-1)]
    print(np.shape(eval_data))
    model.plot_evaluate_model(np.array(eval_data), 1)

