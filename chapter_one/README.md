Simple ML-Algorithms for Classification

formal Definition of an artificial neuron
binary problem with outputs 1 and -1
decision function phi(z) where phi(z) = 1 if z >= theta, phi(z) = -1 else
whereas z = w1 * x1 + w2 * x2 +... wm * xm with x observations and w weights.

by transforming and def. w0 = - theta, x0 = 1 one gets
z = w0 * x0 + ... + wm * xm
and 
phi(z) = 1 if z >= 0, -1 else
w0 is the so-called bias unit

Perceptron learning rule
originally by Rosenblatt´s threshold perceptron and the MCP neuron.
Algorithm:
- Initialize weights to 0 or random number
- For each observation x_i:
    - Compute y_hat
    - update the weights

new weight vector w in each step is a result of 
w_j = w_j + ∆w_j
whereas ∆w_j = nü * (y_i - y_hat_i) * x_j_i
with j is the j-th weight/dimension/feature and i is the i-th observation and i = 0 is the constant.
by that (∆w0,...,∆wm) = nü * (y_i - y_hat_i) * (x_0_i,....,x_m_i)
nü is the learning rate.

Example: For a two-dimensional dataset:
∆w0 = nü * (y_i - y_hat_i)
∆w1 = nü * (y_i - y_hat_i) * x_1_i
∆w2 = nü * (y_i - y_hat_i) * x_2_i
