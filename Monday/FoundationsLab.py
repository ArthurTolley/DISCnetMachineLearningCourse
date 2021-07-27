#%%
import matplotlib.pyplot as plt
import numpy as np

#%%
n_data = 200

# Two-class problem, distinct means, equal covariance matrices

m1 = [[0, 5]]
m2 = [[5, 0]]
C = [[2, 1], [1, 2]]

# Set up the data by generating isotrop Gaussians and rotating them

A = np.linalg.cholesky(C)

U1 = np.random.randn(n_data, 2)
X1 = U1 @ A.T + m1

U2 = np.random.randn(n_data, 2)
X2 = U2 @ A.T + m2

#%%
fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(X1[:, 0], X1[:, 1], c='c')
ax.scatter(X2[:, 0], X2[:, 1], c='m')
ax.grid(True)

# %%
# Concatenate data from two classes into one array
X = np.concatenate((X1, X2), axis=0)

# Set up targets (labels):
#  We set +1 and -1 as labels to indicate the two classes
labelPos = np.ones(n_data)
labelNeg = -1.0 * np.ones(n_data)
y = np.concatenate((labelPos, labelNeg))

# Partioning the data into training and testing sets
rIndex = np.random.permutation(2*n_data)
Xr = X[rIndex,]
yr = y[rIndex]

#  Training and testing sets (half half)
X_train = Xr[0:n_data]
y_train = yr[0:n_data]

X_test = Xr[n_data:2*n_data]
y_test = yr[n_data:2*n_data]

#  print the shapes just to check
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

Ntrain = n_data;
Ntest = n_data;

# Calculating the percentage of correctly classified examples
def PercentCorrect(inputs, targets, weights):
    N = len(targets)
    nCorrect = 0
    for n in range(N):
        OneInput = inputs[n,:]
        if (targets[n] * np.dot(OneInput, weights) > 0):
            nCorrect += 1
    return 100*nCorrect/N

# Iterative error correcting learning
#  Perceptron learning loop

#  Random initialization of weights
w = np.random.randn(2)
print(w)

#  What is the performance with the initial random weights?
print('Initial Percentage Correct: {}'.format(PercentCorrect(X_train,
                                                             y_train,
                                                             w)))

#  Fixed number of iterations
MaxIter = 1000

#  Learning rate
alpha = 0.002

#  Space to save answers for plotting
P_train = np.zeros(MaxIter)
P_test = np.zeros(MaxIter)

#  Main Loop
for iter in range(MaxIter):

    # Select data at random
    r = np.floor(np.random.rand()*Ntrain).astype(int)
    x = X_train[r,:]

    # If it is misclassified, update weights
    if (y_train[r] * np.dot(x, w) < 0):
        w += alpha * y_train[r] * x

    # Evaluate training and test performaces for plotting
    P_train[iter] = PercentCorrect(X_train, y_train, w);
    P_test[iter] = PercentCorrect(X_test, y_test, w);

print('Percentage Correct After Training: {} {}'.format(
                                                        PercentCorrect(X_train,
                                                                       y_train,
                                                                       w),
                                                        PercentCorrect(X_test,
                                                                       y_test,
                                                                       w)))
# %%
print('Percentage Correct After Training: {} {}'.format(
                                                        PercentCorrect(X_train,
                                                                       y_train,
                                                                       w),
                                                        PercentCorrect(X_test,
                                                                       y_test,
                                                                       w)))
# Plot learning curves
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(MaxIter), P_train, 'b', label='Training')
ax.plot(range(MaxIter), P_test, 'r', label='Test')
ax.grid(True)
ax.legend()
ax.set_title('Perceptron Learning')
ax.set_ylabel('Training and Test Accuracties', fontsize=14)
ax.set_xlabel('Iteration', fontsize=14)

# %%
# ScikitLearn Perceptron algorithm comparison
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

model = Perceptron()
model.fit(X_train, y_train)

yh_train = model.predict(X_train)
print('Accuracy on training set: {}'.format(accuracy_score(yh_train,
                                                           y_train)*100))

yh_test = model.predict(X_test)
print('Accuracy on testing set: {}'.format(accuracy_score(yh_test,
                                                           y_test)*100))

if (accuracy_score(yh_test, y_test) > 0.99):
    print("Perfect classification on separable dataset!")

# %%
# Consider the problem with means at m1 = [2.5, 2.5], and m2 = [10.0, 10.0]
#  with equal covariance matrices.
#  Does the perceptron as implemented solve this problem?
#  If not, what modification is needed to help solve this problem?

n_data = 200

m1 = [[2.5, 2.5]]
m2 = [[10.0, 10.0]]
C = [[2, 1], [1, 2]]

A = np.linalg.cholesky(C)
U1 = np.random.randn(n_data, 2)
X1 = U1 @ A.T + m1
U2 = np.random.randn(n_data, 2)
X2 = U2 @ A.T + m2

fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(X1[:, 0], X1[:, 1], c='c')
ax.scatter(X2[:, 0], X2[:, 1], c='m')
ax.grid(True)

# Concatenate data from two classes into one array
X = np.concatenate((X1, X2), axis=0)

# Set up targets (labels):
#  We set +1 and -1 as labels to indicate the two classes
labelPos = np.ones(n_data)
labelNeg = -1.0 * np.ones(n_data)
y = np.concatenate((labelPos, labelNeg))

# Partioning the data into training and testing sets
rIndex = np.random.permutation(2*n_data)
Xr = X[rIndex,]
yr = y[rIndex]

#  Training and testing sets (half half)
X_train = Xr[0:n_data]
y_train = yr[0:n_data]

X_test = Xr[n_data:2*n_data]
y_test = yr[n_data:2*n_data]

Ntrain = n_data;
Ntest = n_data;

w = np.random.randn(2)

#  What is the performance with the initial random weights?
print('Initial Percentage Correct: {}'.format(PercentCorrect(X_train,
                                                             y_train,
                                                             w)))

#  Space to save answers for plotting
P_train = np.zeros(MaxIter)
P_test = np.zeros(MaxIter)

#  Main Loop
for iter in range(MaxIter):

    # Select data at random
    r = np.floor(np.random.rand()*Ntrain).astype(int)
    x = X_train[r,:]

    # If it is misclassified, update weights
    if (y_train[r] * np.dot(x, w) < 0):
        w += alpha * y_train[r] * x

    # Evaluate training and test performaces for plotting
    P_train[iter] = PercentCorrect(X_train, y_train, w);
    P_test[iter] = PercentCorrect(X_test, y_test, w);

print('Percentage Correct After Training: {} {}'.format(
                                                        PercentCorrect(X_train,
                                                                       y_train,
                                                                       w),
                                                        PercentCorrect(X_test,
                                                                       y_test,
                                                                       w)))
# %%
print('Percentage Correct After Training: {} {}'.format(
                                                        PercentCorrect(X_train,
                                                                       y_train,
                                                                       w),
                                                        PercentCorrect(X_test,
                                                                       y_test,
                                                                       w)))
# Plot learning curves
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(MaxIter), P_train, 'b', label='Training')
ax.plot(range(MaxIter), P_test, 'r', label='Test')
ax.grid(True)
ax.legend()
ax.set_title('Perceptron Learning')
ax.set_ylabel('Training and Test Accuracties', fontsize=14)
ax.set_xlabel('Iteration', fontsize=14)

print('It doesn\'t do a very good job')
# %%

# We add another columns of biases
O = np.ones((2*n_data, 1))
X = np.append(X, O, axis=1)

# Set up targets (labels):
#  We set +1 and -1 as labels to indicate the two classes
labelPos = np.ones(n_data)
labelNeg = -1.0 * np.ones(n_data)
y = np.concatenate((labelPos, labelNeg))

# Partioning the data into training and testing sets
rIndex = np.random.permutation(2*n_data)
Xr = X[rIndex,]
yr = y[rIndex]

#  Training and testing sets (half half)
X_train = Xr[0:n_data]
y_train = yr[0:n_data]

X_test = Xr[n_data:2*n_data]
y_test = yr[n_data:2*n_data]

Ntrain = n_data;
Ntest = n_data;

w = np.random.randn(3)

#  What is the performance with the initial random weights?
print('Initial Percentage Correct: {}'.format(PercentCorrect(X_train,
                                                             y_train,
                                                             w)))

#  Space to save answers for plotting
P_train = np.zeros(MaxIter)
P_test = np.zeros(MaxIter)

#  Main Loop
for iter in range(MaxIter):

    # Select data at random
    r = np.floor(np.random.rand()*Ntrain).astype(int)
    x = X_train[r,:]

    # If it is misclassified, update weights
    if (y_train[r] * np.dot(x, w) < 0):
        w += alpha * y_train[r] * x

    # Evaluate training and test performaces for plotting
    P_train[iter] = PercentCorrect(X_train, y_train, w);
    P_test[iter] = PercentCorrect(X_test, y_test, w);

print('Percentage Correct After Training: {} {}'.format(
                                                        PercentCorrect(X_train,
                                                                       y_train,
                                                                       w),
                                                        PercentCorrect(X_test,
                                                                       y_test,
                                                                       w)))
# %%
print('Percentage Correct After Training: {} {}'.format(
                                                        PercentCorrect(X_train,
                                                                       y_train,
                                                                       w),
                                                        PercentCorrect(X_test,
                                                                       y_test,
                                                                       w)))
# Plot learning curves
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(MaxIter), P_train, 'b', label='Training')
ax.plot(range(MaxIter), P_test, 'r', label='Test')
ax.grid(True)
ax.legend()
ax.set_title('Perceptron Learning')
ax.set_ylabel('Training and Test Accuracties', fontsize=14)
ax.set_xlabel('Iteration', fontsize=14)

print('This one does much better!')


# %%
