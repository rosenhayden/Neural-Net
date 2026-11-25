import math
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="Train a model from x.npy and y.npy. with multinomial logistic regression.",action="store_true")
parser.add_argument("--predict", help="Run prediction on a dev set using a trained model.",action="store_true")
parser.add_argument("--eval", help="Evaluate a trained model on a dev set (prints accuracy).",action="store_true")
parser.add_argument("--trainx", help="Path to inputs x.npy.")
parser.add_argument("--trainy", help="Path to inputs y.npy.")
parser.add_argument("--model", help="Path to model file to be written (train) or read (predict/eval).")
parser.add_argument("--preds", help="Path to write predictions (predict).")
parser.add_argument("--devx", help="Path to inputs npy file")
parser.add_argument("--devy", help="Path to inputs npy file")
parser.add_argument("--lr", help="Learning rate for gradient descent (train).", type=float)
parser.add_argument("--mb", help="minibatch size, defaults 32.", type=int, default = 32)
parser.add_argument("--patience", help="Early stoppind patience number is the number of dev evals without improvement allowed, default is 500", type=int, default = 10)
parser.add_argument("--dev_freq", help="evaluate on the dev set every dev_freq number of steps,default 500", type=int, default = 500)
parser.add_argument("--verbose", help="If flagged minibatch loss will be printed", action = 'store_true')
parser.add_argument("--regression", help= "Use identity output activation with squared error loss" , action="store_true")
parser.add_argument("--classification", help= "Use softmax output activation with cross-entropy loss" , action="store_true")
parser.add_argument("--max-steps", help="Max mini batch grad, default 5000.",type=int, default=5000)
parser.add_argument("--activation", help="{relu,tanh,sigmoid} activation, default 'relu'", default = "relu")
parser.add_argument("--L", help="Number of hidden layers. comma delimited list of nunits by nlayers. Ex: '32x3', '128x2,16x1,128x2'", default = "")
parser.add_argument("--C", help="Number of classes (classification) or output dimension (regression).",type=int)
args = parser.parse_args()

def train(X,Y,devX,devY, modelName, nk, iters, batchSize,devFreq, verbose,patience,task,activation, layers, C):
    if task == "classification":
        X, Y = np.load(X), np.load(Y).astype(int)
        n =np.shape(Y)[0]
        devX,devY = np.load(devX), np.load(devY).astype(int)
        y = Y
        Yhot = np.zeros((n,C))
        Yhot[np.arange(n),y] = 1
    else:
        X, Y = np.load(X), np.load(Y)
        devX,devY = np.load(devX), np.load(devY)
        n = np.shape(Y)[0]
    if devX.shape[1] != X.shape[1]:
        print("Shape Error in devX/X")
        return

    D = np.shape(X)[1]
    W,B= glorot(D, layers, C)

    update = 0
    epoch = math.ceil(iters / (n // batchSize)) + 1
    bestDevLoss = float('inf')
    p=patience
    f = str(activation)
    g = str(task)

    l = len(layers)+1
    for e in range(1,epoch):
        shuffled = np.random.permutation(len(X))
        for i in range(0, len(X) - (len(X) % batchSize), batchSize):
            batchPos = shuffled[i: batchSize+i]
            if task == "classification":
                curX, curY = X[batchPos],Yhot[batchPos]
            else:
                curX, curY = X[batchPos],Y[batchPos]
            n = curX.shape[0]
            one = np.ones(n)

            A = curX
            pTild,activations,preActivation = forwardPassTrain(A,W,B,f,g,l)

            errSig = (pTild - curY)
            for k in range(len(W)-1, -1, -1):
                dW = np.transpose(activations[k]) @  errSig  / n
                dB = np.transpose(one) @ errSig / n
                if k > 0:
                    errSig = errSig @ np.transpose(W[k]) * derivativeActivations[activation](preActivation[k-1])
                W[k] -= nk * dW
                B[k] -= nk * dB

            update += 1
            if verbose:
                if task == "regression":
                    L = 1/curY.shape[0] * np.linalg.norm(pTild-curY) ** 2
                else:
                    L= -np.sum(curY * np.log(pTild)) / n #probably unstable
                print(f"{L:.3f} batch, update {update}, epoch {e}")
            if update % devFreq == 0:
                A = devX
                prediction,dev,m = forwardPassTrain(A,W,B,f,g,l)
                dev = prediction
                if task == "classification":
                    dLoss = -np.mean(np.log(dev[np.arange(len(devY)),devY])) #CEL
                    acc = np.mean(np.argmax(dev,axis=1) == devY)
                    print(f"dev loss={dLoss:.3f} acc={acc:.3f} (as of update {update}, epoch {e})")
                else:
                    dLoss = 1/devY.shape[0] * np.linalg.norm(prediction-devY) ** 2 #SEL
                    r2 = 1 - np.sum((prediction-devY)**2)/ np.sum((devY-devY.mean(axis=0))**2)
                    print(f"dev loss={dLoss:.3f} R2={r2:.3f} (as of update {update}, epoch {e})")
                if dLoss < bestDevLoss:
                    modelSave(W,B,task,activation,modelName)
                    bestDevLoss = dLoss
                    p = patience
                else:
                    p -= 1
                    if p <= 0:
                        return
            if update == iters:
                if bestDevLoss == float('inf'):
                    modelSave(W,B,task,activation,modelName)

                return

def modelSave(W,B, task, activation, modelName):
    splitWeights = {}
    splitBias = {}
    layers = []
    for layer in range(len(W)):
        splitWeights[f'W_{layer+1}'] = W[layer]
        splitBias[f'b_{layer+1}'] = B[layer]
        layers.append(W[layer].shape[1])
    np.savez(modelName, **splitWeights ,**splitBias, activation=activation,task=task, layer_sizes=layers)

def glorot(D, layers, output):
    L = [D] + list(layers) + [output]
    W,B=[],[]
    for i in range(len(L)-1):
        L_km1, L_k = L[i], L[i+1]
        sampleRange = np.sqrt(6/ (L_km1 + L_k))
        W.append(np.random.uniform(-sampleRange,sampleRange,size=(L_km1,L_k)))
        B.append(np.zeros(L_k))
    return W,B

def evaluate(model, devX, devY):
    m = np.load(model)
    task = m["task"]
    Y = np.load(devY)
    prediction = forwardPass(model,devX)
    if task == "regression":
        r2 = 1 - np.sum((prediction-Y)**2)/ np.sum((Y-Y.mean(axis=0))**2)
        print(f"{r2:.3f}")
    else:
        print(f"{(prediction.argmax(axis=1) == Y).mean():.3f}")

def predict(model, X, predictionName):
    np.save(predictionName, forwardPass(model,X))

def forwardPass(model, X):
    A = np.load(X)
    model = np.load(model)
    f = str(model["activation"])
    g = str(model['task'])
    layers = len(model["layer_sizes"])
    z=None
    for layer in range(1, layers+1):
        W = model[f'W_{layer}']
        b = model[f'b_{layer}']
#        print(f"A size {A.shape} W size {W.shape}")
        if A.shape[1] != W.shape[0]:
            print("Forward pass inner dim shape missmatch of Z")
            return
        z = A @ W + b
        A = activationFunctions[f](z)
    A = activationFunctions[g](z)
    return A

def forwardPassTrain(A,W,B,f,g,l):
    z=None
    activations= [A]
    preActivation = []
    for layer in range(l):
        Wk = W[layer]
        bk = B[layer]
        z = A @ Wk + bk
        A = activationFunctions[f](z)
        activations.append(A)
        preActivation.append(z)
    return activationFunctions[g](z), activations, preActivation


def relu(X):
    return np.maximum(0,X)
def linear(X):
    return X
def tanh(X):
    ex, nex =np.exp(X), np.exp(-X)
    return (ex - nex) / (ex + nex)
def drelu(X):
    return np.where(X>0, 1.0, 0.0)
def dtanh(X):
    return 1 - tanh(X) ** 2
def dsigmoid(X):
    sig = sigmoid(X)
    return sig * (1-sig)
def sigmoid(X):
    z= np.exp(-np.abs(X))
    return np.where(X>=0, 1/ (1+z), z/(1+z))

def softmax(x):
    ex=np.exp(x - np.max(x,keepdims=True,axis=-1)) # Page 186 goodfellow.
    y = ex / np.sum(ex, keepdims=True, axis=-1)
    return y
activationFunctions = {
    "relu" : relu,
    "regression" : linear,
    "tanh" : tanh,
    "sigmoid" : sigmoid,
    "classification" : softmax
}
derivativeActivations = {
    "tanh" : dtanh,
    "relu" : drelu,
    "sigmoid" : dsigmoid,
}


def buildLayers(L):
    layers = []
    if L == "":
        return []
    for x in L.split(","):
        a, b = x.strip().lower().split("x")
        a,b = int(a),int(b)
        layers += [a] * b
    return layers

if args.predict:
    predict(args.model, args.devx, args.preds)
elif args.eval:
    evaluate(args.model,args.devx,args.devy)
else:
    if args.classification:
        task = "classification"
    else:
        task = "regression"
    layers = buildLayers(args.L)
    train(args.trainx,args.trainy,args.devx, args.devy , args.model, args.lr, args.max_steps, args.mb,args.dev_freq, args.verbose,args.patience,task,args.activation, layers, args.C)
