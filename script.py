import torch
import torch.nn as nn
import numpy as np
import argparse
import csv
import os
from copy import deepcopy

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input_size', type=int, default=1, required=False)
parser.add_argument('--hidden_size', type=int, default=3, required=False)
parser.add_argument('--num_layers', type=int, default=1, required=False)
parser.add_argument('--batch_size', type=int, default=1, required=False)
parser.add_argument('--T_train', type=int, default=5, required=False)
parser.add_argument('--T_test', type=int, default=25, required=False)
parser.add_argument('--num_tasks', type=int, default=1000, required=False)
parser.add_argument('--num_epochs', type=int, default=1000, required=False)
parser.add_argument('--objective', type=str, choices=["mimick", "perf"], default="mimick", required=False)
parser.add_argument('--label_as_input', default=False, required=False)
parser.add_argument('--num_runs', type=int, default=1, required=False)
parser.add_argument('--param_range', type=float, default=5, required=False)
parser.add_argument('--xrange', type=float, default=5, required=False)
parser.add_argument('--learning_rate', type=float, default=1e-2, required=False)
parser.add_argument('--evaluate_model', type=str, default=None, required=False)
parser.add_argument('--debug', action="store_true", default=False)

args = parser.parse_args()
print(args)

if type(args.label_as_input) == type("string"):
    args.label_as_input = args.label_as_input == "True"

assert not (args.label_as_input and args.objective == "perf"), (
    "Cannot pass ground-truth output as input as that would be cheating for this performance maximization and the output is constants over time"
)  
# Train an LSTM to perform backprop on a distribution of tasks
if args.label_as_input:
    args.input_size = args.input_size + 1

if not args.evaluate_model is None:
    assert os.path.isdir("./results/"+args.evaluate_model), "Could not find specified model"
    try:
        inp_size, hsize, nlayers, bsize, T_train, T_test, ntasks, labelinput, objective = args.evaluate_model.split("-")[1:]
    except:
        pass

    try: 
        inp_size, hsize, nlayers, bsize, T_train, T_test, ntasks, labelinput, objective, learning_rate = args.evaluate_model.split("-")[1:]
    except:
        raise ValueError("Could not parse the evaluate_model string")

    args.input_size = int(inp_size)
    args.hidden_size = int(hsize)
    args.num_layers = int(nlayers)
    args.batch_size = int(bsize)
    args.T_train = int(T_train)
    args.T_test = int(T_test)
    args.label_as_input = labelinput == "True"
    args.learning_rate = float(learning_rate)
    num_tasks = ntasks
else:
    num_tasks = args.num_tasks
    

RDIR = "./results/"
TNAME = f"rec-{args.input_size}-{args.hidden_size}-{args.num_layers}-{args.batch_size}-{args.T_train}-{args.T_test}-{num_tasks}-{args.label_as_input}-{args.objective}-{args.learning_rate}"
TDIR = f"{RDIR}{TNAME}/"


if not args.debug:
    for direct in [RDIR, TDIR]:
        if not os.path.isdir(direct):
            os.mkdir(direct)


def fn(x,w,b):
    return w*x+b

class RandomNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1,1)
    
    def forward(self, x):
        return self.lin(x)
    

class GeneralLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x has shape [seq length, batch size, input features]
        # ht has shape [num layers, batch size, hidden dim]
        x, (ht, ct) = self.lstm(x)
        
        # compute output for all hidden states (present in x)
        # x now has shape [seq length, batch size, hidden dim]
        out = self.output(x)
        return out # [seq len, batch size, out dim]
        

args.total_time_steps = args.T_train + args.T_test
torch.manual_seed(1337)

for run in range(args.num_runs):
    ########################################################################
    ### Create training data
    rn = RandomNetwork()
    rn_opt = torch.optim.SGD(rn.parameters(), lr=args.learning_rate, momentum=0)
    X = []
    Y = []
    GT = []
    Ws, Bs = [], []
    for n in range(args.num_tasks):
        # sample from [-param_range, +param_range]
        w, b = -args.param_range + torch.rand(1)*(2*args.param_range), -args.param_range + torch.rand(1)*(2*args.param_range)
        Ws.append(w.item()); Bs.append(b.item())
        randx = -args.xrange + torch.rand(1)*(2*args.xrange) #sample from [-xrange, +xrange]
        X.append(randx.repeat(args.total_time_steps).reshape(-1,1,1))
        GT.append(fn(x=randx, w=w, b=b).repeat(args.total_time_steps).reshape(-1,1,1))
        currY = []
        for t in range(args.total_time_steps):
            preds = rn(randx)
            gt = fn(randx,w,b)
            currY.append(preds.item())
            loss = (preds - gt)**2
            loss.backward()
            rn_opt.step()
            rn_opt.zero_grad()
        currY = torch.Tensor(currY).reshape(-1,1,1)
        Y.append(currY)
    X = torch.cat(X,dim=1)
    Y = torch.cat(Y,dim=1)
    GT = torch.cat(GT,dim=1)
    #######################################################################

    lstm = GeneralLSTM(input_size=args.input_size, hidden_size=args.hidden_size, 
                    num_layers=args.num_layers, output_size=1)
    opt = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    if not args.evaluate_model is None:
        mpath = f"{TDIR}model-{run}.pkl"
        print(f"loading model from {mpath}")
        lstm.load_state_dict(torch.load(mpath))
    
    best_epoch_test_loss = float("inf")
    best_epoch_train_loss = float("inf")
    best_weights = None 
    best_epoch_losslist = None # best losses (list of numpy arrays of size [seq len, batch_size, infeatures])
    best_outdomain_losslist = None

    train_losses=[]
    test_losses=[]
    for epoch in range(args.num_epochs):
        perm = torch.randperm(X.size(1))
        count = 0
        x_itemls = []
        loss_itemls = []
        Wls, Bls = [], []
        for bid in range(0, X.size(1), args.batch_size):
            # stack the batch along the batch dimension 1
            input_batch = X[:,perm[bid:bid+args.batch_size],:] #[seq len, batch size, infeatures]
            output_batch = Y[:,perm[bid:bid+args.batch_size],:]
            gt_batch = GT[:,perm[bid:bid+args.batch_size],:]
            w = Ws[perm[bid]]; b = Bs[perm[bid]]
            
            Wls.append(w)
            Bls.append(b)
            x_itemls.append(input_batch[0].numpy().reshape(-1))

            if args.objective == "mimick":
                if args.label_as_input:
                    init_zeros = torch.zeros(input_batch.size(2)*input_batch.size(1)).reshape(1, input_batch.size(1), input_batch.size(2))
                    shifted_output_batch = torch.cat([init_zeros, output_batch], dim=0) # [seq len + 1, batch size, out dim]
                    input_batch = torch.cat([input_batch, shifted_output_batch[:input_batch.size(0),:,:]], dim=2) #[seq len, batch_size, infeatures+1]

                pred = lstm(input_batch)
                losses = (output_batch-pred)**2 #[seq len, batch_size, infeatures+1]
            else:
                pred = lstm(input_batch)
                losses = (gt_batch-pred)**2
            
            loss_items = losses.clone().detach().numpy()
            loss_itemls.append(loss_items)
            train_loss = losses[:args.T_train,:,:].mean()
            test_loss = losses[args.T_train:,:,:].mean()
            
            # else make no more adjustments and just evaluate
            if args.evaluate_model is None:
                train_loss.backward()
                opt.step()
                opt.zero_grad()

            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            count += 1

        epoch_train_loss = np.mean(train_losses[-count:])
        epoch_test_loss = np.mean(test_losses[-count:])
        print("train loss:", epoch_train_loss, "test loss:", epoch_test_loss)
        if epoch_test_loss < best_epoch_test_loss:
            best_epoch_train_loss = epoch_train_loss
            best_epoch_test_loss = epoch_test_loss
            best_weights = deepcopy(lstm.state_dict())
            best_epoch_losslist = np.array(loss_itemls) 
            best_x_itemls = np.array(x_itemls)
            best_ws = np.array(Wls)
            best_bs = np.array(Bls)


    if not args.debug:
        # Evaluate and save the results
        
        if args.evaluate_model is None:
            prefix = ""
        else:
            prefix = "cross-"

        wfn = f"{TDIR}{prefix}ws-run{run}.npy"
        np.save(wfn, best_ws)

        bfn = f"{TDIR}{prefix}bs-run{run}.npy"
        np.save(bfn, best_bs)

        xfn = f"{TDIR}{prefix}inputs-run{run}.npy"
        np.save(xfn, best_x_itemls)

        model_fn = f"{TDIR}{prefix}model-{run}.pkl"# model file name
        torch.save(best_weights, model_fn) # save best weights for this run
        
        # save detailed loss list for every run separately
        # can be used to extract both train and test losses
        np.save(f"{TDIR}{prefix}detailed_loss-{run}.npy", best_epoch_losslist)
        
        # save test performances
        if run == 0:
            mode = "w+"
        else:
            mode = "a"
        
        # write train loss
        train_fn = f"{TDIR}{prefix}train_perfs.csv"
        with open(train_fn, mode, newline="") as f:
            writer = csv.writer(f)
            writer.writerow([str(best_epoch_train_loss)])
        
        print(f"writing to {TDIR}{prefix}test_perfs.csv")

        test_fn = f"{TDIR}{prefix}test_perfs.csv"
        with open(test_fn, mode, newline="") as f:
            writer = csv.writer(f)
            writer.writerow([str(best_epoch_test_loss)])



    # import matplotlib.pyplot as plt

    # plt.plot(train_losses, label='train', color='red')
    # plt.plot(test_losses, label='test', color='blue')
    # plt.legend()
    # plt.yscale('log')
    # plt.show()