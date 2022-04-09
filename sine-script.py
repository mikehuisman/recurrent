import torch
import torch.nn as nn
import numpy as np
import argparse
import csv
import os
from copy import deepcopy
from collections import OrderedDict

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
parser.add_argument('--fixed_init', default=False, required=False)
parser.add_argument('--num_runs', type=int, default=1, required=False)
parser.add_argument('--xrange', type=float, default=5, required=False)
parser.add_argument('--average_cell', default=False, required=False)
parser.add_argument('--evaluate_model', type=str, default=None, required=False)
parser.add_argument('--debug', action="store_true", default=False)

args = parser.parse_args()
print(args)

if type(args.label_as_input) == type("string"):
    args.label_as_input = args.label_as_input == "True"
args.average_cell = args.average_cell == "True"
args.fixed_init = args.fixed_init == "True"

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
        inp_size, hsize, nlayers, bsize, T_train, T_test, ntasks, labelinput, objective, average_cell = args.evaluate_model.split("-")[1:]
        args.average_cell = average_cell == "True"
    except:
        pass

    try: 
        inp_size, hsize, nlayers, bsize, T_train, T_test, ntasks, labelinput, objective, average_cell, fixed_init = args.evaluate_model.split("-")[1:]
        args.fixed_init = fixed_init == "True"
        args.average_cell = average_cell == "True"
    except:
        raise ValueError("Could not parse the evaluate_model string")

    args.input_size = int(inp_size)
    args.hidden_size = int(hsize)
    args.num_layers = int(nlayers)
    args.batch_size = int(bsize)
    args.T_train = int(T_train)
    args.T_test = int(T_test)
    args.label_as_input = labelinput == "True"


    num_tasks = ntasks
else:
    num_tasks = args.num_tasks
    

RDIR = "./results/"
TNAME = f"sine-{args.input_size}-{args.hidden_size}-{args.num_layers}-{args.batch_size}-{args.T_train}-{args.T_test}-{num_tasks}-{args.label_as_input}-{args.objective}-{args.average_cell}-{args.fixed_init}"
TDIR = f"{RDIR}{TNAME}/"


if not args.debug:
    for direct in [RDIR, TDIR]:
        if not os.path.isdir(direct):
            os.mkdir(direct)


def fn(x,a,p):
    return a*torch.sin(x-p)

class SineNetwork(nn.Module):
    def __init__(self, criterion=torch.nn.MSELoss(), in_dim=1, out_dim=1, zero_bias=True, **kwargs):
        super().__init__()
        self.relu = nn.ReLU()
        
        self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(in_dim, 40)),
            ('relu1', nn.ReLU()),
            ('lin2', nn.Linear(40, 40)),
            ('relu2', nn.ReLU()),
            ]))
        })
        
        # Output layer
        self.model.update({"out": nn.Linear(40, out_dim)}) # should be 40,1
        self.criterion = criterion
        
        if zero_bias:
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    m.bias = nn.Parameter(torch.zeros(m.bias.size()))

    def forward(self, x):
        features = self.model.features(x)
        out = self.model.out(features)
        return out
    

class GeneralLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, prevh=None, prevc=None):
        # x has shape [seq length, batch size, input features]
        # ht has shape [num layers, batch size, hidden dim]
        if prevh is None or prevc is None:
            x, (ht, ct) = self.lstm(x)
        else:
            x, (ht, ct) = self.lstm(x, (prevh, prevc))
        
        return x, (ht,ct) # [seq len, batch size, out dim]
    
    def predict(self, x):
        # ompute output for all hidden states (present in x)
        # x now has shape [seq length, batch size, hidden dim]
        return self.output(x)

        

args.total_time_steps = args.T_train + args.T_test
torch.manual_seed(1337)

for run in range(args.num_runs):
    ########################################################################
    ### Create training data
    rn = SineNetwork()
    param_clone = deepcopy(rn.state_dict())
    rn_opt = torch.optim.SGD(rn.parameters(), lr=1e-2, momentum=0)
    X = []
    Y = []
    GT = []
    Ws, Bs = [], []
    for n in range(args.num_tasks):
        if args.fixed_init:
            rn.load_state_dict(param_clone)
        ########################################################
        # Different batches at every time step? Replace repeat!
        ########################################################
        # sample amplitude from [0.1, 5] and phase from [0, pi]
        a, p = 0.1 + torch.rand(1)*(5.0 - 0.1), 0 + torch.rand(1)*(np.pi-0) 
        Ws.append(a.item()); Bs.append(p.item())
        randx = -args.xrange + torch.rand(args.batch_size)*(2*args.xrange) #sample from [-xrange, +xrange] between -5 and +5
        X.append(randx.repeat(args.total_time_steps, 1).unsqueeze(2)) # [T, batch_size, 1]
        GT.append(fn(x=randx, a=a, p=p).repeat(args.total_time_steps, 1).unsqueeze(2))

        randx = randx.reshape(-1,1)
        currY = []
        for t in range(args.total_time_steps):
            preds = rn(randx)
            gt = fn(randx,a=a,p=p)
            currY.append(preds.detach().numpy())
            loss = torch.mean((preds - gt)**2)
            loss.backward()
            rn_opt.step()
            rn_opt.zero_grad()
        currY = torch.Tensor(currY).reshape(args.total_time_steps,args.batch_size,1)
        Y.append(currY)

    # shape of these things: [num_tasks, T, batch size, 1]
    X = torch.stack(X) 
    Y = torch.stack(Y)
    GT = torch.stack(GT)

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
        perm = torch.randperm(X.size(0)) # permute number of tasks
        count = 0
        loss_itemls = []
        Wls, Bls = [], []
        for bid in range(0, X.size(0)):
            # stack the batch along the batch dimension 1
            input_batch = X[perm[bid]] #[seq len, batch size, infeatures]
            output_batch = Y[perm[bid]]
            gt_batch = GT[perm[bid]]

            a = Ws[perm[bid]]; p = Bs[perm[bid]]
            
            Wls.append(a)
            Bls.append(p)

            if args.objective == "mimick":
                # if label as input, make the first row (sequence item) target equal to 0 
                if args.label_as_input:
                    init_zeros = torch.zeros(input_batch.size(2)*input_batch.size(1)).reshape(1, input_batch.size(1), input_batch.size(2))
                    shifted_output_batch = torch.cat([init_zeros, output_batch], dim=0) # [seq len + 1, batch size, out dim]
                    input_batch = torch.cat([input_batch, shifted_output_batch[:input_batch.size(0),:,:]], dim=2) #[seq len, batch_size, infeatures+1]

                hn, cn = None, None
                predictions = []
                for t in range(args.total_time_steps):
                    output, (hn, cn)  = lstm(input_batch[t,:,:].unsqueeze(0), prevh=hn, prevc=cn)
                    preds = lstm.predict(output)
                    predictions.append(preds)
                    # average over hidden states for every layer and repeat along batch dimension
                    hn = hn.mean(dim=1).unsqueeze(1).repeat(1,args.batch_size,1) # hn shape: [num_layers, batch size, hidden size]
                    if args.average_cell:
                        cn = cn.mean(dim=1).unsqueeze(1).repeat(1,args.batch_size,1)

                pred = torch.cat(predictions)
                losses = (output_batch-pred)**2 #[seq len, batch_size, infeatures+1]
            else:
                hn, cn = None, None
                predictions = []
                for t in range(args.total_time_steps):
                    output, (hn, cn)  = lstm(input_batch[t,:,:].unsqueeze(0), prevh=hn, prevc=cn)
                    preds = lstm.predict(output)
                    predictions.append(preds)
                    # average over hidden states for every layer and repeat along batch dimension
                    hn = hn.mean(dim=1).unsqueeze(1).repeat(1,args.batch_size,1)  # hn shape: [num_layers, batch size, hidden size]
                    if args.average_cell:
                        cn = cn.mean(dim=1).unsqueeze(1).repeat(1,args.batch_size,1)

                pred = torch.cat(predictions)
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
        print(f"Epoch {epoch} - train loss: {epoch_train_loss}, test loss: {epoch_test_loss}")
        if epoch_test_loss < best_epoch_test_loss:
            best_epoch_train_loss = epoch_train_loss
            best_epoch_test_loss = epoch_test_loss
            best_weights = deepcopy(lstm.state_dict())
            best_epoch_losslist = np.array(loss_itemls) 
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