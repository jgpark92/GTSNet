import warnings
warnings.filterwarnings('ignore')

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import sys
from sklearn.metrics import classification_report

from utils import input_pipeline, count_parameters_in_MB, AvgrageMeter, accuracy, data_info
from model import GTSNet

save_path = './savemodel/'
epoches = 500
batch_size = 128
seed = 2

dataset = "UCI_HAR"
#dataset = "WISDM"
#dataset = "OPPORTUNITY"
#dataset = "PAMAP2"

input_nc, segment_size, class_num = data_info(dataset)

def weight_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant(m.weight,1)
        nn.init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant(m.bias, 0)

def train(train_queue, model, criterion, optimizer):
    cl_loss = AvgrageMeter()
    cl_acc = AvgrageMeter()
    model.train() # mode change

    for step, (x_train, y_train) in enumerate(train_queue):
        
        n = x_train.size(0)
        x_train = Variable(x_train, requires_grad=False).cuda().float()
        y_train = Variable(y_train, requires_grad=False).cuda().long() # It can handle the compute of model and memory transfer on GPU at the same time.

        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)

        loss.backward() # weight compute
        optimizer.step() # weight update


        prec1 = accuracy(logits.cpu().detach(), y_train.cpu())
        cl_loss.update(loss.data.item(), n)
        cl_acc.update(prec1, n)

    return cl_loss.avg, cl_acc.avg

def infer(eval_queue, model, criterion):
    cl_loss = AvgrageMeter()
    model.eval()

    preds = []
    with torch.no_grad():
        for step, (x, y) in enumerate(eval_queue):
            x = Variable(x).cuda().float()
            y = Variable(y).cuda().long()

            logits = model(x)
            loss = criterion(logits, y)
            preds.extend(logits.cpu().numpy())

            n = x.size(0)
            cl_loss.update(loss.data.item(), n)


    return cl_loss.avg, np.asarray(preds)

train_queue, eval_queue, y_test_unary = input_pipeline(dataset, input_nc, batch_size)

if not torch.cuda.is_available():
    print('no gpu device available')
    sys.exit(1)

np.random.seed(seed)
torch.cuda.set_device(0)
torch.manual_seed(seed)
cudnn.benchmark = True 
torch.cuda.manual_seed(seed)

criterion = nn.CrossEntropyLoss().cuda()
model = GTSNet(input_nc, class_num).cuda()
model.apply(weight_init)
print("param size = %fMB" % count_parameters_in_MB(model))


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9,0.999),
    weight_decay=1e-4,
    eps=1e-08
)
max_f1 = 0
weighted_avg_f1 = 0

for epoch in range(epoches):
    
    # training
    train_loss, train_acc = train(train_queue, model, criterion, optimizer)

    # evaluating
    eval_loss, y_pred = infer(eval_queue, model, criterion)
    results = classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True)
    weighted_avg_f1 = results['weighted avg']['f1-score']
    if (epoch+1) % 50 == 0:
        print('training... ', epoch+1)
    if max_f1 < weighted_avg_f1:
        # torch.save(model.state_dict(), save_path +dataset+'.pt')
        print("epoch %d, loss %e, weighted f1 %f, best_f1 %f" % (epoch+1, eval_loss, weighted_avg_f1, max_f1))
        max_f1 = weighted_avg_f1
        print(classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4))
        if dataset=='PAMAP2':
            print('ADL:',classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=[0,1,2,3,4,5,6,10,11,12,13,17])['weighted avg']['f1-score'])
            print('Complex:',classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=[7,8,9,14,15,16])['weighted avg']['f1-score'])
        
 