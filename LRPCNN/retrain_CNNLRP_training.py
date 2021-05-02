
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
from retrain_CNNLRP_model import Net, train, test, val
from retrain_data_preprocessing import EEG_train_dataset, EEG_test_dataset, EEG_val_dataset


# Network parameters
class Params(object):
    batch_size = 10
    test_batch_size = 10
    epochs = 85
    lr = 0.01
    momentum = 0.5
    no_cuda = False
    seed = 1
    log_interval = 10
    
    def __init__(self):
        pass

args = Params()
torch.manual_seed(args.seed)
device = torch.device("cuda:3")
kwargs = {}

#load data, dataloader
train_dataset = EEG_train_dataset()
val_dataset = EEG_val_dataset()
test_dataset = EEG_test_dataset()

train_loader=DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader=DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader=DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

model = Net().double()
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

val_accuracy_via_val_loss = 0
check_min_val_loss = 100
max_test_accuracy = 0

for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)

    val_loss, val_accuracy = val(args, model, device, val_loader)


    if val_accuracy_via_val_loss < val_accuracy :
        check_min_val_loss = val_loss
        val_accuracy_via_val_loss = val_accuracy

        print("Val Accuracy changes ! save model params\n")
        torch.save(model.state_dict(), "real_model_parameter_ch118/stft_aw/ch6_cv_aw_1")


    print("current min_val_loss :", check_min_val_loss, "current val_accuracy :", val_accuracy_via_val_loss)

    if epoch == 85:
        print("final min val accuracy :", val_accuracy_via_val_loss)

test_model = Net().double()
test_model.load_state_dict(torch.load("real_model_parameter_ch118/stft_aw/ch6_cv_aw_1"))
test_model = test_model.to(device)
test_accuracy = test(args, test_model, device, test_loader)
       
