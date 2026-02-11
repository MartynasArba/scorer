import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# load into SleepSignals
#do scoring: launcher function 
from scorer.data.loaders import SleepTraining
from scorer.models.sleep_cnn import SleepCNN, EphysSleepCNN, FreqSleepCNN


import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt

#helpers
import tqdm
from pathlib import Path

def train_model(model, trainloader, optimizer, criterion, device = 'cuda', epochs = 20, save_n_epochs = None, save_path = None):
    """
    trains model
    """
    model.train()
    
    stop_criterion = 20
    not_improved = 0
    prev_loss = 9999
    
    losses = []
    
    for epoch in range(epochs):  #
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):

            sample, label = data
            
            # # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(sample)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            #early stopping if loss doesn't improve in 20 batches
            if loss.item() >= prev_loss:
                not_improved += 1
                prev_loss = loss.item()
            else:
                not_improved = 0
            
            if not_improved >= stop_criterion:
                print(f'stopped early at epoch {epoch}, iter {i}')
                break

            # print statistics
            running_loss += loss.item()       
            
            if i % 200 == 199:    # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                losses.append(running_loss / 200)
                running_loss = 0.0
        if save_n_epochs is not None:
            if epoch % save_n_epochs == 0 and epoch > 1:
                 torch.save(model, save_path / f'weights/sleepcnn_{epoch}_2026-01-19.pt')
                
    print('train done')
    return losses

def evaluate_model(model, testloader):
    
    total = 0
    correct  = 0

    all_preds = []
    all_labels = []
    maxprobs = []
        
    model.eval()
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(testloader)):
            sample, label = data
            outputs = model(sample)
            probs = F.softmax(outputs, dim = 1)
            _, pred = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
            
            #to get final predictions
            all_preds.extend(pred.to('cpu').numpy().tolist())
            all_labels.extend(label.to('cpu').numpy().tolist())
            probs, _ = torch.max(probs, 1)
            maxprobs.extend(probs.to('cpu').numpy().tolist())
            
            if not i % 2000:
                print(f'accuracy: {correct/total}')
                
    #correct for 3- and 4- state models where there are 5  label classes
    if len(np.unique(all_preds)) != len(np.unique(all_labels)):
        print('correcting label mismatch')
        all_preds = np.array(all_preds) + 1  #should be good for 4state, 0 is not classified
        if len(np.unique(all_preds)) == 3: #3state will need reordering
            print('3state found')
            all_preds[all_preds == 3] += 1
        else:
            print('4state found')

        
    return all_labels, all_preds, maxprobs
        
def eval_plots(all_labels, all_preds, maxprobs, save_path, tag = '_'):
    statedict = {
        0 : "Unlabeled",
        1 : "Awake",
        2 : "NREM",
        3 : "IS",
        4 : "REM"
    }

    im = plt.imshow(confusion_matrix(all_labels, all_preds, normalize = 'true'))#could normalize also
    cb = plt.colorbar(im)
    plt.xlabel('prediction')
    plt.ylabel('label')
    plt.savefig(save_path / f'training_plots/{tag}conf_matrix.png')
    plt.close()

    #check "confidence" by state - plot maximum probability
    fig, ax = plt.subplots(1, 5, figsize = (10, 3), sharex = True)

    for state in (0, 1, 2, 3, 4):
        stateprobs = np.array(maxprobs)[np.where(np.array(all_preds) == state)[0]]
        ax[state].hist(stateprobs, alpha = 0.3)
        ymin, ymax = ax[state].get_ylim()
        ax[state].vlines(np.mean(stateprobs), ymin, ymax)
        ax[state].set(xlabel = statedict[state])
    fig.supxlabel("Confidence when predicting state")
    plt.tight_layout()
    plt.savefig(save_path / f'training_plots/{tag}_state_confidence.png')
    plt.close()

    #[0-unlabeled, 1-AWAKE, 2-NREM, 3-IS, 4-REM]
    print(classification_report(all_labels, all_preds))        #"0-unlabeled",, target_names = ["1-AWAKE", "2-NREM", "3-IS", "4-REM"]


if __name__ == "__main__":
    import glob
    
    torch.set_grad_enabled(False)
    
    save_path = Path(r"C:\Users\marty\Projects\scorer\scorer\models")
    
    metadata = {'ecog_channels' : '1', 'emg_channels' : '2', 'sample_rate' : '250', 'ylim' : 'standard', 'device':'cuda'}
    
    model_names = ['heuristic', '3state_ephysCNN', '4state_ephysCNN', '3state_fftCNN', '4state_fftCNN','3state_CNN', '4state_CNN']
    
    model_paths = glob.glob(r'C:\Users\marty\Projects\scorer\scorer\models\weights\*.pt')
    #testing all models
    
    dataset = SleepTraining(
                            data_path = 'G:/oslo_data',
                            n_files_to_pick = 100,
                            random_state = 0,
                            device = 'cuda',
                            transform = None,
                            augment = False,
                            metadata = metadata
                        )
    loader = DataLoader(dataset, batch_size = 64)
    
    for path in model_paths:
        print(f'starting eval of {path}...')
        model = torch.load(path, weights_only= False)
        all_labels, all_preds, maxprobs = evaluate_model(model, loader)
        print(accuracy_score(all_labels, all_preds))
        eval_plots(all_labels, all_preds, maxprobs, save_path, tag = path[-30:-4].replace('\\', '_'))                        
    
    # #model to implement
    # torch.set_grad_enabled(True)
    # #load dataset
    # dataset = SleepTraining(
    #     data_path = 'G:/oslo_data',
    #     n_files_to_pick = 300,
    #     random_state = 0,
    #     device = 'cuda',
    #     transform = None,
    #     augment = True,
    #     metadata = metadata, 
    #     balance = 'oversample',
    #     exclude_labels = (0,3),#add labels to exclude here
    #     merge_nrem = False
    # )    
    
    # #create dataloaders
    # train_size = .8
    # test_size = .2
    
    # lengths = [int(len(dataset) * train_size), int(len(dataset) * test_size)]
    # while sum(lengths) != len(dataset):
    #     if sum(lengths) > len(dataset):
    #         lengths[0] -= 1
    #     elif sum(lengths) < len(dataset):
    #         lengths[0] += 1
            
    # train_set, val_set = torch.utils.data.random_split(dataset, lengths)
    # trainloader = DataLoader(train_set, batch_size = 64)
    # testloader = DataLoader(val_set, batch_size = 64)
    # num_classes = len(torch.unique(dataset.all_labels))
    # device = metadata.get('device', 'cuda')
    
    # model_name = 'weights/3state_ephysCNN_2026-01-27.pt'
    # #create model
    # model = EphysSleepCNN(num_classes = num_classes).to(device= device)#SleepCNN / FreqSleepCNN / EphysSleepCNN to use all features, mean_std = (dataset.mean, dataset.std) to standardize inputs
    # #crossentropy loss
    # criterion = nn.CrossEntropyLoss() # crossentropy for classification
    # # optimizer adam
    # optimizer = optim.Adam(model.parameters(),lr = 1e-3)
    # #train 
    # losses = train_model(model, trainloader, optimizer, criterion, device = 'cuda', epochs = 200, save_n_epochs = 20, save_path = save_path)
    # #should add save model, then load later if needed
    # torch.save(model, save_path / model_name)
    # print('model saved!')
    # torch.cuda.empty_cache()
    
    # #plot loss
    # plt.plot(losses)
    # plt.savefig(save_path / 'training_plots/loss.png')
    # plt.close()
    
    # # model = torch.load(save_path / model_name, weights_only= False)
    # all_labels, all_preds, maxprobs = evaluate_model(model, testloader)
    # print(accuracy_score(all_labels, all_preds))
    # eval_plots(all_labels, all_preds, maxprobs, save_path)