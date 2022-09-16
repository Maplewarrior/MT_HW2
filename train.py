'''
Train Loop
'''

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from src.models import UNET
from .utils import ( save_checkpoint, check_accuracy )
from .utils import DiceLoss2D
import os

def train_fn(train_loader, val_loader, model, optimizer, loss_fn, scheduler, num_epochs, checkpoint_name: str):

    # Send model to compute device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device)
    bnorm = nn.BatchNorm2d(num_features=3)
    dice = [0]

    for epoch in range(num_epochs):
        print(f'Starting epoch #{epoch+1}...')
        with tqdm(train_loader) as loop:
            for batch_idx, (data, targets) in enumerate(loop):
                data = bnorm(data).to(device=device)
                targets = targets.float().unsqueeze(1).to(device=device)

                # forward
                with torch.cuda.amp.autocast():
                    predictions = torch.sigmoid(model(data))
                    loss = loss_fn(predictions, targets)
                    print(f'epoch #{epoch}; loss = {loss.item()}')

                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # update tqdm loop
                loop.set_postfix(loss=loss.item())

        # Get a val accuracy here
        val_dice = check_accuracy(val_loader, model)

        # Save a model checkpoint if current state yields better dice score
        if val_dice > float(max(dice)):
            val_string = int(100*val_dice)
            save_checkpoint(state=model.state_dict(), filename=f'checkpoints/{checkpoint_name}_epoch{epoch}_{val_string}ds.pth')
        dice.append(val_dice)

        scheduler.step()
    # Saving model after last epoch
    save_checkpoint( state=model.state_dict(), filename=f'checkpoints/{checkpoint_name}pth' )

def train_model(model, data, epochs, verbose=True):
    model.train()
    start = time.time()
    total_loss = 0
    
    source_all = data['input_ids']
    target_all = data['target']
    
    # loop over epochs
    for epoch in range(epochs):
        
        # loop over all sentences
        for i in range(len(source_all)):
            
            # unsqueeze to avoid dim mismatch between embedder and pe
            src = source_all[i].unsqueeze(0)
            trg = target_all[i].unsqueeze(0)
            size = len(trg)
            
            source_pad = source_all[i] == 0
            
            target_pad = target_all[i] == 0
            
            input_msk = (source_all[i] != source_pad).unsqueeze(1)
            
            # trg_ipt = trg[:, :-1]
            # targets = trg[:, 1:].contiguous().view(-1)
            
            nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
            nopeak_mask = torch.autograd.Variable(torch.from_numpy(nopeak_mask) == 0)
            
            target_msk = (target_all[i] != target_pad).unsqueeze(1)
            target_msk = target_msk & nopeak_mask
            
            print("getting preds...")
            # preds = model.forward(src, trg , None, None)
            preds = model.forward(src, trg, input_msk, target_msk)
            print("preds gotten...")
            optim.zero_grad()    
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ignore_idx=target_pad)
            loss.backward()
            optim.step()
            total_loss += loss.data[0]
            if verbose:
                print("time =",time.time()-start, "\n loss:", loss.data[0], "\n total loss:", total_loss)