import torch
import wandb
def train(k,model, optimizer,loader_train,loader_val,epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    #Sync to Wandb
    wandb.init(name=k,project="pytorch")

    # Using GPU or CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cuda.cufft_plan_cache.clear()
    else:
        device = torch.device('cpu')
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    print("Device:",device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    for e in range(1,epochs+1):
        
        for _,(x, y) in enumerate(loader_train):
            
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            try:
                loss = torch.nn.functional.cross_entropy(scores[0], y)
            except:
                loss = torch.nn.functional.cross_entropy(scores, y)

            wandb.log({"train_loss": loss}) #log the loss for each iteration
            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            scheduler.step()

        val_acc,val_loss=check_accuracy(loader_val, model)
        train_acc,train_loss=check_accuracy(loader_train,model)
        wandb.log({"epochs":e,"train_acc":train_acc,"val_acc":val_acc}) #logging accuracy for each epochs
        wandb.log({"epochs":e,"train_loss":train_loss,"val_loss":val_loss}) #logging loss for each epochs
        print('epochs:%d, loss =%.4f, train_acc:%.4f, val_acc:%.4f' % (e, loss.item(),train_acc,val_acc))
        with open("Inception.log",'a') as file:
            file.write("epochs:"+str(e)+" loss:"+str(loss)+
            " train_acc:"+str(train_acc*100)+"%"+" val_acc:"+str(val_acc*100)+"%\n")
            file.close()        
        print()


def check_accuracy(loader, model):
    #GPU or CPU?
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    num_correct = 0
    num_samples = 0
    wrong_x=torch.([])
    wrong_y=trch.([])
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            #Calulate the loss
            try:
                loss = torch.nn.functional.cross_entropy(scores[0], y)
            except:
                loss = torch.nn.functional.cross_entropy(scores, y)

            _, preds = scores.max(1)
            correct=(preds==y)
            num_correct += correct.sum()
            num_samples += preds.size(0)

            #Extract the wrong samples
            index=torch.argwhere(correct).flatten()
            wrong_x=torch.concatenate((wrong_x,x[index]))
            wrong_y=torch.concatenate((wrong_y,y[index]))

            if(num_samples>2000):
                break

        train_partial(model,wrong_x,wrong_y,optimizer)
        acc = float(num_correct) / num_samples
        return acc,loss

def train_partial(model,x,y,optimizer):

    for x,y in generator_numpy(x,y):
        scores = model(x)
        try:
            loss = torch.nn.functional.cross_entropy(scores[0], y)
        except:
            loss = torch.nn.functional.cross_entropy(scores, y)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def generator_numpy(x,y,batch_size=64):  #x.shape(32,32)->shape(x.shape[1],x.shape[0])
    num=x.shape[0]
    while True:
        for i in range(0,num,batch_size):
            batch_x=np.array(x[i:i+batch_size])
            batch_y=np.array(y[i:i+batch_size])
            yield batch_x,batch_y;
