import os
import torch
import numpy as np
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix
import csv


def optimize_group_graph(model, space_conf, dataloader_train,
                  data_loader_val, device, criterion, optimizer,
                  scheduler, batch,num_epochs, models_out=None, train_type = 'bck'):
    best_acc = 0
    best_conf = None
    best_loss = 1000

    for epoch in range(num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # print('TYPE', train_type)
        
        if train_type == 'bck': #classification
            loss = _train_model(model,space_conf, dataloader_train, optimizer, criterion, device,batch,train_type)

            acc, conf = _test_maas(model, space_conf, data_loader_val, criterion, device, batch,train_type)
        
            if(best_acc < acc):
                best_acc = max(acc, best_acc)
                best_conf = conf
                model_target = os.path.join(models_out, str(best_acc)+'.pth')
                print('save model to ', model_target)
                torch.save(model.state_dict(), model_target)

        elif train_type == 'agr': # agreement
            loss = _train_model(model,space_conf, dataloader_train, optimizer, criterion, device,batch,train_type)

            val_loss, my_loss = _test_maas(model, space_conf, data_loader_val, criterion, device, batch, train_type)
    
            if(best_loss > my_loss):
                best_loss = min(my_loss, best_loss)
                # best_conf = conf
                model_target = os.path.join(models_out, str(best_loss)+'.pth')
                print('save model to ', model_target)
                torch.save(model.state_dict(), model_target)

        scheduler.step()

    return model


def _train_model(model,space_conf, dataloader, optimizer, criterion, device, batch, train_type):
    model.train()
    num_nodes, time_l = space_conf
    # Stats vars
    softmax_layer = torch.nn.Softmax(dim=-1)
    running_loss = 0.0

    y_true = []
    y_pred = []

    # Iterate over data
    for idx, dl in enumerate(dataloader):

        print('\t Train iter {:d}/{:d} {:.4f}'.format(idx, len(dataloader), running_loss/(idx+1)) , end='\r')
        vid_key, graph_data = dl
        # graph_data = dl
        graph_data = graph_data.to(device)
        targets = graph_data.y
        targets = targets.type(torch.LongTensor)
        targets = targets.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(graph_data)
            loss = criterion(outputs.float(), targets) #.float()
            loss.backward()
            optimizer.step()

        with torch.set_grad_enabled(False):
            
            targets_new = targets.cpu().detach().numpy()

            # classification
            if train_type == 'bck': #classification
                targets_new = targets_new.reshape([batch,time_l,num_nodes])
                targets_new = np.max(targets_new, axis = -1)
                targets_new = np.sum(targets_new, axis = -1)
                targets_new = [1 if x> int(time_l/2) else 0 for x in targets_new]

                op_new = torch.argmax(outputs, dim=1).cpu().detach().numpy()
                op_new = op_new.reshape([batch,time_l,num_nodes])
                op_new = np.max(op_new, axis = -1)
                op_new = np.sum(op_new, axis = -1)
                op_new = [1 if x> int(time_l/2) else 0 for x in op_new]

            # regression
            elif train_type == 'agr': # agreement
                targets_new = targets_new.reshape([batch,time_l,num_nodes])
                targets_new = np.average(targets_new, axis = -1)
                targets_new = np.average(targets_new, axis = -1)

                op_new = outputs.cpu().detach().numpy()
                op_new = op_new.reshape([batch,time_l,num_nodes])
                op_new =  np.average(op_new, axis = -1)
                op_new =  np.average(op_new, axis = -1)

            y_true.extend(targets_new)
            y_pred.extend(op_new)

        # statistics
        running_loss += loss.item()
        if idx == len(dataloader)-2:
            break

    epoch_loss = running_loss / len(dataloader)
    if train_type == 'bck': #classification
        acc = accuracy_score(y_true, y_pred)
        print('Train Loss: {:.4f}  acc: {:.4f}'.format(epoch_loss, acc))
    elif train_type == 'agr':
        loss = mean_squared_error(y_true,y_pred )
        print('Train Loss: {:.4f}  my loss: {:.4f}'.format(epoch_loss, loss))

    return loss


def _test_maas(model, space_conf, dataloader, criterion, device, batch, train_type):
    model.eval()  # Set model to evaluate mode
    
    num_nodes, time_l = space_conf

    softmax_layer = torch.nn.Softmax(dim=-1)
    running_loss = 0.0

    y_true = []
    y_pred = []

    # Iterate over data.
    for idx, dl in enumerate(dataloader):
        print('\t Val iter {:d}/{:d}'.format(idx, len(dataloader)) , end='\r')
        vid_key, graph_data = dl
        graph_data = graph_data.to(device)
        targets = torch.flatten(graph_data.y)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(graph_data)
            targets = targets.type(torch.LongTensor) 
            targets = targets.to(device)
            loss = criterion(outputs.float(), targets)
            # print('val loss', loss)
            targets_new = targets.cpu().detach().numpy()

            if train_type == 'bck': #classification
                targets_new = targets_new
                op_new = torch.argmax(outputs, dim=1).cpu().detach().numpy()

                targets_new = targets_new.reshape([batch,time_l,num_nodes])
                targets_new = np.max(targets_new, axis = -1)
                targets_new = np.sum(targets_new, axis = -1)
                targets_new = [1 if x> int(time_l/2) else 0 for x in targets_new]

                op_new = op_new.reshape([batch,time_l,num_nodes])
                op_new = np.max(op_new, axis = -1)
                op_new = np.sum(op_new, axis = -1)
                op_new = [1 if x> int(time_l/2) else 0 for x in op_new]

            # agr
            elif train_type == 'agr': # agreement
                targets_new = targets_new.reshape([batch,time_l,num_nodes])
                targets_new = np.average(targets_new, axis = -1)
                targets_new = np.average(targets_new, axis = -1)
                op_new = outputs.cpu().detach().numpy()
                # print(op_new.shape)
                op_new = op_new.reshape([batch,time_l,num_nodes])
                op_new =  np.average(op_new, axis = -1)
                op_new =  np.average(op_new, axis = -1)

            y_true.extend(targets_new)
            y_pred.extend(op_new)

            
        # statistics
        running_loss += loss.item()
        if idx == len(dataloader)-2:
            break

    epoch_loss = running_loss / len(dataloader)
    if train_type == 'bck': 
        acc = accuracy_score(y_true, y_pred)
        cf_matrix = list(confusion_matrix(y_true, y_pred))
        # report = classification_report(label_lst,pred_lst)
        print('Val Loss: {:.4f} acc{:.4f} conf{:}'.format(epoch_loss, acc,cf_matrix))
        # print(acc)
        return acc, cf_matrix

    elif train_type == 'agr':
        
        my_loss = mean_squared_error(y_true,y_pred )
        print('Val Loss: {:.4f} loss {:.4f}'.format(epoch_loss, my_loss))
        return epoch_loss, my_loss


def _get_prediction(model, space_conf, dataloader, criterion, device, train_type, output_file):
    f = open(output_file, 'w')
    writer = csv.DictWriter(f, fieldnames = ['id','label'])
    writer.writeheader()
    row={}

    batch = 1
    model.eval()  # Set model to evaluate mode

    num_nodes, time_l = space_conf

    y_true = []
    y_pred = []

    # Iterate over data.
    for idx, dl in enumerate(dataloader):
        print('\t Val iter {:d}/{:d}'.format(idx, len(dataloader)) , end='\r')
        vid_key, graph_data = dl
        # graph_data = dl
        graph_data = graph_data.to(device)
        targets = torch.flatten(graph_data.y)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(graph_data)
            targets = targets.to(device)
            targets_new = targets.cpu().detach().numpy()

            if train_type == 'bck': #classification
                # label_lst.extend(targets.cpu().numpy().tolist())
                # pred_lst.extend(softmax_layer(outputs).cpu().numpy()[:, 1].tolist())
                targets_new = targets_new
                op_new = torch.argmax(outputs, dim=1).cpu().detach().numpy()
                targets_new = targets_new.reshape([batch,time_l,num_nodes])
                targets_new = np.max(targets_new, axis = -1)
                targets_new = np.sum(targets_new, axis = -1)
                targets_new = [1 if x> int(time_l/2) else 0 for x in targets_new]
                op_new = op_new.reshape([batch,time_l,num_nodes])
                op_new = np.max(op_new, axis = -1)
                op_new = np.sum(op_new, axis = -1)
                op_new = [1 if x> int(time_l/2) else 0 for x in op_new]
            # agr
            elif train_type == 'agr': # agreement
                targets_new = targets_new.reshape([batch,time_l,num_nodes])
                targets_new = np.average(targets_new, axis = -1)
                targets_new = np.average(targets_new, axis = -1)
                op_new = outputs.cpu().detach().numpy()
                # print(op_new.shape)
                op_new = op_new.reshape([batch,time_l,num_nodes])
                op_new =  np.average(op_new, axis = -1)
                op_new =  np.average(op_new, axis = -1)

            y_true.extend(targets_new)
            y_pred.extend(op_new)

            row['id'] = vid_key[0]
            row['label'] = op_new[0]
            writer.writerow(row)
    f.close()
    print(mean_squared_error(y_true,y_pred))
    print(accuracy_score(y_true, y_pred))
    print(list(confusion_matrix(y_true, y_pred)))
    print(classification_report(y_true,y_pred))