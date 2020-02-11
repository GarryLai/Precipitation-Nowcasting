# -*- coding: utf-8 -*-
#! /usr/bin/python3
"""
Created on Thu Sep 21 16:15:53 2017

@author: cx
"""

from util import *
from cell import ConvLSTMCell
import util

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()

###declare some parameters that might be used 
        self.conv_pad = 0
        self.conv_kernel_size = 3
        self.conv_stride = 1
        self.pool_pad = 0
        self.pool_kernel_size = 3
        self.pool_stride = 3
        self.hidden_size =64
        self.size = int((args.img_size+2*self.conv_pad-(self.conv_kernel_size-1)-1)/self.conv_stride+1)
        self.size1 = int((self.size+2*self.pool_pad-(self.pool_kernel_size-1)-1)/self.pool_stride+1)
###define layers
        self.conv = nn.Conv2d(
             in_channels=1,
             out_channels=8,
             kernel_size=3,
             stride=1,
             padding=0)
        self.pool = nn.MaxPool2d(
                     kernel_size=3
                     )
        self.convlstm1 = ConvLSTMCell(
                        shape=[self.size1,self.size1], 
                        input_channel=8, 
                        filter_size=3,
                        hidden_size=self.hidden_size)
        self.convlstm2 = ConvLSTMCell(
                        shape=[self.size1,self.size1], 
                        input_channel=self.hidden_size, 
                        filter_size=3,
                        hidden_size=self.hidden_size)
        self.deconv = nn.ConvTranspose2d(
                        in_channels=self.hidden_size , 
                        out_channels=1, 
                        kernel_size=6,
                        stride=3,
                        padding=0, 
                        output_padding=1, 
                        )
        self.relu = func.relu


    def forward(self,X):
        X_chunked = torch.chunk(X,args.seq_start,dim=1)
        X = None
        output = [None]*args.seq_length
        state_size = [args.batch_size, self.hidden_size]+[self.size1,self.size1]
        if (torch.cuda.is_available() == True):
            hidden1 = Variable(torch.zeros(state_size)).cuda()
            cell1 = Variable(torch.zeros(state_size)).cuda()
            hidden2 = Variable(torch.zeros(state_size)).cuda()
            cell2 = Variable(torch.zeros(state_size)).cuda()
        else:
            hidden1 = Variable(torch.zeros(state_size)).cpu()
            cell1 = Variable(torch.zeros(state_size)).cpu()
            hidden2 = Variable(torch.zeros(state_size)).cpu()
            cell2 = Variable(torch.zeros(state_size)).cpu()
        
        for i in range(args.seq_start):
                                                        
            output[i] = self.conv(X_chunked[i])     
            output[i] = self.pool(output[i])
            hidden1, cell1 = self.convlstm1(output[i],(hidden1,cell1))
            hidden2, cell2 = self.convlstm2(hidden1,(hidden2,cell2))
            output[i] = self.deconv(hidden2)
            output[i] = self.relu(output[i])
        
        for i in range(args.seq_start,args.seq_length):                                                 
            output[i] = self.conv(output[i-1])    
            output[i] = self.pool(output[i])
            hidden1, cell1 = self.convlstm1(output[i],(hidden1,cell1))
            hidden2, cell2 = self.convlstm2(hidden1,(hidden2,cell2))
            output[i] = self.deconv(hidden2)
            output[i] = self.relu(output[i])
            
        return output[args.seq_start:]


def test(model,reload=False):
### loading validation dataset
    self_built_dataset = util.Dataloader0(args.data_dir+args.testset_name,
                                          args.seq_start,
                                          args.seq_length-args.seq_start,
                                          rot=False)
    name_list = self_built_dataset.all_list  
    trainloader = DataLoader(
        self_built_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True) 

    for iteration,valid_data in enumerate(trainloader,0):
        
        print(iteration)
        valid_X,valid_Y = valid_data
        if ( torch.cuda.is_available() == True and args.cpuonly != 1):
            valid_X = Variable(valid_X, requires_grad=False).cuda()
        else:
            valid_X = Variable(valid_X, requires_grad=False).cpu()
        
        with torch.no_grad():
            output_list = model(valid_X)

        for j in range(args.batch_size):

            start_time = name_list[iteration*args.batch_size+j][args.seq_start-1]
            time_list = [name_list[iteration*args.batch_size+j][i] for i in range(args.seq_start,args.seq_length)]
            A = valid_X[j][-1].data.cpu().numpy().reshape(args.img_size,args.img_size)
            A = (A+0.5).astype(np.uint8)
            A = Image.fromarray(A)
            path = args.img_dir+start_time.split("\\")[-1]
            A.save(path)

            for k in range(args.seq_length-args.seq_start):
                A = output_list[k][j,0,:,:].data.cpu().numpy().reshape(args.img_size,args.img_size)
                A = (A+0.5).astype(np.uint8)
                A = Image.fromarray(A)
                path = args.img_dir+time_list[k][:-4].split("\\")[-1]+"_{}.png".format(k)
                A.save(path)

        output_list = None
        
        

def run_training(args,reload=True):     

    #Initialize model
    if reload:
        model_list = []
        print("Reloading exsiting model")
        maximum = 0
        model_name = "model_"+str(maximum)+".pkl"
        for model_name in os.listdir(args.model_dir):
            num = int(model_name.split("_")[1][:-4])
            if num > maximum:
                maximum = num
        model_name = "model_"+str(maximum)+".pkl"
        if (torch.cuda.is_available() == True and args.cpuonly != 1):
            print("CUDA mode: Enable")
            model = torch.load(args.model_dir+model_name, map_location=torch.device('cuda'))
        else:
            print("CUDA mode: Disable")
            model = torch.load(args.model_dir+model_name, map_location=torch.device('cpu'))
        start = maximum+1

    else:
        print('Initiating new model')
        
        model = Model()
        if (torch.cuda.is_available() == True and args.cpuonly != 1):
            print("CUDA mode: Enable")
            model = model.cuda()
        else:
            print("CUDA mode: Disable")
            model = model.cpu()
        start = 0

    torch.manual_seed(1)
    summary = open(args.logs_train_dir+"log.txt","w") ## you can change the name of your summary. 
    self_built_dataset = util.Dataloader0(args.data_dir+args.trainset_name,
                                          args.seq_start,
                                          args.seq_length-args.seq_start)
    trainloader = DataLoader(
        self_built_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last = True)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
    loss_ave = 0

######Train the model#######
    for epoch in range(start,start+args.epoches):

        print("--------------------------------------------")
        print("EPOCH:",epoch)
        t = time.time()

        for iteration,data in enumerate(trainloader,0):
            loss = 0
            # X is the given data while the Y is the real output
            X, Y = data
            if (torch.cuda.is_available() == True):
                X = Variable(X).cuda()
                Y = Variable(Y).cuda()
            else:
                X = Variable(X).cpu()
                Y = Variable(Y).cpu()
            optimizer.zero_grad()         
            
            with torch.no_grad():
                output_list = model(X)
                
                for i in range(args.seq_length-args.seq_start):
                    loss += criterion(output_list[i], Y[:,i,:,:])

            loss_ave += loss.data/100
            loss = Variable(loss, requires_grad = True)
            loss.backward()
            optimizer.step()
            
            """
            name_list = self_built_dataset.all_list  
            for j in range(args.batch_size):

                start_time = name_list[iteration*args.batch_size+j][args.seq_start-1]
                time_list = [name_list[iteration*args.batch_size+j][i] for i in range(args.seq_start,args.seq_length)]
                A = X[j][-1].data.cpu().numpy().reshape(args.img_size,args.img_size)
                A = (A+0.5).astype(np.uint8)
                A = Image.fromarray(A)
                path = args.img_dir+start_time.split("\\")[-1]
                A.save(path)
            
                for k in range(args.seq_length-args.seq_start):
                        A = output_list[k][j,0,:,:].data.cpu().numpy().reshape(args.img_size,args.img_size)
                        A = (A+0.5).astype(np.uint8)
                        A = Image.fromarray(A)
                        path = args.img_dir+time_list[k][:-4].split("\\")[-1]+"_{}.png".format(k)
                        A.save(path)  
                        """
            
            if iteration%100==0 and iteration!=0:
 
                elapsed = time.time()-t
                t = time.time()

                print("EPOCH: %d, Iteration: %s, Duration %d s, Loss: %f" %(epoch,iteration,elapsed,loss_ave.item()))
                summary.write("Epoch: %d ,Iteration: %s, Duration %d s, Loss: %f \n" %(epoch,iteration,elapsed,loss_ave.item()))
                loss_ave = 0
        #print("Testing Model")
        #test(model,reload=False)
        print("Finished an epoch.Saving the net....... ")
        torch.save(model,args.model_dir+"model_{0}.pkl".format(epoch))    

    summary.close()
    output_list = None

if __name__=="__main__":

    run_training(args)