import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    A class for calulcating the focal loss for the loss function
    """
    def __init__(self, Theta,Gamma):
        """
        Initialise the focal loss class

        Inputs
        ------
        Theta: The theta parameter

        Gamma: The gamma parameter
        """
        super(FocalLoss,self).__init__()
        self.theta=Theta
        self.gamma=Gamma
    def __call__(self,Preds,RealVals):
        """
        Inputs
        ------
        Preds: The predicted probabilities of the classes

        RealVals: One-hot encodings of the correct classes
        """
        #First, we are going to get the correct classes from the realvals
        real_classes=torch.argmax(RealVals,dim=-1)
        #Get the probabilities predicted of the targets
        Ps=torch.gather(Preds,dim=2,index=torch.unsqueeze(real_classes,2))
        #Calc loss:
        Loss=-self.theta*(1-Ps)**2*torch.log(Ps)

        return torch.sum(Loss) #Return the sum of the losses

class CreateHeartbeatWindow(nn.Module):
    def __init__(self, maxtimeframe,sigma,device):
        super(CreateHeartbeatWindow,self).__init__()
        self.maxtimeframe=maxtimeframe
        self.sigma=sigma
        self.device=device
    def __call__(self, InputPositions):
        """
        Function to return the window of the heartbeat

        InputPositons size: [Batchsize, NumberofClassifications, 1] (The positions are at the end)
        """
        #To do this, I am first going to change any 0s to above the maximum length, as they are at the end
        InputPositions[InputPositions==0]=self.maxtimeframe*2
        #Append one more as well
        InputPositions=torch.cat((torch.unsqueeze(InputPositions,2),self.maxtimeframe*2*torch.ones([InputPositions.shape[0],1,1]).to(self.device)),dim=1)
        #Now get the start positions and the end positions of the windows:
        StartEnd=InputPositions[:,:-1,:]*self.sigma+(1-self.sigma)*InputPositions[:,1:,:]
        #Append a 0 to the start
        StartEnd=torch.cat((torch.zeros([StartEnd.shape[0],1,1]).to(self.device),StartEnd),dim=1)
        #Make sure we are not giving too wide a window
        StartEnd[StartEnd>self.maxtimeframe]=self.maxtimeframe
        #Return the start and end position by concating them on the 2nd dimension
        StartEnd=torch.cat((StartEnd[:,:-1,:],StartEnd[:,1:,:]),dim=2)
        return StartEnd


class GboxIoULoss(nn.Module):
    """
    A class that does the box gIoU loss function, but also deals with the range for a heart beat to be in.
    This is based off the paper we are doing our base model on
    """
    def __init__(self,device,delta=float(10e-7),maxtimeframe=1080):
        """
        Initialise the gboxiou loss module

        Inputs
        ------

        delta: the delta parameter for the actual loss function

        maxtimeframe: the number of discrete recordings in the time window chosen (1080 is 3 seconds)
        """
        super(GboxIoULoss,self).__init__()
        self.delta=delta
        self.maxtimeframe=maxtimeframe
        self.device=device
    def __call__(self, Preds,TargetWindows):
        """
        Calculate the loss, by creating the windows and then using the loss function

        Inputs
        ------
        Preds: The predicted windows of the heartbeats

        Targets: The actual windows of the heartbeats
        """

        #Get the start of window and end of window
        out1,out2=Preds[:,:,0],Preds[:,:,1]
        tgt1,tgt2=TargetWindows[:,:,0],TargetWindows[:,:,1]
        #Calculate intersetctions,unions and areas
        Intersection=torch.max(torch.zeros_like(out1).to(self.device),torch.min(out2,tgt2)-torch.max(out1,tgt1))
        Union=tgt2-tgt1+out2-out1-Intersection #Union is the size of both minus the intersection between them
        Area=torch.max(tgt2,out2)-torch.min(tgt1,out1) #This is the shortest area that encompases both windows

        #Calc loss
        Loss=1-(Intersection/Union-(Area-Union)/(Area+self.delta))
        

        return torch.nansum(Loss) #Return the sum of all the losses 