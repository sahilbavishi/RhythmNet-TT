import pandas as pd
import numpy as np
from model_architectures import Quite_Big_Titan_Model,Quite_Big_Model
import argparse
import torch
import tqdm
import time
from scipy.signal import resample

#python physionet_tester.py --MLP_path "C:/Users/User/Desktop/Python Code/Uni/SEM 2/MLP" --experiment_name PostBaseModel0001TH6HU24 --is_titan False --timespan 3 --transformer_heads 6 --hidden_units 24

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#First, get info
def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    """
    parser = argparse.ArgumentParser(
        description='Helper script for training and evaluating models on the ECG dataset.'
    )

    # Model and dataset parameters
    parser.add_argument('--experiment_name', nargs="?", type=str, default="Titan", help='Name of the experiment')
    parser.add_argument('--epoch',nargs="?",type=int,default=-1,help="The epoch to test, -1 if you want to load the last epoch")
    parser.add_argument('--MLP_path', nargs="?", type=str, default="C:/Users/User/Desktop/Python Code/Uni/SEM 2", help='File path of where the MLP file is')

    parser.add_argument('--is_titan', nargs="?", type=str2bool, default=True, help='True if this is the titan model')
    parser.add_argument('--timespan',type=int,default=3,help="The number of seconds the model is for")

    #We need all of the hyperparameters that we used in the model creation
    parser.add_argument('--transformer_heads', nargs="?", type=int, default=6, help='Number of transformer heads')
    parser.add_argument('--hidden_units', nargs="?", type=int, default=24, help='Number of hidden units in final layer')
    parser.add_argument('--phi', nargs="?", type=float, default=1, help='Scale of CNN Backbone')
    parser.add_argument('--alpha', nargs="?", type=float, default=0.1, help='Alpha for neural memory')
    parser.add_argument('--nm_hu', nargs="?", type=int, default=128, help='Neural Memory hidden units')
    parser.add_argument('--nm_kqv_size', nargs="?", type=int, default=64, help='Size of query/key/value in neural memory')
    parser.add_argument('--pers_dim', nargs="?", type=int, default=32, help='Size of the persistent memory')
    parser.add_argument('--nu', nargs="?", type=float, default=0.9, help='Nu for neural memory')
    parser.add_argument('--theta', nargs="?", type=float, default=0.3, help='Theta for neural memory')

    args = parser.parse_args()
    return args

args=get_args()

experiment_name=args.experiment_name
epoch=args.epoch
MLP_path=args.MLP_path
timespan=args.timespan
is_titan=args.is_titan

model_name=f"train_model_{epoch}.pth"
fileName=f"{MLP_path}/MLP_CW/DummyExperiment/pytorch_mlp_framework/{experiment_name}/saved_models/"

print(f"Looking for epoch number {epoch} in {fileName}")

if epoch==-1:
    #If the epoch -1, then we need to get the last epoch run
    #Find the final epoch:
    import os #Just to get the file names
    fileNames=os.listdir(fileName)
    for name in fileNames:
        num=int(name[12:-4])
        if num>epoch:
            epoch=num


fileName=f"{MLP_path}/MLP_CW/DummyExperiment/pytorch_mlp_framework/{experiment_name}/saved_models/train_model_{epoch}.pth"


#Time to load the model:

Input_shape=[2,1,timespan*360] #timespan*360Hz... Also, batch size of 2 to make sure the model knows to expect more than 1 batch

if not is_titan:
    #If it is the titan model, then set it up
    model=Quite_Big_Model(input_shape=Input_shape,
                                d_model=6,
                                transformer_heads=args.transformer_heads,
                                hidden_units=args.hidden_units,
                                num_classes=5,
                                phi=args.phi)
else:
    model=Quite_Big_Titan_Model(input_shape=Input_shape,
                                d_model=6,
                                transformer_heads=args.transformer_heads,
                                hidden_units=args.hidden_units,
                                num_classes=5,
                                phi=args.phi,
                                nm_hu=args.nm_hu,
                                nm_kqv_size=args.nm_kqv_size,
                                persistent_dim=args.pers_dim,
                                alpha=args.alpha,
                                nu=args.nu,
                                theta=args.theta)



state=torch.load(fileName,weights_only=False,map_location=torch.device('cpu'))
model.load_state_dict(state['network']) #Load the model state


#Now we can have a look at the performace on the physionet stuff
physionetFile=f"{MLP_path}/MLP_CW/physionet_csv_files/ecg_records/"
annotationFile=f"{MLP_path}/MLP_CW/physionet_csv_files/REFERENCE-v3.csv"
annotations=pd.read_csv(annotationFile)
annotations=annotations[annotations["Their_label"]!="~"] #Get rid of the samples that were not classified

#We also need to make sure we have the physionet data in the right time slots
def resample_ecg(ecg_data,origHz,targetHz):
    """Function to resample the ecg signals"""
    resample_factor=targetHz/origHz
    resampled_ecg=resample(ecg_data,int(len(ecg_data)*resample_factor))
    return resampled_ecg
def save_resample(Num):
    """Funciton to take an ecg record and resample and save it as a seperate file"""
    array=pd.read_csv(f"{physionetFile}/A{Num:05d}_ecg.csv")
    targetHz=360
    CurrentHz=300
    new_ecg=resample_ecg(array,CurrentHz,targetHz)
    df=pd.DataFrame(new_ecg)
    df.to_csv(f"{physionetFile}/A{Num:05d}_resamp_ecg.csv",index=False)
def split_record(Num,size_recording):
    """Takes a resampled csv and splits it into windows"""
    array=np.array(pd.read_csv(f"{physionetFile}/A{Num:05d}_resamp_ecg.csv"))
    startNum=0#Keeps track of the start of the record
    length=array.shape[0]
    new_array=array[startNum:(startNum+size_recording)].T
    startNum+=size_recording
    while startNum<length:
        add_array=array[startNum:(startNum+size_recording)].T
        if add_array.shape[1]<size_recording:
            #If the length of the recording is shorter than the timeframe, pad it at the end
            add_array=np.pad(add_array,pad_width=((0,0),(0,size_recording-add_array.shape[1])))
        new_array=np.vstack([new_array,add_array])
        startNum+=size_recording
    #Now we can save our numpy array back in a new file:
    data=pd.DataFrame(new_array)
    data.to_csv(f"{physionetFile}/A{Num:05d}_{timespan}secs_ecg.csv",index=False)


if not os.path.exists(f"{physionetFile}/A08528_{timespan}secs_ecg.csv"):
    #if it doesn't exist, then we need to preprocess the data to make it exist
    print(f"Data not pre-processed for {timespan} seconds, pre-processing now")
    targetHz=360
    CurrentHz=300
    size_recording=timespan*targetHz
    if not os.path.exists(f"{physionetFile}/A08528_resamp_ecg.csv"):
        #If the resampled files don't exist, make them
        print("ecg records not resampled, resampling now:")
        for i in tqdm.tqdm(range(8528)): #loop through all the files and resample them
            save_resample(i+1)
        print("resampling successful")
    #Now we definitely have resampled files we can split them and save them
    print("Splitting the records:")
    for i in tqdm.tqdm(range(8528)):
        split_record(i+1,size_recording)



#Now we can loop through each annotation and calc some predictions
numCorrect=0

pred_dictionary={
    0:"N",
    1:"A",
    2:"O",#V and F both correspond to the label O
    3:"O"
}

posLabels=np.array(["N","A","O"])#an array of the possible labels

ConfMat=np.zeros((3,4)) #3 rows for the true values, 4 rows for the predictions of the model
NewConfMat=np.zeros((3,4))
TotalTime=0

#This is the mean and std of the test data -depending on the time window
if timespan==3:
    mu=-0.40797796845436096
    sigma=0.480744332075119
elif timespan==7:
    mu=-0.4101879894733429 
    sigma=0.48073437809944153
elif timespan==15:
    mu=-0.4141332805156708
    sigma=0.48012611269950867

for anno_index,recording_num in tqdm.tqdm(enumerate(annotations["Recording"]),total=len(annotations["Recording"])):
    dataFile=physionetFile+f"{recording_num}_{timespan}secs_ecg.csv"
    data=torch.tensor(np.array(pd.read_csv(dataFile)),dtype=torch.float32)
    
    data=(data-mu)/sigma #Normalise the data
    #print(data.type())
    if is_titan: #If is titan model, we need to do the memory stuff
        with torch.set_grad_enabled(True):
            start=time.time()
            predictions=torch.argmax(model.forward(data)[:-2],dim=-1) #Get the class positions
            model.layer_dict["Neural_Memory"].apply_cached_updates()
            model.layer_dict["Neural_Memory"].reset_computational_history()
            end=time.time()
    else:
        start=time.time()
        predictions=torch.argmax(model.forward(data)[:-2],dim=-1) #Get the class positions
        end=time.time()
    TotalTime=TotalTime+end-start

    #Get the prediction counts for each type:
    counts=np.zeros(4)
    for i in range(4):
        counts[i]=int(torch.sum(predictions==i))

    overall_prediction=np.argmax(counts)

    #Now we compare the overall prediction to the one that is in the annotations

    true_annotation=np.array(annotations["Their_label"])[anno_index]

    true_annotation=np.arange(3)[true_annotation==posLabels]

    #Add 1 to the confusion matrix in the right box
    ConfMat[true_annotation,overall_prediction]=ConfMat[true_annotation,overall_prediction]+1 #The row is the true value, the column is the predicted
    NewConfMat[true_annotation,:]=NewConfMat[true_annotation,:]+counts

NewConfMat=np.round(NewConfMat/np.array(np.sum(NewConfMat,axis=1))[:,np.newaxis]*100,2)#Convert to proportions of heartbeats

print(f"Results for {experiment_name}:")

print(f"Time taken for the forward passes: {TotalTime//60} mins, {TotalTime-60*TotalTime//60} seconds")

#print(f"Confusion Matrix: \n {ConfMat}")

print(f"Percentages of heartbeats classified in the test: \n {NewConfMat}")