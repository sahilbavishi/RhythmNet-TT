#This is the python file of the notebook, so we can run it from a terminal

#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#First, function to load the data
DataPath="mit_bih_csv_files/ecg_records/"
def RetrieveData(RecordNum):
    """Retrieves the ecg data from the record provided"""
    ecgData=pd.read_csv(DataPath+str(RecordNum)+"_ecg.csv")
    ecgAnno=pd.read_csv(DataPath+str(RecordNum)+"_annotations.csv")
    return np.array(ecgData),np.array(ecgAnno)

colNumforECG=0 #The column index for the ECG channel we are picking - either 0 or 1
Seconds=3 #The number of seconds we look at
MaxNumBeats=15 #The maximum number of beats per window

MaxNumBeats=MaxNumBeats*2
TimeFrame=Seconds*360

def ConvertAnnos(Anno,Intervals):
    """Takes annotation data and returns a labels within the intervals"""
    ReturnAnnotations=np.ndarray.flatten(Anno[Anno[:,1]<=TimeFrame])
    ReturnAnnotations=[np.pad(ReturnAnnotations,pad_width=(0,MaxNumBeats-len(ReturnAnnotations)))]
    for j,i in enumerate(Intervals[:-1]):
        AddArray=Anno[(Anno[:,1]<=Intervals[j+1])*(Anno[:,1]>=i)] #Get array of annotations and positions
        AddArray[:,1]=AddArray[:,1]-i #Adjust the position values so they are relative to the window
        AddArray=np.ndarray.flatten(AddArray) #Flatten array to append
        ReturnAnnotations=np.append(ReturnAnnotations,[np.pad(AddArray,pad_width=(0,MaxNumBeats-len(AddArray)))],axis=0) #Append the array
    return ReturnAnnotations


def ProcessECGData(Data,Anno):
    """Takes ECG data and the annotations and outputs a matrix with t second chunks with an annotation at the end"""
    #Intervals:
    Intervals=np.arange(TimeFrame,Data.shape[0],TimeFrame)
    NewData=Data[0:TimeFrame][:,np.newaxis].T #New data is our data that is segmented into 30 second chunks
    Labels=ConvertAnnos(Anno,Intervals)
    for j,i in enumerate(Intervals[:-1]):
        NewData=np.append(NewData,[Data[i:Intervals[j+1]]],axis=0)
    NewData=np.append(NewData,Labels,axis=1)
    return NewData

def RetrievePatientData(Num):
    Data,Annotation=RetrieveData(Num)
    return ProcessECGData(Data[:,colNumforECG],Annotation)

RetrievePatientData(101).shape


#Now we have the functions, we can do this for each number
#Numbers: 100-124, 200-234
import os #Just to get the file names
fileNames=os.listdir(DataPath)
df=pd.DataFrame(RetrievePatientData(100))
for fileIndex in np.arange(3,len(fileNames),3):
    Num=fileNames[fileIndex][:3]
    df2=pd.DataFrame(RetrievePatientData(Num))
    df=pd.concat([df,df2],ignore_index=True)



MaxNumofAnnotations=5 # This is the maximum number of beats that we want to classify in the time frame

df=df[df[TimeFrame+(1+MaxNumofAnnotations)*2]==0] #Gets rid of any lines that are more than those annotations

df=df.loc[:, (df != 0).any(axis=0)]


#Save it
df.to_csv("MIT3sec.csv")