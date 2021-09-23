from django.shortcuts import render,redirect
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
from .models import History

import numpy as np
tok_email=pickle.load(open('classifier/pickle/tokenizer_email.sav','rb'))
email_cnn=load_model('classifier/Emailmodel.h5')


tok_sms=pickle.load(open('classifier/pickle/tokenizer_sms.sav','rb'))
sms_cnn=load_model('classifier/Smsmodel.h5')


tok_youtube=pickle.load(open('classifier/pickle/tokenizer_youtube.sav','rb'))
youtube_cnn=load_model('classifier/ytmodel.h5')
yt_combined=pickle.load(open('classifier/yt_best_model.sav','rb'))


oneE=pickle.load(open('classifier/pickle/one_hot.sav','rb'))


# context={'designform':None,'departform':None,'directories':None,'addDirectory':None}


def home_page(request):
    return render(request, 'home.html')


def email(request):
    email_combined=pickle.load(open('classifier/email_best_model.sav','rb'))
    txt=[request.POST['data']]
    model=request.POST['model_name']
    if model =='Combined*':
        res=email_combined.predict(txt)
        return res[0]
    elif model == 'CNN-LSTM':
        seq_=tok_email.texts_to_sequences(txt)
        seq_=pad_sequences(seq_,50)
        c= email_cnn.predict(seq_)
    
        return oneE.inverse_transform(c)[0][0]
    else:
        return "PLEASE SELECT MODEL NAME"


    
    

def sms(request):
    txt=[request.POST['data']]
    sms_combined=pickle.load(open('classifier/sms_best_model.sav','rb'))
    
    txt=[request.POST['data']]
    model=request.POST['model_name']
    if model =='Combined*':

        res=sms_combined.predict(txt)
        return res[0]

    elif model == 'CNN-LSTM':

        seq_=tok_sms.texts_to_sequences(txt)
        seq_=pad_sequences(seq_,50)
        c= sms_cnn.predict(seq_)
        return oneE.inverse_transform(c)[0][0]

    else:

        return "PLEASE SELECT MODEL NAME"

def youtube(request):
    txt=[request.POST['data']]
    model=request.POST['model_name']
    if model =='Combined*':

        res=yt_combined.predict(txt)
        return res[0]

    elif model == 'CNN-LSTM':

        seq_=tok_youtube.texts_to_sequences(txt)
        seq_=pad_sequences(seq_,50)
        c= youtube_cnn.predict(seq_)
        
        return oneE.inverse_transform(c)[0][0]

    else:

        return "PLEASE SELECT MODEL NAME"

def predict(request):
    
    # text_to_seq = pickle.load(open('classifier/pickle/email_text_toseq.sav','rb'))
    txt=''
    y=None
    status=False
    if request.method=='POST':
        
        type_=request.POST['model_type']
        txt=request.POST['data']

        if type_ == 'SMS':
            y=sms(request)
            status=True
        elif type_ =='EMAIL':
            y=email(request)
            status=True
        elif type_=='YOUTUBE':
            y=youtube(request)
            status=True
        else:
            return render(request, 'predict.html' ,context={'result':"Please Select Model Type","data": txt})
        if status:
            his=History.objects.create(
                text=txt,
                model=str(type_+ " - " + request.POST['model_name']),
                result=y
            )
            his.save()


        return render(request, 'predict.html' ,context={'result':y, "data": txt})
    
    return render(request, 'predict.html',context={'result':None,"data": txt})

def history(request):
    his=History.objects.all()
    return render(request, 'history.html',context={'history':his})


      #Instantiate class.
      #Process whatever
      #Dump pickle file. This sets the module of the pickled file to be that of imported. i.e. 'car'