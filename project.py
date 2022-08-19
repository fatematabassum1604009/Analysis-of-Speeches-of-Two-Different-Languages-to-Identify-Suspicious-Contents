from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
new_model=load_model('text_classification.h5')
import speech_recognition as sr
r = sr.Recognizer()
tokenizer=Tokenizer()
with sr.Microphone() as source:
    print('-'*30)
    print('-'*30)
    print("\n\nHi !! Say something and I will try to understand what you are talking about !!\n")
    print('-'*30)
    print("\nListening :\n")
    print('-'*30)
    my_fileban = open("BengaliWordList_112.txt", "r",encoding = 'utf-8')
    databan = my_fileban.read()
    dictionaryban= databan.split("\n")
    my_fileban.close()
    my_fileeng = open("words_alpha.txt", "r")
    dataeng = my_fileeng.read()
    dictionaryeng= dataeng.split("\n")
    my_fileeng.close()
    counterb=0
    countere=0
    audio = r.listen(source)
    textb = r.recognize_google(audio,language='bn-BD')
    print(textb)
    splitsb = textb.split()
    for splitb in splitsb:
        if splitb in dictionaryban:
            counterb=counterb+1
    texte = r.recognize_google(audio,language='en-US')
    print(texte)
    splitse = texte.split()
    for splite in splitse:
        if splite in dictionaryeng:
            countere=countere+1
    if countere>counterb:
        text=texte
        language="English"
    else:
        text=textb
        language="Bengali"
    print(counterb)
    print(countere)
    print("\nYou said : {}".format(text))
    print(language)
    text=[text]
    print(text)
    text =tokenizer.texts_to_sequences(text)
    text_pad=pad_sequences(text,maxlen=78,padding='post')
    
    prediction= new_model.predict(text_pad)
    if int(prediction) == 0:
        print('\nNot Suspiious')
    elif int(prediction) == 1:
        print('\nSuspiious')    
