import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from PyDictionary import PyDictionary

# Run as: 
# python -W ignore Main.py

print("--------------")

def rgb2gray(rgb) :
	return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

def main():

    file = open("word.txt","r")
    guess = str(file.read()) # Read from a file

    key = "b28515ae-24a1-4796-be7c-e5cda4b47014"
    word = guess +"?"
    URL = "https://www.dictionaryapi.com/api/v3/references/sd2/json/"
    URL += word
    URL += key 
    PARAMS = {'word': word,'key': key}
    r = requests.get(url = URL, params = PARAMS) 
    definitions = []


    if r.status_code == 200:
     
        r.encoding = 'utf-8'
        data = r.json()
      
        try: 
            #definition is found
            deff = str(data[0]['def'][0]['sseq'][0])
            deff.split('*bc}')
            print('Definition found')    
            print(deff)

        except:
            #iterates through the definitions, if definition is not found
            #could make simple script that compares length (40% weight), and the number of similar letters (60% weight) to
            #determine the most likely match, then select that word

            print('Definition NOT found, perhaps it is one of these words:')
            for x in range(len(data)):
                print(str(x) + data[x])     
            
            print(guess[0:len(guess)])

    if r.status_code == 204:
        print('No content')
    if r.status_code == 404:
        print('Not found')


if __name__ == "__main__":
	main()