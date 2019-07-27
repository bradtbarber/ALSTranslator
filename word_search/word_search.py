import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests

print('--------------')

def get_word(word):
    key = 'b28515ae-24a1-4796-be7c-e5cda4b47014'
    URL = 'https://www.dictionaryapi.com/api/v3/references/sd2/json/'
    URL += word + '?'
    URL += key 
    PARAMS = {'word': word,'key': key}
    r = requests.get(url = URL, params = PARAMS)
    
    return r 
    

def main():
    #ensure a word string was provided as an argument
    if len(sys.argv) <= 1:
        print('No word argument provided. Exiting...')
        return -1

    #look up word argument from Dictionary.com API
    word = sys.argv[1]
    r = get_word(word)

    #if API request succeeds
    if r.status_code == 200:
        r.encoding = 'utf-8'
        data = r.json()
      
        try: 
            #definition is found
            deff = str(data[0]['def'][0]['sseq'][0][0][1]['dt'][0][1]).replace('{bc}', '')
            print('Definition found: ')    
            print(word + ' - ' + deff)

        except:
            #iterates through the definitions, if definition is not found
            #could make simple script that compares length (40% weight), and the number of similar letters (60% weight) to
            #determine the most likely match, then select that word

            print('Definition NOT found, perhaps you meant:')
            print(data[0])
            r = get_word(data[0])
            data2 = r.json()
            deff2 = str(data2[0]['def'][0]['sseq'][0])
            print('Definition found')    
            print(deff2)

    elif r.status_code == 204:
        print('No content')
    elif r.status_code == 404:
        print('Not found')
    else:
        print('Unable to send API request to Dictionary.com: ' + r.status_code)
        return -1

if __name__ == '__main__':
    main()
