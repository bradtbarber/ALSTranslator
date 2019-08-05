import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests


def get_word(word):
    key = 'b28515ae-24a1-4796-be7c-e5cda4b47014'
    URL = 'https://www.dictionaryapi.com/api/v3/references/sd2/json/'
    URL += word + '?'
    URL += key 
    PARAMS = {'word': word,'key': key}
    r = requests.get(url = URL, params = PARAMS)
    
    return r 
    
def parse_definition(data):
    return str(data[0]['def'][0]['sseq'][0][0][1]['dt'][0][1]).replace('{bc}', '')

def check_word(word):
    #look up word argument from Dictionary.com API
    req_resp = get_word(word)

    #if API request succeeds
    if req_resp.status_code == 200:
        req_resp.encoding = 'utf-8'
        data = req_resp.json()
      
        try: 
            #definition is found
            deff = parse_definition(data)
            print('Definition found: ')    
            print(word + ' - ' + deff)
            return 0

        except:
            #definition not present
            return 1

    #else return status code
    elif r.status_code == 204:
        print('204: No content')
    elif r.status_code == 404:
        print('404: Not found')
    else:
        print(status_code + ': Error sending API request to Dictionary.com')
    return -1
