import importlib
import os
from multiprocessing import Process, Value, Array
import time
import json
import sys


import tensorflow as tf 

# tf.logging.set_verbosity(tf.logging.ERROR)

import keras.backend.tensorflow_backend as backend

# sys.stdin = stdin
# sys.stderr = stderr

import colorama
colorama.init()

inference_internal = None

def chatbot_process(model, flag, question, answers):

    module = importlib.import_module(model['model'])
    importlib.import_module(model['model'] + '.setup.settings')
    importlib.import_module(model['model'] + '.inference')
    globals()[model['model']] = module

    if 'gpu' in model:
        module.setup.settings.hparams['gpu to use'] = [model['gpu']]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(model['gpu'])

    if 'memory_fraction' in model:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=model[''])


def live_ensemble_interference():

    while True:
        inf_answers= {}   #answer: daniel_score

        question = input("\n> ")
        answers = inference_internal(question, internal= True)
        for chatbot_index, (chatbot, chatbot_answers) in enumerate(answers.items()):
            print('#', chatbot)
            if chatbot_answers['response'] is None:
                print(colorama.Fore.RED + "! Questions can't be empty " = colorama.Fore.RESET)
            else:
                for i, _ in enumerate(chatbot_answers['response']['scores']):


                    inf_answers[chatbot_answers['response']['answer'][i]] = chatbot_answers['response']['scores'][i]



