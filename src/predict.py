from src.data_generator import data_gen
from src.models import *
from src.data_generator import vis_train_features, plot_raw_audio
from src.utils import int_sequence_to_text
from src.wer import wer
import numpy as np
from keras import backend as K
from IPython.display import Audio
from IPython.display import Markdown, display
import time
import pickle


def predict(index, partition, model, verbose=True):
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        model (Model): The acoustic model
    """
    audio_path,data_point,transcr,prediction = predict_raw(index, partition, model)
    output_length = [model.output_length(data_point.shape[0])]
    pred_ints = (K.eval(K.ctc_decode(
                prediction, output_length, greedy=False)[0][0])+1).flatten().tolist()
    predicted = ''.join(int_sequence_to_text(pred_ints)).replace("<SPACE>", " ")
    wer_val = wer(transcr, predicted)
    if verbose:
        display(Audio(audio_path, embed=True))
        print('Truth: ' + transcr)
        print('Predicted: ' + predicted)
        print("wer: %d" % wer_val)
    return wer_val

def predict_raw(index, partition, model):
    """ Get a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        model (Model): The acoustic model
    """

    if partition == 'validation':
        transcr = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'train':
        transcr = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation"')
        
    prediction = model.predict(np.expand_dims(data_point, axis=0))
    return (audio_path,data_point,transcr,prediction)

def calculate_wer(model, model_name, partition, length):
    start = time.time()
    def wer_single(i):
        wer = predict(i, partition, model, verbose=False)
        if (i%10==0) and i>0:
            print("processed %d in %d minutes" % (i, ((time.time() - start)/60)))
        return wer
    wer = list(map(lambda i: wer_single(i), range(0, length)))
    print("Total time: %f" % ((time.time() - start)/60))
    with open('models/' + model_name + '_wer.pickle', 'wb') as handle:
        pickle.dump(wer, handle)
    return wer


def load_wer(model_name):
    return pickle.load(open("models/" + model_name + "_wer.pickle", "rb"))