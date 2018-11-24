from src.data_generator import data_gen
from src.models import *
from src.data_generator import vis_train_features, plot_raw_audio
from src.utils import int_sequence_to_text
from src.wer import wer
import numpy as np
from keras import backend as K
from IPython.display import Audio
from IPython.display import Markdown, display
from IPython.display import Audio

def predict(index, partition, model, verbose=True):
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        model (Model): The acoustic model
    """
    # load the train and test data

    # obtain the true transcription and the audio features 
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
        
    # obtain and decode the acoustic model's predictions
    prediction = model.predict(np.expand_dims(data_point, axis=0))
    output_length = [model.output_length(data_point.shape[0])]
    pred_ints = (K.eval(K.ctc_decode(
                prediction, output_length, greedy=False)[0][0])+1).flatten().tolist()
    predicted = ''.join(int_sequence_to_text(pred_ints)).replace("<SPACE>", " ")
    if verbose:
        display(Audio(audio_path, embed=True))
        print('Truth: ' + transcr)
        print('Predicted: ' + predicted)
    wer_val = wer(transcr, predicted)
    print("wer: %d" % wer_val)