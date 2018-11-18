from src.data_generator import AudioGenerator
from src.models import *
from src.data_generator import vis_train_features, plot_raw_audio
from src.utils import int_sequence_to_text
import numpy as np
from keras import backend as K
from IPython.display import Audio
from IPython.display import Markdown, display
from IPython.display import Audio

def predict(index, partition, model, model_name):
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        model (Model): The acoustic model
        model_name (str): Name of model. Path to saved acoustic model's weights expected in file model_name.h5
    """
    # load the train and test data

    data_gen = AudioGenerator(spectrogram=False)
    data_gen.load_train_data()
    data_gen.load_validation_data()

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
    model.load_weights('models/' + model_name + '.h5')
    prediction = model.predict(np.expand_dims(data_point, axis=0))
    output_length = [model.output_length(data_point.shape[0])]
    pred_ints = (K.eval(K.ctc_decode(
                prediction, output_length, greedy=False)[0][0])+1).flatten().tolist()
    
    # play the audio file, and display the true and predicted transcriptions
    print('True transcription:\n' + '\n' + transcr)
    print('-'*80)
    print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))
    print('-'*80)
    display(Audio(audio_path, embed=True))