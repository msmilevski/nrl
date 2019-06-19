import preprocessing_util as preprocess
import rus_preprocessing_udpipe
import sys
import numpy as np

seed = np.random.seed(1113)

ids, descriptions, image_arrays = preprocess.initial_preprocess(sys.argv[3])

if sys.argv[1] == 'fastText':
    descriptions, rows_with_nan_desc = preprocess.preprocess_corpus(ids, descriptions, lemmatization=True)

if sys.argv[1] == 'rusVec':
    descriptions, rows_with_nan_desc = rus_preprocessing_udpipe.main(ids, desciptions)

random_image = preprocess.select_random_image(ids.tolist(), image_arrays, seed=seed)

if len(rows_with_nan_desc) > 0:
    random_image.drop(index=rows_with_nan_desc, inplace=True)

if sys.argv[2] == 'training':
    frequency = preprocess.create_word_frequency_document(descriptions['descriptions'], path_to_json_file=None)
    vocabulary = preprocess.generate_vocabulary(word_frequency_file=frequency, vocabulary_file=sys.argv[4],
                                                vocabulary_size=10000)
    preprocess.sentences_to_indecies(descriptions['itemID'].tolist(), descriptions['descriptions'], random_image['image_id'].tolist(),
                                     processed_text_file_path=sys.argv[5], vocab_file=vocabulary, max_length=100)
else:
    preprocess.sentences_to_indecies(descriptions['itemID'].tolist(), descriptions['descriptions'], random_image['image_id'].tolist(),
                                     processed_text_file_path=sys.argv[5], vocab_file=sys.argv[4], max_length=100)