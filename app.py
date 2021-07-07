from os import truncate
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tensorflow.python.keras.backend import dtype
from tqdm.notebook import tqdm as tqdm
import gensim
import string
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from traitlets.traitlets import default
from tensorflow import keras
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, SpatialDropout1D, Activation
from keras.layers import Conv1D, Bidirectional, GlobalMaxPool1D, BatchNormalization
from keras.models import Model, Input, Sequential, load_model
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from datetime import datetime as dt
from keras.optimizers import Adam
from keras import backend as K
from sklearn.metrics import classification_report

def load_css(css_file):
    with open(css_file) as f:
        st.markdown('<style>{}</style>'.format(f.read()),unsafe_allow_html = True)

def load_images(image_name):
    img = Image.open(image_name)
    return st.image(img, width=200)

######## PREPROCESSING DATA ########

#### Data Cleaning
def data_cleaning_process(text):
    text = re.sub(r'\\x..',"", str(text))
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\', "") #remove tab, new line, and back slice
    text = re.sub('  +', ' ', str(text)) # Remove extra spaces
    text = text.encode('ascii','replace').decode('ascii') #remove non ASCII; emoticon, chinese word, etc.
    text = re.sub('RT',' ',text) # Remove every retweet symbol
    text = re.sub('USER',' ',str(text))
    text = re.sub('URL',' ',text)
    text = re.sub('Retweeted',' ',text)
    text = re.sub('&amp;',' ',text)
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    text = re.sub(r'\d+', ' ', str(text))
    text = re.sub(' amp ',' ',text)
    text = text.translate(str.maketrans('','',string.punctuation))
    text = re.sub('\s+',' ', text)
    return text

#### Case Folding
def lowercase(text):
    return text.lower()
def case_folding(text):
    text = text.apply(lowercase)
    return text

#### Normalisasi Teks
def normalize_alay(text):
    alay_dict = pd.read_csv('./data/new_kamusalay.csv', encoding='latin-1', header=None)
    alay_dict = alay_dict.rename(columns={0: 'original', 1: 'replacement'})
    alaydict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
    return ' '.join([alaydict_map[word] if word in alaydict_map else word for word in text.split(' ')])

#### Stopword Removal
def remove_stopword(text):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    return stopword.remove(text)
def stopword_removal(text):
    text = text.apply(remove_stopword)
    return text

#### Tokenization
def split_token(text):
    return text.split()

def tokenization(token_result, text):
    proses = split_token(text)
    token_result.append(proses)
    text = token_result
    return token_result

### Drop blank space record
def drop_blank(text):
    text.replace('', np.nan, inplace=True)
    text.dropna(inplace=True)

############### TOKENIZER ###############
def tokenizer(token_result):
    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(token_result)
    sequences = tokenizer_obj.texts_to_sequences(token_result)

    return(sequences)

############### GLOVE ###############
def glove_to_w2v():
    glovefilename = "glove_tweet_100_win2.txt"
    glove_file = datapath(glovefilename)
    tmp_file = get_tmpfile("w2vec_glove_tweet.txt")

    glove2word2vec(glove_file, tmp_file)
    return(tmp_file)

def glove(token_result, embedding_dims):
    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(token_result)
    word_index = tokenizer_obj.word_index
    
    tmp_file = glove_to_w2v()

    embeddings_index = {}
    word2vec_file = open(os.path.join('', tmp_file), encoding = "utf-8")

    for line in word2vec_file:
        values = line.split()
        word = values[0]
        coefficient = np.asarray(values[1:])
        embeddings_index[word] = coefficient
    word2vec_file.close()

    num_words = len(word_index)+1
    embedding_matrix = np.zeros((num_words, embedding_dims))

    for word, i  in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return(embedding_matrix)

############### ACCURACY ###############
def accuracy(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None

        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1.0
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )

        acc_list.append(tmp_a)
    full = list(filter(lambda x: x == 1.0, acc_list))
    partial = list(filter(lambda x: x == 0.5, acc_list))
    none = list(filter(lambda x: x == 0.0, acc_list))
    acc = np.mean(acc_list)
    return  acc

###############

def main():
    # load data
    df = pd.read_csv('./data/simulasi.csv', encoding='latin-1')
    
    page = st.sidebar.selectbox("Menu", ["Tentang", "Preprocess Data", "Prediksi Tweet", "Simulasi Training", "Sample Data"])
    
    token_result = []
    embedding_dims = 100
    batch_size2 = 128


    load_css("style.css")
    
    if page =="Tentang":
        st.markdown('''
        <h1 class="title">Klasifikasi Multi-label Hate Speech Menggunakan Metode BiLSTM</h1>
        <h3 class="subtitle"><b>Tentang Penelitian Ini</b></h3>
        <p class="p">
            <b>Judul Penelitian<br></b>
            Klasifikasi Teks Multi-label Ujaran Kebencian Berbahasa Indonesia di Media Sosial Twitter Menggunakan Metode <i>Bidirectional Long Short-term Memory</i> (BiLSTM)<br>
        </p>
        <br>
        <p class="p">
            <b>Abstrak</b><br>
            Sejak menjadi salah satu produk teknologi paling digemari masyarakat, mengemukakan pendapat melalui media sosial menjadi sesuatu yang amat mudah. Siapa pun dapat memberikan pendapatnya mengenai sesuatu di sana. Sayangnya, kemudahan tersebut kemudian juga menjadi sebuah bumerang dengan semakin mudahnya pula untuk menggaungkan ujaran kebencian. Hal ini menjadi salah satu sisi gelap dari hadirnya media sosial, ujaran kebencian dapat memberikan bahaya yang fatal mulai dari kekerasan, konflik sosial, bahkan sampai pada penghilangan nyawa suatu individu atau kelompok yang menjadi sasarannya. Oleh karena itu, mencegah terjadinya bahaya yang dapat ditimbulkan oleh ujaran kebencian menjadi hal yang harus dilakukan.
            <br><br>
            Penelitian ini dilakukan sebagai upaya yang dapat dilakukan untuk mencegah dampak-dampak buruk dati ujaran kebencian. Penelitian ini akan menerapkan klasifikasi teks multi-label dengan menggunakan algoritma Bidirectional Long Short-term Memory (BiLSTM). Klasifikasi multi-label ini akan melabeli setiap data dataset ujaran kebencian yang bersumber dari Twitter dengan 12 label mengenai ujaran kebencian. Dari penelitian tersebut, hasil terbaik berhasil diraih oleh model dengan hyperparameter berupa epoch sebanyak 10 iterasi, dimensi kata 100, learning rate sebesar 0,005, jumlah unit 100, dan terakhir threshold klasifikasi sebesar 0,42. Model BiLSTM tersebut berhasil memberikan performa akurasi sebesar 82,31%, precision sebesar 83,41%, recall sebesar 87,28%, dan F1-score sebesar 85,30%.<br><br>
        </p>
        ''', unsafe_allow_html=True)

        expander = st.beta_expander('Informasi Data')
        with expander:
            st.markdown('''
            <div id="expander">
                <p class="p">
                    Pada penelitian ini, dataset yang digunakan diambil dari salah satu penelitian mengenai ujaran kebencian berbahasa Indonesia yang berjudul “Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter”. Dataset tersebut dapat diakses pada akun Github milik peneliti yang menyediakan data tersebut secara publik (https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection). Isi dari dataset tersebut diambil dari Twitter menggunakan library Tweepy pada periode waktu 20 Maret – 10 September 2018. 
                    <br><br>
                    Dari pengambilan data tersebut didapat sebanyak 13.169 tweet pada dataset tersebut dilabeli oleh 12 label. Kedua belas label tersebut, antara lain: HS, Abusive, HS_Individual, HS_Group, HS_Religion, HS_Race, HS_Physical, HS_Gender, HS_Other, HS_Weak, HS_Moderate, dan HS_Strong. Dataset tersebut secara garis besar terbagi menjadi ujaran kebencian dan bukan ujaran kebencian. Jumlah tweet yang dilabeli sebagai ujaran kebencian atau dengan label HS adalah 5.561 tweet dan sisanya sebanyak 7.608 tweet berupa bukan ujaran kebencian.
                    <br><br>
                    Berikut merupakan cuplikan dari dataset tersebut:
                </p>
            </div>
            ''',unsafe_allow_html=True)
            st.write(df.head(5))

        expander = st.beta_expander('Tentang Penulis dan Pembimbing')
        with expander:
            st.markdown('''
            <div id="expander">
                <h4 class="subtitle"><b>Penulis</b></h4>
                <p class="p">
                    Refa Annisatul Ilma<br>
                    140810170060<br>
                    Teknik Informatika FMIPA Unpad
                </p>
            </div>

            <div id="expander">
                <h4 class="subtitle"><b>Pembimbing</b></h4>
                <p class="p">
                    <b>Pembimbing 1</b><br>
                    Setiawan Hadi, M.Sc,Cs<br>
                    NIP 19620701 199302 1 001<br><br>
                    <b>Pembimbing 2</b><br>
                    Afrida Helen, M.Kom.<br>
                    NIP 19650128 199703 2 001<br><br>
                </p>
            </div>
            ''', unsafe_allow_html=True)

    elif page == "Preprocess Data":
        # st.title("Preprocessing Data")
        st.markdown('''
        <h1 class="title"><i>Preprocessing Data</i></h1>
        
        <p>
            Data preprocessing adalah tahap yang dilakukan untuk mengekstraksi data menjadi format yang sesuai dengan kebutuhan pengolahan. Preprocessing diperlukan untuk menyesuaikan data real menyesuaikan kebutuhan dari algoritma deep learning. Pada penelitian ini, preprocessing data yang diaplikasikan antara lain: Data Cleaning, Case Folding, Normalisasi Teks, Stopword Removal, dan Tokenisasi.
        </p>
        <b>Cara Menggunakan:</b><br>
        <div id="list">    
            <ol>
                <li>Masukkan kalimat yang diinginkan</li>
                <li>Tekan tombol "Preprocess"</li>
                <li>Lihat hasil pada expander sesuai dengan fitur preprocess
            </ol>
        </div>
        ''', unsafe_allow_html=True)

        tweet = st.text_input(label='Masukkan tweet')
        preprocess = st.button('Preprocess')

        st.markdown('''
        <h4>Hasil Preprocessing Data</h4><br>
        ''', unsafe_allow_html=True)
        expander = st.beta_expander('Data Cleaning')
        with expander:
            st.markdown('''
            <p>Data cleaning adalah operasi dasar yang harus dilakukan untuk membersihkan data mentah dari elemen-elemen yang tidak seharusnya ada dan mengganggu proses pengolahan data. Elemen-elemen tersebut di antaranya yaitu URL, hastag, mention, bahkan emoticon.</p>
            ''', unsafe_allow_html=True)
            cleaned = data_cleaning_process(tweet)
            st.markdown('<b>Hasil Data Cleaning:</b>', unsafe_allow_html=True)
            if preprocess:
                st.write(cleaned)
        expander = st.beta_expander('Case Folding')
        with expander:
            st.markdown('<p>Case folding merupakan proses yang menyeragamkan semua huruf dalam data yang dimiliki menjadi lowercase atau huruf kecil.</div>', unsafe_allow_html=True)
            case_fold = lowercase(cleaned)
            st.markdown('<b>Hasil Case Folding:</b>', unsafe_allow_html=True)
            if preprocess:
                st.write(case_fold)
        expander = st.beta_expander('Normalisasi Teks')
        with expander:
            st.markdown('<p>Normalisasi teks adalah proses mengubah kata-kata tidak baku menjadi kata-kata baku sehingga mengurangi redundansi fitur pada dataset.</p>', unsafe_allow_html=True)
            st.markdown('<b>Hasil Normalisasi Teks:</b>', unsafe_allow_html=True)
            normalized = normalize_alay(case_fold)
            if preprocess:
                st.write(normalized)
        expander = st.beta_expander('Stopword Removal')
        with expander:
            st.markdown('<p>Stopword removal adalah penghapusan kata-kata stopword dari data sehingga dapat mereduksi dimensi dari data yang dimiliki. Stopword yang digunakan pada penelitian ini berdasarkan pada library Sastrawi.</p>', unsafe_allow_html=True)
            no_stopword = remove_stopword(case_fold)
            st.markdown('<b>Hasil Stopword Removal:</b>', unsafe_allow_html=True)
            if preprocess:
                st.write(no_stopword)
        expander = st.beta_expander('Tokenization')
        with expander:
            st.markdown('Tokenization adalah sebuah proses yang digunakan untuk memotong sequence of strings menjadi beberapa bagian yang dapat berupa kata, kata kunci, frasa, simbol, atau bahkan karakter dalam string.', unsafe_allow_html=True)
            tokenize = tokenization(token_result, tweet)
            st.markdown('<b>Hasil Tokenization:</b>', unsafe_allow_html=True)
            if preprocess:
                st.table(tokenize)
        expander = st.beta_expander('Seluruh Preprocessing')
        with expander:
            st.markdown('Proses ini akan menampilkan hasil yang didapat ketika seluruh tahap preprocessing diaplikasikan.', unsafe_allow_html=True)
            token_result = []
            cleaned = data_cleaning_process(tweet)
            case_fold = lowercase(cleaned)
            normalized = normalize_alay(case_fold)
            no_stopword = remove_stopword(normalized)
            tokenize = tokenization(token_result, no_stopword)
            st.markdown('<b>Hasil Seluruh Preprocessing:</b>', unsafe_allow_html=True)
            if preprocess:
                st.table(tokenize)

    elif page == "Prediksi Tweet":
        st.markdown('''
        <h1 class="title">Prediksi <i>Tweet</i></h1>
        <p>
            Halaman ini digunakan untuk memprediksi tweet dengan model yang telah dilatih sebelumnya.
        </p>
        <b>Cara Menggunakan:</b><br>
        <div id="list">    
            <ol>
                <li>Masukkan kalimat atau tweet yang ingin diprediksi</li>
                <li>Tekan enter pada keyboard</li>
                <li>Hasil akan keluar langsung setelahnya dan menunjukkan label mana saja yang mengategorikan tweet/kalimat tersebut</li>
            </ol>
        </div>
        ''', unsafe_allow_html=True)
        tweet = st.text_input('Masukkan tweet')
        predict = st.button('Predict')

        st.markdown('<b>Hasil Prediksi:</b>', unsafe_allow_html=True)
        if predict:
            cleaned = data_cleaning_process(tweet)
            case_fold = lowercase(cleaned)
            normalized = normalize_alay(case_fold)
            no_stopword = remove_stopword(normalized)
            tokenize = tokenization(token_result, no_stopword)
            # st.text('hasil preprocess')
            # st.write(tokenize)

            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer_obj = pickle.load(handle)
            X = tokenizer_obj.texts_to_sequences(tokenize)
            X = pad_sequences(X, maxlen=100, value=0)
            # st.text('hasil tokenizer')
            # st.write(X)
            embedding_matrix = glove(tokenize, embedding_dims=embedding_dims)
            # st.text('embedding matrix')
            # st.write(embedding_matrix)

            model = load_model('./model-fix.h5', compile=False)
            y_pred = model.predict(X)
            # st.text('matrix hasil pred')
            # st.write(y_pred)
            y_pred[y_pred < 0.42] = 0
            y_pred[y_pred >= 0.42] = 1
            for i in range(y_pred.shape[0]):
                set_pred = set(np.where(y_pred[i])[0])

            # st.write(set_pred)
            label = list(set_pred)
            # st.text(label)
            if label == []:
                st.write('Tidak ada label yang terprediksi.')
            else: 
                for i in range(len(label)):
                    if label[i] == 0:
                        st.write('Hate Speech')
                    elif label[i] == 1:
                        st.write('Abusive')
                    elif label[i] == 2:
                        st.write('HS_Individual')
                    elif label[i] == 3:
                        st.write('HS_Group')
                    elif label[i] == 4:
                        st.write('HS_Religion')
                    elif label[i] == 5:
                        st.write('HS_Race')
                    elif label[i] == 6:
                        st.write('HS_Physical')
                    elif label[i] == 7:
                        st.write('HS_Gender')
                    elif label[i] == 8:
                        st.write('HS_Other')
                    elif label[i] == 9:
                        st.write('HS_Weak')
                    elif label[i] == 10:
                        st.write('HS_Moderate')
                    elif label[i] == 11:
                        st.write('HS_Strong')
                    else:
                        st.write('Tidak ada label yang terprediksi.')

    elif page == "Simulasi Training":
        st.markdown('''
        <h1 class="title">Simulasi <i>Training</i></h1>

        <p>Halaman ini akan mensimulasikan proses training yang dilalui pada saat melakukan penelitian ini. Jumlah data yang digunakan pada penelitian ini hanya 10% dari data sebenarnya, sebanyak 1.137 data, sehingga tidak memerlukan waktu eksekusi yang lama. Selain itu, pada halaman ini hyperparameter untuk training dapat diatur sesuai dengan keinginan.</p>
        
        <b>Cara Menggunakan:</b><br>
        <div id="list">    
            <ol>
                <li>Masukkan hyperparameter yang ada untuk di-assign ke dalam pengaturan</li>
                <li>Tekan tombol "Train", tunggu sesaat untuk model training.</li>
                <li>Hasil akan keluar berupa lama waktu training yang dibutuhkan dan hasil rata-rata dari 10-fold cross validation untuk masing-masing metrices evaluasi: accuracy, precision, recall, dan f1-score</li>
            </ol>
        </div>
        ''', unsafe_allow_html=True)

        form = st.form(key='my-form')
        # name = form.text_input('Enter your name')
        num_epochs = form.number_input('Epoch', min_value=1, max_value=20)
        max_len = form.number_input('Dimensi Data', min_value=10, max_value=200)
        learning_rate = form.number_input('Learning Rate',min_value=0.005, max_value=0.5)
        blstm_unit = form.number_input('LSTM Unit', min_value=10, max_value=100)
        threshold = form.slider('Threshold', min_value=0.01, max_value=1.00)
        submit = form.form_submit_button('Train')
        
        # preprocessing
        df['Tweet'] = df['Tweet'].apply(data_cleaning_process)
        df['Tweet'] = df['Tweet'].apply(lowercase)
        df['Tweet'] = df['Tweet'].apply(normalize_alay)
        df['Tweet'] = df['Tweet'].apply(remove_stopword)
        
        for desc in df['Tweet']:
            tokenization(token_result,desc)
        df['Tweet'] = token_result

        # tokenizer
        sequences = tokenizer(token_result)
        X = pad_sequences(sequences)
        y = df[["HS","Abusive","HS_Individual","HS_Group","HS_Religion","HS_Race", "HS_Physical", "HS_Gender","HS_Other","HS_Weak","HS_Moderate","HS_Strong"]].values
        
        # glove
        embedding_matrix = glove(token_result, embedding_dims=embedding_dims)

        # hyperparameter
        val_split = 0.1

        st.markdown('<b>Hasil Training: </b>', unsafe_allow_html=True)
        if submit:            
            with st.spinner('Training sedang berlangsung....'):
                # model bilstm
                BiLSTM_model = Sequential([
                Embedding(input_dim =embedding_matrix.shape[0], 
                          input_length=max_len, 
                          output_dim=embedding_matrix.shape[1],
                          weights=[embedding_matrix], 
                          trainable=False),
                SpatialDropout1D(0.5),
                Bidirectional(LSTM(blstm_unit, return_sequences=True)),
                BatchNormalization(),
                Dropout(0.5),
                GlobalMaxPool1D(),
                Dense(100, activation = 'relu'),
                Dense(12, activation = 'sigmoid')
                ])

                # # fit model
                callback = EarlyStopping(monitor="val_loss",
                                        min_delta=0,
                                        patience=4,
                                        verbose=0,
                                        mode="auto",
                                        baseline=None,
                                        restore_best_weights=True)

                start = dt.now()

                kfold = MultilabelStratifiedKFold(n_splits = 10, 
                                          random_state = 42, 
                                          shuffle = True)

                cvscores = []
                f1scores = []
                recalls = []
                precisions = []
                fold_var = 1
                for train, test in kfold.split(X, y):
                    X_tra, X_val = X[train], X[test]
                    y_tra, y_val = y[train], y[test]
    
                    print("ke - " + str(fold_var))
                    model = BiLSTM_model
                    model.compile(loss='binary_crossentropy', 
                                  optimizer=Adam(learning_rate), 
                                  metrics=['accuracy'])
                    model.fit(X_tra, y_tra, 
                    batch_size=batch_size2, 
                    epochs=num_epochs, 
                    validation_data=(X_val, y_val),
                    callbacks=[callback])
    
                    # predict tweet
                    y_pred = model.predict(X_val)
                    y_pred[y_pred < threshold] = 0
                    y_pred[y_pred >= threshold] = 1
    
                    # evaluation metrices
                    acc = accuracy(y_val, y_pred)
                    precision = precision_score(y_val, y_pred, average='micro')
                    recall = recall_score(y_val, y_pred, average='micro')
                    f1score = f1_score(y_val, y_pred, average='micro')
                    recalls.append(recall)
                    precisions.append(precision)
                    f1scores.append(f1score)
                    cvscores.append(acc)
                    fold_var += 1
            avg_acc = np.mean(cvscores)*100
            avg_pre = np.mean(precisions)*100
            avg_rec = np.mean(recalls)*100
            avg_f1s = np.mean(f1scores)*100

            st.write('Accuracy: %2f' % avg_acc, '%')
            st.write('Precisions: %2f' % avg_pre, '%')
            st.write('Recall: %2f' % avg_rec, '%')
            st.write('F1-score: %2f' % avg_f1s, '%')
            end = dt.now()
            total = end-start
            st.write('Waktu Eksekusi: ', total)


    elif page == "Sample Data":
        st.markdown('''
        <h1>Sample Data untuk Pengujian</h1>
        ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()