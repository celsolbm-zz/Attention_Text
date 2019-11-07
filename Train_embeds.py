from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import argparse
import collections
import hashlib
import math
import os
import random
import sys
from tempfile import gettempdir
import zipfile
import pickle
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
data_index = 0
import io
import pandas as pd

#carrega os dados para serem treinados em um data frame
dados = pd.read_csv('train.csv')

#carrega os anuncios em portugues (fiz apenas para os em portugues)
port = dados.loc[dados.language == 'portuguese']

def tokens(data): #utilizado para separar as frases em palavras separadas
    ret = []
    for s in data.iterrows():
        ret +=s[1][0].split()
    return ret

def data_prepro2(data): #pré-processamento para limpar os dados
    k = len(data)
    i=0
    for s in data.iterrows():
        result = ''.join([i for i in s[1][0] if not ( (i.isdigit() or ( not i.isalpha() )) and i!=' ' ) ]) #remove numero, e digitos unicos
        result = result.lower()
        i=0
        result = ' '.join(i for i in result.split() if not (i.isalpha() and len(i)<2))
        s[1][0] = result

"""
Função vestigial, nao eh mais utilizada por isso foi comentada
def data_prepro3(data): #pré-processamento para limpar os dados
    k = len(data)
    i=0
    z = 0
    for s in data:
        for i in s:
            if i == '_':
                i = ' '
        result = ''.join([i for i in s if not ( (i.isdigit() or ( not i.isalpha() )) and i!=' ' ) ]) #remove numero, e digitos unicos
        result = result.lower()
        s = result
        z+=1
        print(z)
    return data
"""

def build_dataset(words, n_words): #criar o dataset em tensorflow
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1)) #pega as n_words mais comuns, evita entao ter palavras bizarras
    dictionary = {word: index for index, (word, _) in enumerate(count)} #dicionario para fazer a conversao de int - palavra
    data = []
    unk_count = 0
    for word in words:
      index = dictionary.get(word, 0)
      if index == 0:  # dictionary['UNK']
        unk_count += 1
      data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


vocabulary_size = 30000 #numero de palavras unicas que irá ter no vocabulario

data_prepro2(port) #faz a preparacao dos dados, tirando numeros, letras isoladas e simbolos como barras e underlines
tok = tokens(port) #tokeniza  as palavras para formar uma lista

data, count, unused_dictionary, reverse_dictionary = build_dataset( #constroi o dataset
      tok, vocabulary_size)


def generate_batch(batch_size, num_skips, skip_window): #gera a batch para um formato que o tensorflow entende
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # o tamanho da janela de previsao do skio
    buffer = collections.deque(maxlen=span)  # magia negra que precisei colocar, sei la pra que serve saporra mas soh funciona com isso
    if data_index + span > len(data):
      data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
      context_words = [w for w in range(span) if w != skip_window]
      words_to_use = random.sample(context_words, num_skips)
      for j, context_word in enumerate(words_to_use):
        batch[i * num_skips + j] = buffer[skip_window]
        labels[i * num_skips + j, 0] = buffer[context_word]
      if data_index == len(data):
        buffer.extend(data[0:span])
        data_index = span
      else:
        buffer.append(data[data_index])
        data_index += 1
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch_size = 128 
embedding_size = 256  # Dimensao do embedding vector. Hyperparametro que pode ser mudado e ajustado para cada modelo
skip_window = 1  # A janela ja referida anteriormente
num_skips = 2  # quantas vezes o label vai ser reusado
num_sampled = 64  # Para o negative sampling, usando 64 pq eh o recomendado.

valid_size = 16  # As palavras quse seram utilziadas para avaliar a similaridade.
valid_window = 100  #janeal para a sampling das palavras de similaridade.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()


current_path = os.path.dirname(os.path.realpath(sys.argv[0])) #formalidadaes para rodar o codigo, meio inutil
parser = argparse.ArgumentParser()
parser.add_argument( #parser para passar os dados para o treino
      '--log_dir',
      type=str,
      default=os.path.join(current_path, 'log'),
      help='The log directory for TensorBoard summaries.')

flags, unused_flags = parser.parse_known_args()

log_dir = flags.log_dir

if not os.path.exists(log_dir): #usado mais para o caso do colab, mas na maquina local nao precisa.
    os.makedirs(log_dir)


with graph.as_default():
    # Input data
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    # Utilizando CPU pq tava fazendo no meu mac, nao tem tanto ganho assim com gpu ai deixei assim mesmo
    with tf.device('/cpu:0'):
        # Cria os embeddings
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
      # Constroi as variaveis para a NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))
    # Adiciona o valor da loss
    tf.summary.scalar('loss', loss)
    # COnstroi o optimizer usando gradient descent e uma learning rate de 1.0.
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    # Calcula a similaridade de cada embedding
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
    # Junta td
    merged = tf.summary.merge_all()
    # INicializador
    init = tf.global_variables_initializer()
    # Meio inutil, codigo vestigial
    saver = tf.train.Saver()

num_steps = 400001

with tf.compat.v1.Session(graph=graph) as session:
    # Gambiarra do tensorflow, abre esse writer para escrever em um arquivo para o futuro
    writer = tf.summary.FileWriter(log_dir, session.graph)
    # inicializa as variaveis
    init.run()
    print('Initialized')
    average_loss = 0
    for step in xrange(num_steps):
      batch_inputs, batch_labels = generate_batch(batch_size, num_skips,
                                                  skip_window)
      feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
      # Mais uma magia negra do tensorflow, define a metadata
      run_metadata = tf.RunMetadata()

      _, summary, loss_val = session.run([optimizer, merged, loss],
                                         feed_dict=feed_dict,
                                         run_metadata=run_metadata)
      average_loss += loss_val
      writer.add_summary(summary, step)

      if step == (num_steps - 1):
        writer.add_run_metadata(run_metadata, 'step%d' % step)
      if step % 2000 == 0:
        if step > 0:
          average_loss /= 2000
 
        print('Average loss at step ', step, ': ', average_loss)
        average_loss = 0

      if step % 10000 == 0:
        sim = similarity.eval()
        for i in xrange(valid_size):
          valid_word = reverse_dictionary[valid_examples[i]]
          top_k = 8  # numero de classes utilizados para a comparacao
          nearest = (-sim[i, :]).argsort()[1:top_k + 1]
          log_str = 'Nearest to %s:' % valid_word
          print(
              log_str,
              ', '.join([reverse_dictionary[nearest[k]] for k in range(top_k)]))
    final_embeddings = normalized_embeddings.eval()

    with open(log_dir + '/metadata.tsv', 'w') as f:
      for i in xrange(vocabulary_size):
        f.write(reverse_dictionary[i] + '\n')
    #salva o modelo
    saver.save(session, os.path.join(log_dir, 'model.ckpt'))
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

pickle.dump( final_embeddings, open( "embeds_final_np.p", "wb" ) )
pickle.dump( unused_dictionary, open( "unused_dic.p", "wb" ) )
pickle.dump( port, open( "port.p", "wb" ) )
