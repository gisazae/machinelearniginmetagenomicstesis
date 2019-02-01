# Importación de módulos.

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# Resetear grafos
tf.reset_default_graph()

# Lectura del archivo de entrada

FILE_PATH = 'dataGen.csv'  # Path to .csv dataset
raw_data = pd.read_csv(FILE_PATH, sep=',', encoding='utf-8')  # Open raw .csv
raw_data = raw_data.loc[:, ~raw_data.columns.str.contains('^Unnamed')]

print(raw_data.head())
print("Data set base cargado satisfactoriamente...\n")

# ------------------------------------------------------------------------------
# Variables
print(len(raw_data.keys().tolist()))

Y_LABEL = "y"  # Nombre de la variable a ser predecida
KEYS = [i for i in raw_data.keys().tolist() if i != Y_LABEL]  # Número de predictores
N_INSTANCES = raw_data.shape[0]  # Número de instancias
N_INPUT = raw_data.shape[1] - 1  # Tamaño de la entrada
N_CLASSES = raw_data[Y_LABEL].unique().shape[0]  # Numero de clases (Grupos CTXM)
TEST_SIZE = 0.1  # Tamaño de set de entrenamiento (% of dataset)
TRAIN_SIZE = int(N_INSTANCES * (1 - TEST_SIZE))  # Tamaño de entrenamiento.
LEARNING_RATE = 0.001  # Taza de aprendizaje
TRAINING_EPOCHS = 50  # Número de epocas (pasos o ciclos de entrenamiento) - 400
BATCH_SIZE = 100  # Tamaño del lote.
DISPLAY_STEP = 20  # Muestra del progreso cada X epocas
HIDDEN_SIZE = 100  # Número de neuronas ocultas - 200
ACTIVATION_FUNCTION_OUT = tf.nn.tanh  # Función de activación.
STDDEV = 0.1  # Desviación Standar (para pesos iniciales aleatorios)
RANDOM_STATE = 100  # Estado aleatorio para prueba de entrenamiento dividido

print("Variables cargadas satisfactoriamente...\n")
print("Número de predictores \t%s" % (N_INPUT))
print("Número de clases \t%s" % (N_CLASSES))
print("Número de instancias \t%s" % (N_INSTANCES))
print("\n")
print("Métricas presentadas:\tPrecision\n")
# ------------------------------------------------------------------------------
# Carga de datos.
data = raw_data[KEYS].get_values()  # X data
labels = raw_data[Y_LABEL].tolist()  # y data
# Codificación en ejecución de las etiquetas.
# para dos clases
if N_CLASSES == 2:
    labels_ = np.zeros((N_INSTANCES, N_CLASSES))
    labels_[np.arange(N_INSTANCES), labels] = 1
# para tres o mas clases
if N_CLASSES >= 3:
    lb = preprocessing.LabelBinarizer()
    labels_ = lb.fit_transform(labels)

print(N_INPUT)
print(N_CLASSES)
print(labels_)

# prueba de entrenamiento dividida
data_train, data_test, labels_train, labels_test = train_test_split(data,
                                                                    labels_,
                                                                    test_size=TEST_SIZE,
                                                                    random_state=RANDOM_STATE)

print("Datos cargados y divididos satisfactoriamente...\n")
# ------------------------------------------------------------------------------

with tf.name_scope("NN_Construction"):
    # Construcción de la red neuronal

    # Parámetros de la red
    n_input = N_INPUT  # Etiquetas de entrada
    n_hidden_1 = HIDDEN_SIZE  # 1st capa
    n_hidden_2 = HIDDEN_SIZE  # 2nd capa
    n_hidden_3 = HIDDEN_SIZE  # 3rd capa
    n_hidden_4 = HIDDEN_SIZE  # 4th capa
    n_classes = N_CLASSES  # salida m clases

    # Tf placeholders
    X = tf.placeholder(tf.float32, [None, n_input], name="CTXM_input")
    y: object = tf.placeholder(tf.float32, [None, n_classes], name="Group_CTXM")
    dropout_keep_prob = tf.placeholder(tf.float32)


    def mlp(_X, _weights, _biases, dropout_keep_prob):
        layer1 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])), dropout_keep_prob,
                               name="Hidden_Layer_1")
        layer2 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer1, _weights['h2']), _biases['b2'])), dropout_keep_prob,
                               name="Hidden_Layer_2")
        layer3 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer2, _weights['h3']), _biases['b3'])), dropout_keep_prob,
                               name="Hidden_Layer_3")
        layer4 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer3, _weights['h4']), _biases['b4'])), dropout_keep_prob,
                               name="Hidden_Layer_4")
        out = ACTIVATION_FUNCTION_OUT(tf.add(tf.matmul(layer4, _weights['out']), _biases['out']), name="out")
        return out


    with tf.name_scope("Weights"):
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=STDDEV), name="Weight_input"),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=STDDEV), name="Weight_h2"),
            'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=STDDEV), name="Weight_h3"),
            'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], stddev=STDDEV), name="Weight_h4"),
            'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], stddev=STDDEV), name="Weight_out")
        }

    with tf.name_scope("Bias"):
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1]), name="bias_h1"),
            'b2': tf.Variable(tf.random_normal([n_hidden_2]), name="bias_h2"),
            'b3': tf.Variable(tf.random_normal([n_hidden_3]), name="bias_h3"),
            'b4': tf.Variable(tf.random_normal([n_hidden_4]), name="bias_h4"),
            'out': tf.Variable(tf.random_normal([n_classes]), name="bias_out")
        }
    # -------------------------------------------------------------------

    with tf.name_scope("NN_CTXM"):
        with tf.name_scope("Model"):
            # Construyendo el modelo
            pred = mlp(X, weights, biases, dropout_keep_prob)

        with tf.name_scope("Cost"):
            # pérdida y optimizador
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  # softmax loss

        with tf.name_scope("Optimization"):
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

        with tf.name_scope("Accuracy"):
            # Precisión
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print("Red construida satisfactoriamente...\n")
        print("Iniciando el entrenamiento...\n")

# ENTRENAMIENTO

# Inicializando variables
init_all = tf.initialize_all_variables()
# init_all = tf.global_variables_initializer

# Definición de escalares--------------------
# Create a summary to monitor cost tensor
tf.summary.scalar("Cost", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("Accuracy", accuracy)

# Definiendo histogramas

tf.summary.histogram("MLP", pred)
# Merge all summaries into a single op

merged_summary_op = tf.summary.merge_all()
# ------------------------------------------------------------------------------
# Lanzando la sesión
with tf.Session() as sess:
    sess.run(init_all)
    # Escribimos el grafo para tensorboard
    summary_writer = tf.summary.FileWriter("output", graph=tf.get_default_graph())
    summary_writer.flush()

    for epoch in range(TRAINING_EPOCHS):
        avg_cost = 0.
        total_batch = int(data_train.shape[0] / BATCH_SIZE)
        # Ciclo sobre los lotes.
        for i in range(total_batch):
            randidx = np.random.randint(int(TRAIN_SIZE), size=BATCH_SIZE)
            batch_xs = data_train[randidx, :]
            batch_ys = labels_train[randidx, :]
            # Entrenando usando datos en lotes.
            _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                     feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob: 0.9})
            # Calculando costo total.
            summary_writer.add_summary(summary, epoch * total_batch + i)
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob: 1.}) / total_batch

        # Mostrando el progreso
        if epoch % DISPLAY_STEP == 0:
            print("Epoch: %03d/%03d cost: %.9f" % (epoch, TRAINING_EPOCHS, avg_cost))
            train_acc = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob: 1.})
            print("Precisión del entrenamiento: %.3f" % (train_acc))



    # ------------------------------- Guardar el objeto entrenado-----------------------------------

    # Guardamos el objeto de sesión. Con un 90% de la bd entranarla. con el mejor performance.

    # saver.save(sess, 'C:\FTP\Tesis\def_NN_4C_200N_Tang\NN_CTXM_entrenada.ckpt') #Generación del modelo entrenado.

    # -----------------------------------------//---------------------------------------------------
    print("Fin del entrenamiento.\n")
    print("Probando...\n")
    # ------------------------------------------------------------------------------
    # Probando
    # -------------------------------------------------------------------------------------

    test_acc = sess.run(accuracy, feed_dict={X: data_train, y: labels_train, dropout_keep_prob: 1.})
    print("Prueba de precisión: %.3f" % (test_acc))

#Imprimiendo CTX-M con mayor probabilidad.
    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print("accuracy", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    prediction = tf.argmax(pred, 1)
    print("predictions", prediction.eval(feed_dict={X: data_train, dropout_keep_prob:1.}, session=sess))
    salida_NN = np.argmax([labels_train.T],axis=1)
    print(salida_NN[0,:])

#-------------------------
    sess.close()
    print("Sesión cerrada!")

    summary_writer.close()



# Definiendo curva ROC parte 1
"""
tp = tf.metrics.true_positives(labels=pred, predictions=accuracy)
fp = tf.metrics.false_positives(labels=pred, predictions=accuracy)
total = len(fp)

# Definiendo curva ROC parte 2 ------------------------------------------------
#actual = [1,1,1,0,0,0]
#predictions = [0.9,0.9,0.9,0.1,0.1,0.1]
#false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
false_positive_rate, true_positive_rate = roc_curve(tp, fp)

roc_auc = auc(false_positive_rate, true_positive_rate)
#Graficar
plt.title('Receiver Operating Characteristic')
plt.plot(fp, tp, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
"""