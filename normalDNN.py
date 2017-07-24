import sys
import tensorflow as tf

class normal_dnn(object):
    def __init__(self, n_in, n_hidden, n_out):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out


    def inference(self, x, keep_prob):
        for i, n_hidden in enumerate(self.n_hidden):
            if i == 0:
                input = x
                input_dim = self.n_in
            elif i > 0:
                input = output
                input_dim = self.n_hidden[i-1]

            with tf.name_scope("hidden" + str(i+1)):
                weights = tf.Variable(tf.truncated_normal([input_dim, self.n_hidden[i]], stddev = 0.01), name = "weights")
                biases = tf.Variable(tf.zeros([self.n_hidden[i]]), name = "biases")
                out = tf.nn.relu(tf.matmul(input, weights) + biases)
                output = tf.nn.dropout(out, keep_prob, name = "hidden{}".format(i+1) + "_output")
        with tf.name_scope("Output"):
            weights = tf.Variable(tf.truncated_normal([self.n_hidden[-1],self.n_out], stddev = 0.01), name = "weights")
            biases = tf.Variable(tf.zeros([self.n_out]), name = "biases")
            y = tf.nn.softmax(tf.matmul(output, weights) + biases, name = "output")
        return y

    def loss(self, y, t):
        cross_entropy = -tf.reduce_sum(t * tf.log(tf.clip_by_value(y,1e-10,1.0)))
        return cross_entropy

    def training(self,loss):
        optimizer = tf.train.AdamOptimizer(0.001)
        train_step = optimizer.minimize(loss)
        return train_step

    def accuracy(self, y, t):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def fit(self, X_train, Y_train, epoch, p_keep=1.0):
        #model set
        x = tf.placeholder(tf.float32, shape=[None, self.n_in])
        t = tf.placeholder(tf.float32, shape=[None, self.n_out])
        keep_prob = tf.placeholder(tf.float32)
        y = self.inference(x, keep_prob)
        loss = self.loss(y, t)
        train_step = self.training(loss)
        accuracy = self.accuracy(y, t)

        # save variables
        self._x = x
        self._t = t
        self._y = y
        self._keep_prob = keep_prob

        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)
        self._sess = sess

        #training
        for i in range(epoch):
            self._sess.run(train_step, feed_dict={
            x: X_train,
            t: Y_train,
            keep_prob: p_keep})
            if (i+1) % 100 ==0:
                y_ = y.eval(session = sess, feed_dict={
                    x: X_train,
                    keep_prob: p_keep})
                loss_ = loss.eval(session=sess, feed_dict={
                    x: X_train,
                    t: Y_train,
                    keep_prob: 1.0})
                accuracy_ = accuracy.eval(session = sess, feed_dict={
                    x: X_train,
                    t: Y_train,
                    keep_prob: 1.0})
                print("epoch: "+ str(i+1),"loss: " + str(loss_),"accuracy: "+ str(accuracy_), y_)

    def evaluate(self, X_test, Y_test):
        accuracy = self.accuracy(self._y, self._t)
        return accuracy.eval(session = self._sess, feed_dict={
            self._x: X_test,
            self._t: Y_test,
            self._keep_prob: 1.0})

    def predict(self, samples):
        predictions = tf.argmax(self._y, 1)
        return self._sess.run(predictions, {self._x: samples,self._keep_prob: 1.0})
