# Solution is available in the other "solution.py" tab
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# cross entropy
calc_crossentropy = - tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))

with tf.Session() as sess:
    # TODO: Feed in the logit data
    print(sess.run(calc_crossentropy, feed_dict={one_hot: one_hot_data, softmax : softmax_data}))
    #print(cross_hot)