#%%
#import libraries
import tensorflow as tf
#%%
def run():
    output = None
    logit_data = [0.2, 0.1, 0.01]
    logits = tf.placeholder(tf.float32)

    # TODO: Calculate the softmax of the logits
    softmax = tf.nn.softmax(logits)

    with tf.Session() as sess:
        # TODO: Feed in the logit data
        output = sess.run(softmax,  feed_dict={logits: logit_data}  )
    return output

#%%
run()
