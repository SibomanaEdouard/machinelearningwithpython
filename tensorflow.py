import tensorflow as tf

# this is to initialize
x1=tf.constant([1,2,3,4])
x2=tf.constant([5,6,7,8])

# this is to multiply
product=tf.multiply(x1,x2)

# this is to initialize the session
sess=tf.session()
# let me print the result according to the session started 
print(sess(product))
# let me close the session
sess.close()


