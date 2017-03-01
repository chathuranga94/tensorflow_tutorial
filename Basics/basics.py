import tensorflow as tf

'''
node1 = tf.constant(3.0, tf.float32)
print(node1)   # Not used yet
'''

a = tf.placeholder(tf.float32)  
b = tf.placeholder(tf.float32)
c = tf.add(a, b)
d = tf.multiply(c,3)

sess = tf.Session()
print(sess.run(d , {a:[2.5,1.5], b:[2,5]}))