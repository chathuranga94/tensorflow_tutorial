import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# Linear Model for input x
print(sess.run(linear_model, {x:[1,2,3,4]}))
# Loss Function : Predicted output y from input x -vs- Actual output y
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# Manually adjusting weights : fixW = tf.assign(W, [-1.]) => sess.run([fixW])