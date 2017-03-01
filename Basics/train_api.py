import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(100):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
  # Loss function through iterations
  print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# Linear Regression through ML
print(sess.run([W, b]))