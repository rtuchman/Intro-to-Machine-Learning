import backprop_dataimport skeleton_backprop_networktraining_data, test_data = backprop_data.load(train_size=10000, test_size=5000)net = skeleton_backprop_network.Network([784, 40, 10])net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)#training_data, test_data = backprop_data.load(train_size=50000,test_size=10000)#net = skeleton_backprop_network.Network([784, 40, 10])#net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)