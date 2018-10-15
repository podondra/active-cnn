from active_cnn import model
import sys
import h5py


# get iteration number
iteration = int(sys.argv[1])

# load data matrixes
hdf5 = h5py.File('data/data.hdf5', 'r+')
group = hdf5['iteration_{:02}'.format(iteration)]
X_train = group['X_train'][...]
X = group['X'][...]
y_train = group['y_train'][...]

# train network (run it outside the notebook)
cnn = model.CNN()
cnn.train(X_train, y_train)

# save network's weights
cnn.save('data/cnn-{:02}.hdf5'.format(iteration))

# get and save predictions
y_pred = cnn.predict(X)
pred_dt = group.create_dataset('y_pred', y_pred.shape, y_pred.dtype)
pred_dt[...] = y_pred
