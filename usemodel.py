import pickle  # Pickle is used to load the stored state of a python object.

# Pass new value to predict from the loaded model.
exp = float(input('Enter years of experience: '))

filename = 'finalized_model.sav'

# Load the model from the disk
loaded_model = pickle.load(open(filename, 'rb'))

result = loaded_model.predict([[exp]])

print('The new predicted salary is', result[0][0])
