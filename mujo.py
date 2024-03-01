import pickle

filename = "/home/basel/PycharmProjects/pythonProject/checkpoints/mountaincarcontinuous/policies/default_policy/policy_state.pkl"

with open(filename, 'rb') as file:
   data = pickle.load(file)


print(data)