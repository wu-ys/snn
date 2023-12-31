import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import argparse

result_path = Path("/home/wuys/wys/snn/results")
plot_path = Path("/home/wuys/wys/snn/plots")

# parser = argparse.ArgumentParser()
# parser.add_argument("-s", "--surrogate", type=str, help="Surrogate function type")
# parser.add_argument("-a", "--alpha", type=float, help="Alpha value")

# args = parser.parse_args()

# specify the name of surrogate function
surrogate_name = "soft_sign"

path = Path(result_path / surrogate_name)

# specify the list of alpha values
if surrogate_name == "power":
  alpha_list = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9] # for power
  x_range = [0,1]
elif surrogate_name == "piecewise_quad":
  alpha_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0] # for piecewise_quad
  x_range = [0,2]
elif surrogate_name == "trigono":
  alpha_list = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] # for trigono
  x_range = [0,10]
elif surrogate_name == "sigmoid":
  alpha_list = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] # for sigmoid
  x_range = [0,10]
else:
  alpha_list = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0] # for others
  x_range = [0,5]

test_acc_list = []
train_acc_list = []

for alpha in alpha_list:
  model_name = "snn_surro_" + surrogate_name + "_alpha" + str(alpha)
  d = np.load(path / (model_name + ".npz"))

  train_acc_list.append(d['train_acc'][-1])
  test_acc_list.append(d["test_acc"][-1])

df = pd.DataFrame({'alpha': alpha_list, 'train_acc': train_acc_list, 'test_acc': test_acc_list})

plt.plot(alpha_list, train_acc_list, color='tab:blue', marker='.', label="train accuracy")
plt.plot(alpha_list, test_acc_list, color='tab:red', marker='.', label = "test accuracy")

plt.yticks(np.arange(0.9, 1.01, 0.01))
plt.ylim([0.9, 1.0])
plt.xlim(x_range)

plt.xlabel("alpha")
plt.ylabel("accuracy")
plt.legend(loc="lower right")

plt.savefig(plot_path / "acc_alpha" / "{}.jpg".format(surrogate_name))
df.to_csv(plot_path / "acc_alpha" / "{}.csv".format(surrogate_name))


# alpha = args.alpha
# 
# model_name = "snn_surro_" + surrogate_name + "_alpha" + str(alpha)
# 
# datafile_name = model_name + ".npz"
# d = np.load(result_path / surrogate_name / datafile_name)


# plt.plot(d['train_acc'], color='tab:blue', label="train accuracy")
# plt.plot(d['test_acc'], color='tab:red', label="test accuracy")
# 
# plt.xticks(np.arange(0,21,2))
# plt.yticks(np.arange(0.9,1,0.01))
# plt.xlim([-1,21])
# plt.ylim([0.9,1])
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Model: {} with alpha = {}".format(surrogate_name, alpha))
# plt.legend(loc="lower right")
# 
# plt.annotate('Final train acc:\n{:4f}'.format(d['train_acc'][-1]), xy=(19, d['train_acc'][-1]), xycoords='data',
#             xytext=(-40, -25), textcoords='offset points')
# plt.annotate('Final test acc:\n{:4f}'.format(d['test_acc'][-1]), xy=(19, d['test_acc'][-1]), xycoords='data',
#             xytext=(-40, -25), textcoords='offset points')
# 
# plt.savefig()