import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

result_path = Path("/home/wuys/wys/snn/results")
plot_path = Path("/home/wuys/wys/snn/plots")
surrogate_name = 'soft_sign'
alpha = 2.0

model_name = "snn_surro_" + surrogate_name + "_alpha" + str(alpha)

datafile_name = model_name + ".npz"
d = np.load(result_path / surrogate_name / datafile_name)

plt.plot(d['train_acc'], color='tab:blue', label="train accuracy")
plt.plot(d['test_acc'], color='tab:red', label="test accuracy")

plt.xticks(np.arange(0,21,2))
plt.yticks(np.arange(0.9,1,0.01))
plt.xlim([-1,21])
plt.ylim([0.9,1])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model: {} with alpha = {}".format(surrogate_name, alpha))
plt.legend(loc="lower right")

plt.annotate('Final train acc:\n{:4f}'.format(d['train_acc'][-1]), xy=(19, d['train_acc'][-1]), xycoords='data',
            xytext=(-40, -25), textcoords='offset points')
plt.annotate('Final test acc:\n{:4f}'.format(d['test_acc'][-1]), xy=(19, d['test_acc'][-1]), xycoords='data',
            xytext=(-40, -25), textcoords='offset points')

plt.savefig(plot_path / "{}.jpg".format(model_name))