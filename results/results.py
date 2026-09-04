import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def update(angle):
    ax.view_init(elev=30, azim=angle)

data = [
{'noise': 0.0, 'mae_metallicity': 0.008970556520121367, 'mae_cloud': 0.07484983596686928, 'mae_temperature': 0.2749050429737867, 'rmse_metallicity': 0.012087284716660323, 'rmse_cloud': 0.10446146733567695, 'rmse_temperature': 0.39705669269145255, 'r2_metallicity': 0.9883543163341152, 'r2_cloud': 0.9854677512594077, 'r2_temperature': 0.9263407374182422},
{'noise': 0.01, 'mae_metallicity': 0.006963465938303652, 'mae_cloud': 0.1430970474871089, 'mae_temperature': 0.3160493272350128, 'rmse_metallicity': 0.010241303351834955, 'rmse_cloud': 0.1742948126510419, 'rmse_temperature': 0.44898060883562396, 'r2_metallicity': 0.991639775388953, 'r2_cloud': 0.9595433887164979, 'r2_temperature': 0.9058159212317547},
{'noise': 0.02, 'mae_metallicity': 0.012941097361944803, 'mae_cloud': 0.12685718731532258, 'mae_temperature': 0.30306421434378594, 'rmse_metallicity': 0.015380892225096195, 'rmse_cloud': 0.14647451447714593, 'rmse_temperature': 0.418158467482302, 'r2_metallicity': 0.9811430893579929, 'r2_cloud': 0.9714277321208873, 'r2_temperature': 0.9183033749171527},
{'noise': 0.03, 'mae_metallicity': 0.008230382411676757, 'mae_cloud': 0.12761027060128827, 'mae_temperature': 0.3088996455697537, 'rmse_metallicity': 0.010365994251225717, 'rmse_cloud': 0.16515986960663806, 'rmse_temperature': 0.440300410942596, 'r2_metallicity': 0.9914349596694118, 'r2_cloud': 0.9636729909905576, 'r2_temperature': 0.9094224630705074},
{'noise': 0.04, 'mae_metallicity': 0.007576937290709931, 'mae_cloud': 0.10712216579878403, 'mae_temperature': 0.29290641511955745, 'rmse_metallicity': 0.009864163236844805, 'rmse_cloud': 0.137973392480474, 'rmse_temperature': 0.4061782070205993, 'r2_metallicity': 0.9922441752702678, 'r2_cloud': 0.9746480561310362, 'r2_temperature': 0.9229175408371584},
{'noise': 0.05, 'mae_metallicity': 0.009196915429444459, 'mae_cloud': 0.16427245020064776, 'mae_temperature': 0.2814740274868187, 'rmse_metallicity': 0.012143587321957472, 'rmse_cloud': 0.1850939762712471, 'rmse_temperature': 0.3937821072466685, 'r2_metallicity': 0.9882455724050513, 'r2_cloud': 0.9543747613169274, 'r2_temperature': 0.9275506851804641},
{'noise': 0.06, 'mae_metallicity': 0.009202163295579545, 'mae_cloud': 0.10994346207046488, 'mae_temperature': 0.2835791977965908, 'rmse_metallicity': 0.012530023839896579, 'rmse_cloud': 0.13340039727994474, 'rmse_temperature': 0.39442488735559694, 'r2_metallicity': 0.9874855640591607, 'r2_cloud': 0.9763007378538079, 'r2_temperature': 0.9273139705839817},
{'noise': 0.07, 'mae_metallicity': 0.009385856312276923, 'mae_cloud': 0.08649653537521042, 'mae_temperature': 0.3037219903860424, 'rmse_metallicity': 0.011870443559084605, 'rmse_cloud': 0.11882161795360331, 'rmse_temperature': 0.4242837596814834, 'r2_metallicity': 0.9887684064239605, 'r2_cloud': 0.9811976772869618, 'r2_temperature': 0.9158924190326609},
{'noise': 0.08, 'mae_metallicity': 0.010995915153224818, 'mae_cloud': 0.1046715583890953, 'mae_temperature': 0.2859046134253235, 'rmse_metallicity': 0.013965353098580956, 'rmse_cloud': 0.13949763261600767, 'rmse_temperature': 0.4031879660576314, 'r2_metallicity': 0.984454262622876, 'r2_cloud': 0.9740848185681085, 'r2_temperature': 0.9240483089708661},
{'noise': 0.09, 'mae_metallicity': 0.012871344257096186, 'mae_cloud': 0.1677982414955062, 'mae_temperature': 0.3177309995773888, 'rmse_metallicity': 0.015893811568814597, 'rmse_cloud': 0.21104209756484524, 'rmse_temperature': 0.42904262013514144, 'r2_metallicity': 0.9798644449959362, 'r2_cloud': 0.9406857875497955, 'r2_temperature': 0.9139950995478616},
{'noise': 0.1, 'mae_metallicity': 0.00980621003567257, 'mae_cloud': 0.13510413302777072, 'mae_temperature': 0.32228457545114764, 'rmse_metallicity': 0.012657175008136835, 'rmse_cloud': 0.17089594702915514, 'rmse_temperature': 0.43687867782259393, 'r2_metallicity': 0.9872302893967205, 'r2_cloud': 0.9611058660837617, 'r2_temperature': 0.9108248140705867}
]

noise = [d['noise'] for d in data]

mae_met = [d['mae_metallicity'] for d in data]
mae_cloud = [d['mae_cloud'] for d in data]
mae_temp = [d['mae_temperature'] for d in data]

r2_met = [d['r2_metallicity'] for d in data]
r2_cloud = [d['r2_cloud'] for d in data]
r2_temp = [d['r2_temperature'] for d in data]
plt.figure()
plt.plot(noise, mae_met, label="Metallicity")
plt.plot(noise, mae_cloud, label="Cloud")
plt.plot(noise, mae_temp, label="Temperature")
plt.xlabel("Noise Level")
plt.ylabel("MAE")
plt.title("Noise vs MAE")
plt.legend()
plt.savefig("mae_result.png")
plt.show()
plt.figure()
plt.plot(noise, r2_met, label="Metallicity")
plt.plot(noise, r2_cloud, label="Cloud")
plt.plot(noise, r2_temp, label="Temperature")
plt.xlabel("Noise Level")
plt.ylabel("R² Score")
plt.title("Noise vs R²")
plt.savefig("r2_result.png")
plt.legend()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("results/output.csv")

# pivot into grid
pivot = df.pivot(index="gen", columns="epoch", values="val_loss")

X = pivot.columns.values   # epochs
Y = pivot.index.values     # noise/gen
X, Y = np.meshgrid(X, Y)
Z = pivot.values
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df["epoch"], df["gen"], df["val_loss"])

ax.set_xlabel("Epoch")
ax.set_ylabel("Noise")
ax.set_zlabel("Val Loss")
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
plt.savefig("val_loss_result")
plt.show()

