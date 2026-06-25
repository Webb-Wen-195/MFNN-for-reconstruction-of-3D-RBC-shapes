import matplotlib.pyplot as plt
import numpy as np

pred_data = np.load("./mfS_predict.npz")
rbc_pred = pred_data["predict_rbc_hf"]


plt.figure(figsize=(8, 8))
for pred in rbc_pred:
    plt.scatter(pred[0], pred[1], c="r", s=10)
plt.axis("equal")
plt.title("y-x")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("./x-y.png", dpi=400)
plt.close()

plt.figure(figsize=(8, 8))
for pred in rbc_pred:
    plt.scatter(pred[0], pred[2], c="r", s=10)
plt.axis("equal")
plt.title("z-x")
plt.xlabel("x")
plt.ylabel("z")
plt.savefig("./x-z.png", dpi=400)
plt.close()

plt.figure(figsize=(8, 8))
for pred in rbc_pred:
    plt.scatter(pred[1], pred[2], c="r", s=10)
plt.axis("equal")
plt.title("z-y")
plt.xlabel("y")
plt.ylabel("z")
plt.savefig("./y-z.png", dpi=400)
plt.close()