import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

pred_data = np.load('./mfS_predict.npz')
ref_data = np.load('../data/D_A1.05_hf.npz')

rbc_ref = ref_data['rbc']

rbc_pred = pred_data['predict_rbc_hf']


plt.figure(figsize=(8,8))
for ref, pred in zip(rbc_ref, rbc_pred):
    plt.scatter(ref[1],ref[2], c ='b', s = 10)
    plt.scatter(pred[0],pred[1], c ='r', s = 10)
plt.axis('equal')
plt.title('y-x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(handles=[mpatches.Patch(color='b', label='Ref'), mpatches.Patch(color='r', label='Pred')])
plt.savefig('./x-y.png',dpi=400)
plt.close()

plt.figure(figsize=(8,8))
for ref, pred in zip(rbc_ref, rbc_pred):
    plt.scatter(ref[1],ref[3], c ='b', s = 10)
    plt.scatter(pred[0],pred[2], c ='r', s = 10)
plt.axis('equal')
plt.title('z-x')
plt.xlabel('x')
plt.ylabel('z')
plt.legend(handles=[mpatches.Patch(color='b', label='Ref'), mpatches.Patch(color='r', label='Pred')])
plt.savefig('./x-z.png',dpi=400)
plt.close()

plt.figure(figsize=(8,8))
for ref, pred in zip(rbc_ref, rbc_pred):
    plt.scatter(ref[2],ref[3], c ='b', s = 10)
    plt.scatter(pred[1],pred[2], c ='r', s = 10)
plt.axis('equal')
plt.title('z-y')
plt.xlabel('y')
plt.ylabel('z')
plt.legend(handles=[mpatches.Patch(color='b', label='Ref'), mpatches.Patch(color='r', label='Pred')])
plt.savefig('./y-z.png',dpi=400)
plt.close()