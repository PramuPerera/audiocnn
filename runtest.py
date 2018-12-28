import finetuneAnn
import test
results = [];
for mic in [0,2,3,4]:
	finetuneAnn.test('/home/labuser/AudioCNN/new data/pettijohncnormal/mic'+str(mic)+'/combined/', outname='trainscores.npy', k=6, dim = 64, model='audiocnn')
	finetuneAnn.test('/home/labuser/AudioCNN/new data/annotated/mic'+str(mic)+'/combined/', outname='testscores.npy', k=6, dim = 64, model='audiocnn')
	[acc1, f1, acc2, auc] = test.test('testscores.npy','trainscores.npy','plot'+str(mic)+'.png')
	print("Balance acc : " + str(acc1) + ", Acc : " +str(acc2) + " F1: " + str(f1) + " AUC : " + str(auc))
	results.append([acc1, f1, acc2, auc])
print(results)

