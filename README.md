# audiocnn
Normal training data files are located in /new data/pettijohncnormal.
Test data files are located in /new data/pettijohncnormal.
Pretrained model is located in /original folder.

runtest.py is a sample test script. To train on mic0 data run:
finetuneAnn.test('/home/labuser/AudioCNN/new data/pettijohncnormal/mic0/combined/', outname='trainscores.npy', k=6, dim = 64, model='audiocnn')
finetuneAnn.test('/home/labuser/AudioCNN/new data/annotated/mic0/combined/', outname='testscores.npy', k=6, dim = 64, model='audiocnn')
k is the batch size. model is the cnnmodel. dim is is the dimension of output.
If Model is 'audiocnn', dim can be either 4 or 64
If Model is 'resnet', dim is 1000
These commands will generate two files with extracted features from training/ testing files. To perform evaluation run:
[acc1, f1, acc2, auc] = test.test('testscores.npy','trainscores.npy','plot0.png')
This commapnd will also generate a TSNE plot in plot0.png file.

For example, to extarct resnet features and perform testing run:
finetuneAnn.test('/home/labuser/AudioCNN/new data/pettijohncnormal/mic0/combined/', outname='trainscores.npy', k=6, dim = 1000, model='resnet')
finetuneAnn.test('/home/labuser/AudioCNN/new data/annotated/mic0/combined/', outname='testscores.npy', k=6, dim = 1000, model='resnet')
[acc1, f1, acc2, auc] = test.test('testscores.npy','trainscores.npy','plot0.png')
