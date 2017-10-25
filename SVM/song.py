import numpy
from libsvm.python.svmutil import *



y, x = svm_read_problem('train_song.txt')#读入训练数据
yt, xt = svm_read_problem('test_song.txt')#训练测试数据
m = svm_train(y, x )#训练
p_label, p_acc, p_val = svm_predict(yt,xt,m)#测试


file_object = open('sampleSubmission1.csv', 'w')
file_object.write("id,prediction\n" )
for item in range(len(p_label)):
    file_object.write("%s," % item)
    file_object.write("%s\n" % int(p_label[item]) )
file_object.close()


