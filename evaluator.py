import torch
import os
import torch.nn as nn
import torchvision.models as models

'''===============================================================
1. Title:     

DLP summer 2023 Lab6 classifier

2. Purpose:

For computing the classification accruacy.

3. Details:

The model is based on ResNet18 with only chaning the
last linear layer. The model is trained on iclevr dataset
with 1 to 5 objects and the resolution is the upsampled 
64x64 images from 32x32 images.

It will capture the top k highest accuracy indexes on generated
images and compare them with ground truth labels.

4. How to use

You may need to modify the checkpoint's path at line 40.
You should call eval(images, labels) and to get total accuracy.
images shape: (batch_size, 3, 64, 64)
labels shape: (batch_size, 24) where labels are one-hot vectors
e.g. [[1,1,0,...,0],[0,1,1,0,...],...]
Images should be normalized with:
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

==============================================================='''


class evaluation_model():
    def __init__(self):
        #modify the path to your own path
        checkpoint = torch.load('./checkpoint.pth')
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512,24),
            nn.Sigmoid()
        )
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.cuda()
        self.resnet18.eval()
        self.classnum = 24
    def compute_acc(self, out, onehot_labels, txt_path):
        batch_size = out.size(0)
        acc = 0
        total = 0
        for i in range(batch_size):
            k = int(onehot_labels[i].sum().item())
            total += k
            # choose top k value in ith output
            # outv, outi is value, index respectively
            outv, outi = out[i].topk(k) 
            lv, li = onehot_labels[i].topk(k)
            item_acc = 0
            for j in outi: # if classifier detect that obj j is in img, then j = 1, else j = 0
                if j in li:
                    acc += 1
                    item_acc += 1
            print("===",i,"===")
            # print(outi)
            # print(li)
            print("Object " + str(i) + " Accuracy : " , item_acc/k )
            if not os.path.exists(txt_path):
                with open(txt_path, "w") as file:
                    pass
            with open(txt_path, 'a') as test_record:
                test_record.write(('Task {} : {}\n'.format(i,item_acc/k)))
        return acc / total
    def eval(self, images, labels, txt_path):
        with torch.no_grad():
            #your image shape should be (batch, 3, 64, 64)
            out = self.resnet18(images) # what is output shape?
            acc = self.compute_acc(out.cpu(), labels.cpu(), txt_path)
            return acc