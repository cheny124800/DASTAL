import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()



    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1).expand(C, A)
        )

        weight_CV[weight_CV != weight_CV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.Ave - ave_CxA).pow(2))

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                           .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_CV) + ave_CxA.mul(weight_CV)).detach()

        self.Amount += onehot.sum(0)




class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num, beta =0.5):
        super(ISDALoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num)

        self.class_num = class_num

        self.cross_entropy = nn.CrossEntropyLoss()
        self.NoReducecross_entropy = nn.CrossEntropyLoss(reduction="none")
        self.beta = beta




    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]

        NxW_ij = weight_m.expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = cv_matrix[labels]

        # sigma2 = ratio * \
        #          torch.bmm(torch.bmm(NxW_ij - NxW_kj,
        #                              CV_temp).view(N * C, 1, A),
        #                    (NxW_ij - NxW_kj).view(N * C, A, 1)).view(N, C)

        # sigma2 = ratio * \
        #          torch.bmm(torch.bmm(NxW_ij - NxW_kj,
        #                              CV_temp),
        #                    (NxW_ij - NxW_kj).permute(0, 2, 1))
        # sigma2 = sigma2.mul(torch.eye(C).cuda()
        #                     .expand(N, C, C)).sum(2).view(N, C)

        sigma2 = ratio * (weight_m - NxW_kj).pow(2).mul(
            CV_temp.view(N, 1, A).expand(N, C, A)
        ).sum(2)

        aug_result = y + 0.5 * sigma2

        #hard_result = -y + 0.5 * sigma2

        return aug_result#,hard_result


    def forward(self, model, x, target_x, ratio,Val_Flag = False):

        y, features = model(x, isda=True)
        # y = fc(features)
        if Val_Flag == False:
            self.estimator.update_CV(features.detach(), target_x)

            isda_aug_y = self.isda_aug(model.fc, features, y, target_x, self.estimator.CoVariance.detach(), ratio)

            loss = self.cross_entropy(isda_aug_y, target_x)
            #if Aug_Flag == False:
            #loss = self.cross_entropy(y, target_x)
            #    return loss1,y
            return loss, y
        else:
            pro = F.softmax(y,dim=1)
            #print(torch.max(pro, 1))
            Pre_Target = torch.max(pro, 1).indices     
            self.estimator.update_CV(features.detach(), Pre_Target)

            isda_aug_y = self.isda_aug(model.fc, features, y, Pre_Target, self.estimator.CoVariance.detach(), ratio)

            emc = self.NoReducecross_entropy(isda_aug_y, Pre_Target)
            emc = torch.exp(emc) -1
            #print(emc)
            return emc     

    def forward1(self, model, x, target_x, ratio):
        y, features = model(x, isda=True)

        # y = fc(features)

        self.estimator.update_CV(features.detach(), target_x)
        isda_aug_y,isda_hard_y = self.isda_aug(model.fc, features, y, target_x, self.estimator.CoVariance.detach(), ratio) #model.module.fc 
        item_total =0 
        for i in range(len(y)):
            tmp =  self.cross_entropy(isda_aug_y[i].unsqueeze(0), target_x[i].unsqueeze(0))
            tmp1 =  self.cross_entropy(isda_hard_y[i].unsqueeze(0), target_x[i].unsqueeze(0))            
            #print(tmp)
            #item_total = item_total + tmp + torch.log(1+ self.beta * math.tan(self.beta)*torch.exp(tmp))
            if 1:
                if tmp1 <self.the: #self.the:
                    #item_total = item_total + tmp + torch.log(1+ math.tan(self.beta)*torch.sin(self.beta * torch.exp(tmp)) + math.tan(self.beta)*(1 + math.sin(self.beta)))
                    item_total = item_total + tmp + torch.log(1+ math.tan(self.beta)*torch.sin(self.beta * (torch.exp(tmp1)-1)))
                    #item_total = item_total + tmp + torch.log(1+ math.tan(self.beta)*torch.exp(tmp))
                    #print("!!!!!!!!!")
                    #print(self.beta * self.beta*math.tan(self.beta)*torch.sin(tmp))
                    #before item_total = item_total + tmp + torch.log(1+ math.tan(self.beta)*torch.sin(self.beta * torch.exp(tmp)))
                else:
                    item_total = item_total + tmp
            
        loss =  item_total / len(y) 
        #loss = self.cross_entropy(isda_aug_y, target_x)

        return loss, y