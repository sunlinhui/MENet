import torch.nn as nn
import torch.nn.functional as F
from model import menet_util
import torch


class MENet_single (nn.Module):
    def __init__(self, num_class, normal_channel=False):
        super(MENet_single, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = menet_util.PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                                           [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = menet_util.PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                                           [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)
        self.sa3 = menet_util.PointNetSetAbstraction_nonmax(None, None, None, 640 + 4, [256, 512, 1024],
                                                                True)
        self.fc1 = nn.Linear(1280, 512)

        self.sa_append1 = menet_util.PointNetSetAbstractionMsg(32, [0.4], [32], 1024, [[512, 256, 64]])

        self.memory_mlp1 = menet_util.mlp_nomax(None, None, None, 64, [128, 256], True)

        self.memory_shape = [16, 64]
        self.memory_w = nn.init.normal_(torch.empty(self.memory_shape), mean=0.0, std=1.0)
        self.memory_w = nn.Parameter(self.memory_w, requires_grad=True)


    def forward(self, xyz):

        B, _, _ = xyz.shape

        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        _, memory_feature_1 = self.sa_append1(l3_xyz, l3_points)
        B1, C1, N1 = memory_feature_1.shape
        query_norm_1 = F.normalize(memory_feature_1, dim=1).transpose(dim0=1, dim1=2)  # B*C*N  (B*N*C)
        memory_norm = F.normalize(self.memory_w, dim=1)
        query_norm_1 = torch.reshape(query_norm_1, (-1, C1))
        s_1 = torch.mm(query_norm_1, memory_norm.transpose(dim0=0, dim1=1))  # BN*16
        addressing_vec_1 = F.softmax(s_1, dim=1)
        memory_feature_1 = torch.mm(addressing_vec_1, self.memory_w)
        memory_feature_1 = torch.reshape(memory_feature_1, (-1, N1, C1))
        memory_feature_1 = self.memory_mlp1(memory_feature_1.transpose(dim0=1, dim1=2))
        x = torch.cat([torch.max(memory_feature_1, 2)[0], torch.max(l3_points, 2)[0]], dim=1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x

class MENet (nn.Module):
    def __init__(self, num_class, normal_channel=False):
        super(MENet, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = menet_util.PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                                           [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = menet_util.PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                                           [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = menet_util.PointNetSetAbstraction_nonmax(None, None, None, 640 + 4, [256, 512, 1024], True)

        self.fc1 = nn.Linear(1280, 512) ###
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256) #####
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

        self.x2_sa=menet_util.PointNetSetAbstraction_nonmax(None, None, None, 4, [64,256, 512, 1024], True)

        self.sa_append1 = menet_util.PointNetSetAbstractionMsg(32, [0.4], [32], 1024, [[512, 256,64]])
        self.sa_append2 = menet_util.PointNetSetAbstractionMsg(32, [0.4], [32], 1024, [[512, 256,64]])
        self.memory_mlp1 = menet_util.mlp_nomax(None, None, None, 64, [128, 256], True)
        self.memory_mlp2 = menet_util.mlp_nomax(None, None, None, 64, [256, 512], True)

        self.memory_shape = [16, 64]
        self.memory_w = nn.init.normal_(torch.empty(self.memory_shape), mean=0.0, std=1.0)
        self.memory_w = nn.Parameter(self.memory_w, requires_grad=True)


        self.fc1_2 = nn.Linear(1536, 512)
        self.bn1_2 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2_2 = nn.Linear(512, 256)  #####
        self.bn2_2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3_2 = nn.Linear(256, num_class)


    def forward(self, xyz,xyz2):
        B, _, _ = xyz.shape

        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x2_xyz, x2_points = self.x2_sa(xyz2, None)
        _, memory_feature_1 = self.sa_append1(l3_xyz, l3_points)
        B1, C1, N1 = memory_feature_1.shape
        query_norm_1 = F.normalize(memory_feature_1, dim=1).transpose(dim0=1, dim1=2)  # B*C*N  (B*N*C)
        memory_norm = F.normalize(self.memory_w, dim=1)
        query_norm_1 = torch.reshape(query_norm_1, (-1, C1))
        s_1 = torch.mm(query_norm_1, memory_norm.transpose(dim0=0, dim1=1))  # BN*16
        addressing_vec_1 = F.softmax(s_1, dim=1)
        memory_feature_1 = torch.mm(addressing_vec_1, self.memory_w)
        memory_feature_1 = torch.reshape(memory_feature_1, (-1, N1, C1))
        memory_feature_1 = self.memory_mlp1(memory_feature_1.transpose(dim0=1, dim1=2))

        x = torch.cat([torch.max(memory_feature_1, 2)[0], torch.max(l3_points, 2)[0]], dim=1)

        _, memory_feature_2 = self.sa_append2(x2_xyz, x2_points)
        B2, C2, N2 = memory_feature_2.shape
        query_norm_2 = F.normalize(memory_feature_2, dim=1).transpose(dim0=1, dim1=2)
        query_norm_2 = torch.reshape(query_norm_2, (-1, C2))
        s_2 = torch.mm(query_norm_2, memory_norm.transpose(dim0=0, dim1=1).detach())
        addressing_vec_2 = F.softmax(s_2, dim=1)
        memory_feature_2 = torch.mm(addressing_vec_2, self.memory_w.detach())
        memory_feature_2 = torch.reshape(memory_feature_2, (-1, N2, C2))
        memory_feature_2 = self.memory_mlp2(memory_feature_2.transpose(dim0=1, dim1=2))

        x2 = torch.cat(
            [torch.max(memory_feature_2, 2)[0], torch.max(l3_points.detach(), 2)[0] + torch.max(x2_points, 2)[0]],
            dim=1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        x2 = self.drop1(F.relu(self.bn1_2(self.fc1_2(x2))))
        x2 = self.drop2(F.relu(self.bn2_2(self.fc2_2(x2))))
        x2 = self.fc3_2(x2)

        x = F.log_softmax(x, -1)
        x2 = F.log_softmax(x2, -1)
        return x,x2


class MENet_test (nn.Module):
    def __init__(self, num_class, normal_channel=False):
        super(MENet_test, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = menet_util.PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                                           [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = menet_util.PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                                           [[64, 64, 128], [128, 128, 256], [128, 128, 256]])


        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)
        self.sa3 = menet_util.PointNetSetAbstraction_nonmax(None, None, None, 640 + 4, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1280, 512)
        self.x2_sa = menet_util.PointNetSetAbstraction_nonmax(None, None, None, 4, [64, 256, 512, 1024], True)
        self.sa_append1 = menet_util.PointNetSetAbstractionMsg(32, [0.4], [32], 1024, [[512, 256, 64]])
        self.sa_append2 = menet_util.PointNetSetAbstractionMsg(32, [0.4], [32], 1024, [[512, 256, 64]])
        self.memory_mlp1 = menet_util.mlp_nomax(None, None, None, 64, [128, 256], True)
        self.memory_mlp2 = menet_util.mlp_nomax(None, None, None, 64, [256, 512], True)

        self.fc1_2 = nn.Linear(1536, 512)
        self.bn1_2 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2_2 = nn.Linear(512, 256)
        self.bn2_2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3_2 = nn.Linear(256, num_class)

        self.memory_shape = [16, 64]
        self.memory_w = nn.init.normal_(torch.empty(self.memory_shape), mean=0.0, std=1.0)
        self.memory_w = nn.Parameter(self.memory_w, requires_grad=True)

    def forward(self, xyz,xyz2,old_feature,flag):
        if flag==0:
            B, _, _ = xyz.shape

            if self.normal_channel:
                norm = xyz[:, 3:, :]
                xyz = xyz[:, :3, :]
            else:
                norm = None
            l1_xyz, l1_points = self.sa1(xyz, norm)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
            _, memory_feature_1 = self.sa_append1(l3_xyz, l3_points)
            B1, C1, N1 = memory_feature_1.shape
            query_norm_1 = F.normalize(memory_feature_1, dim=1).transpose(dim0=1, dim1=2)  # B*C*N ZHUAN (B*N*C)
            memory_norm = F.normalize(self.memory_w, dim=1)
            query_norm_1 = torch.reshape(query_norm_1, (-1, C1))
            s_1 = torch.mm(query_norm_1, memory_norm.transpose(dim0=0, dim1=1))  # BN*16
            addressing_vec_1 = F.softmax(s_1, dim=1)
            memory_feature_1 = torch.mm(addressing_vec_1, self.memory_w)
            memory_feature_1 = torch.reshape(memory_feature_1, (-1, N1, C1))
            memory_feature_1 = self.memory_mlp1(memory_feature_1.transpose(dim0=1, dim1=2))

            x = torch.cat([torch.max(memory_feature_1, 2)[0], torch.max(l3_points, 2)[0]], dim=1)
            x = self.drop1(F.relu(self.bn1(self.fc1(x))))
            x = self.drop2(F.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)
            x = F.log_softmax(x, -1)
            return x, l3_points
        else:


            x2_xyz, x2_points = self.x2_sa(xyz2, None)
            _, memory_feature_2 = self.sa_append2(x2_xyz, x2_points)
            B2, C2, N2 = memory_feature_2.shape
            query_norm_2 = F.normalize(memory_feature_2, dim=1).transpose(dim0=1, dim1=2)
            query_norm_2 = torch.reshape(query_norm_2, (-1, C2))
            memory_norm = F.normalize(self.memory_w, dim=1)
            s_2 = torch.mm(query_norm_2, memory_norm.transpose(dim0=0, dim1=1).detach())
            addressing_vec_2 = F.softmax(s_2, dim=1)
            memory_feature_2 = torch.mm(addressing_vec_2, self.memory_w.detach())
            memory_feature_2 = torch.reshape(memory_feature_2, (-1, N2, C2))
            memory_feature_2 = self.memory_mlp2(memory_feature_2.transpose(dim0=1, dim1=2))

            x2 = torch.cat(
                [torch.max(memory_feature_2, 2)[0],
                 torch.max(old_feature.detach(), 2)[0] + torch.max(x2_points, 2)[0]],
                dim=1)
            x2 = self.drop1(F.relu(self.bn1_2(self.fc1_2(x2))))
            x2 = self.drop2(F.relu(self.bn2_2(self.fc2_2(x2))))
            x2 = self.fc3_2(x2)
            x2 = F.log_softmax(x2, -1)
            return x2, old_feature






