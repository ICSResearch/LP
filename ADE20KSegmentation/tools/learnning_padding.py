import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class LearningPaddingAllStride(nn.Module):
    def __init__(self, in_channels, stride=1):
        super(LearningPaddingAllStride, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                 kernel_size=(1, 1), stride=1)
        self.conv_cor = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                  kernel_size=(2, 2), stride=2)

        self.stride = stride

    def forward(self, x):
        row = torch.cat((x[:, :, 0:self.stride], x[:, :, -self.stride:]), dim=2)
        col = torch.cat((x[:, :, :, 0:self.stride], x[:, :, :, -self.stride:]), dim=3)

        corner = torch.cat(
            (x[:, :, 0:2 * self.stride, 0:2 * self.stride], x[:, :, 0:2 * self.stride, -2 * self.stride:],
             x[:, :, -2 * self.stride:, 0:2 * self.stride], x[:, :, -2 * self.stride:, -2 * self.stride:]), dim=3)

        row = self.conv1x1(row)
        col = self.conv1x1(col)
        corner = self.conv_cor(corner)

        x = torch.cat((row[:, :, 0:self.stride], x, row[:, :, self.stride:]), dim=2)

        col = torch.cat((corner[:, :, :, 0:self.stride * 2], col, corner[:, :, :, self.stride * 2:]), dim=2)

        x = torch.cat((col[:, :, :, 0:self.stride], x, col[:, :, :, self.stride:]), dim=3)

        return x


class LearningPaddingA_AllStride(nn.Module):
    def __init__(self, in_channels, stride=1):
        super().__init__()
        self.theTa = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, groups=in_channels,
                               stride=1, bias=False, kernel_size=(1, 1))
        self.phi = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, groups=in_channels,
                             stride=1, bias=False, kernel_size=(1, 1))
        self.g = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, groups=in_channels,
                           stride=1, bias=False, kernel_size=(1, 1))
        self.w = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, groups=in_channels,
                           stride=1, kernel_size=(1, 1))

        self.stride = stride

        self.w.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()

    def forward(self, x):
        row_now = torch.cat((x[:, :, 0:self.stride], x[:, :, -self.stride:]), dim=2)
        row_past = torch.cat((x[:, :, self.stride:2 * self.stride], x[:, :, -2 * self.stride:-self.stride]), dim=2)
        col_now = torch.cat((x[:, :, :, 0:self.stride], x[:, :, :, -self.stride:]), dim=3)
        col_past = torch.cat((x[:, :, :, self.stride:2 * self.stride], x[:, :, :, -2 * self.stride:-self.stride]),
                             dim=3)

        corner_now = torch.cat((x[:, :, 0:self.stride, 0:self.stride], x[:, :, 0:self.stride, -self.stride:],
                                x[:, :, -self.stride:, 0:self.stride], x[:, :, -self.stride:, -self.stride:]), dim=3)
        corner_past = torch.cat((x[:, :, self.stride:2 * self.stride, self.stride:2 * self.stride],
                                 x[:, :, self.stride:2 * self.stride, -2 * self.stride:-self.stride],
                                 x[:, :, -2 * self.stride:-self.stride, self.stride:2 * self.stride],
                                 x[:, :, -2 * self.stride:-self.stride, -2 * self.stride:-self.stride]), dim=3)

        g_row = self.g(row_now).permute(0, 1, 3, 2)
        g_col = self.g(col_now).permute(0, 1, 3, 2)
        g_corner = self.g(corner_now).permute(0, 1, 3, 2)

        theta_row = self.theTa(row_now).permute(0, 1, 3, 2)
        theta_col = self.theTa(col_now).permute(0, 1, 3, 2)
        theta_corner = self.theTa(corner_now).permute(0, 1, 3, 2)

        phi_row = self.phi(row_past)
        phi_col = self.phi(col_past)
        phi_corner = self.phi(corner_past)

        f_row = torch.matmul(theta_row, phi_row)
        f_col = torch.matmul(theta_col, phi_col)
        f_corner = torch.matmul(theta_corner, phi_corner)

        f_row = F.softmax(f_row, dim=-1)
        f_col = F.softmax(f_col, dim=-1)
        f_corner = F.softmax(f_corner, dim=-1)

        row = torch.matmul(f_row, g_row).permute(0, 1, 3, 2)
        col = torch.matmul(f_col, g_col).permute(0, 1, 3, 2)
        corner = torch.matmul(f_corner, g_corner).permute(0, 1, 3, 2)

        row = self.w(row)
        col = self.w(col)
        corner = self.w(corner)

        x = torch.cat((row[:, :, 0:self.stride], x, row[:, :, self.stride:]), dim=2)
        col = torch.cat((corner[:, :, :, 0:2 * self.stride], col, corner[:, :, :, 2 * self.stride:]), dim=2)
        x = torch.cat((col[:, :, :, 0:self.stride], x, col[:, :, :, self.stride:]), dim=3)

        return x
