/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../estimator/parameters.h"

// 视觉损失函数
/*
2: 残差长度2(err_x, err_y)
7: 第1个优化参数pose_i的长度(para_Pose[imu_i]=(px,py,pz,qx,qy,qz,qw) )
7: 第2个优化参数pose_j的长度(para_Pose[imu_j])
7: 第3个优化参数外参的长度(para_Ex_Pose[0])
1: 第4个优化参数feature_inverse_depth的长度(para_Feature[feature_index])
1: 第5个优化参数td传感器时延(para_Td[0])
*/
class ProjectionTwoFrameOneCamFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1>
{
  public:
    // 两帧之间使用一个相机进行重投影
    ProjectionTwoFrameOneCamFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
    				   const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
    				   const double _td_i, const double _td_j);

    /* Evaluate 计算所有状态变量构成的残差和雅克比矩阵
    这个函数通过传入的优化变量值parameters，以及先验值（对于先验残差就是上一时刻的先验残差last_marginalization_info，
    对于IMU就是预计分值pre_integrations[1]，对于视觉就是空间的的像素坐标pts_i, pts_j）
    可以计算出各项残差值residuals，以及残差对应个优化变量的雅克比矩阵jacobians。
    原文链接：https://blog.csdn.net/weixin_44580210/article/details/95748091*/
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    
    void check(double **parameters);

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Vector3d velocity_i, velocity_j;
    double td_i, td_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};
