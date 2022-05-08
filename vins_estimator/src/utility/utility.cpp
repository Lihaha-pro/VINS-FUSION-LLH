/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "utility.h"
#include <iostream>

//从输入的加速度和重力加速度得到一个初始位姿
Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();//对输入的加速度归一化
    Eigen::Vector3d ng2{0, 0, 1.0};//这个是理想的重力加速度 //llh 就是z轴的正方向

    //旋转的过程中可能改变了yaw角，再把yaw旋转回原来的位置。
    //这个是沿z轴的旋转，因此不改变g沿z轴的方向。只是坐标系的x,y轴改变了。
    // 返回一个四元数，它表示两个任意向量ng1和ng2之间的旋转
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();

    //这里是对yaw取反之后乘在原来的R0，让yaw=0的一个措施
    //TODO:如果加入磁力计的话就在这里
    double yaw = Utility::R2ypr(R0).x();//?这里就是获得了yaw角的大小吗？
    std::cout << "origin yaw = " << yaw << std::endl;
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
