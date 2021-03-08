/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM3
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;          
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

/**
 * 3d-2d，投影MP集合点到当前帧，计算描述子距离，寻找当前帧匹配特征点
 * 1、遍历MP集合，根据跟踪视差计算搜索窗大小，在当前帧投影点相邻搜索窗中提取fast特征点
 * 2、计算MP与候选特征点的描述子距离，最佳距离小于阈值，最佳与次佳不在同一金字塔层级、或者次佳与最佳有一定距离，认为匹配成功
 * 3、更新F的mvpMapPoints，记录对应特征点
*/
int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints)
{
    int nmatches=0, left = 0, right = 0;

    // th！=1表示刚刚经历重定位，需要扩大搜索范围
    const bool bFactor = th!=1.0;

    // 遍历MP集合
    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {
        MapPoint* pMP = vpMapPoints[iMP];
        if(!pMP->mbTrackInView && !pMP->mbTrackInViewR)
            continue;

        // 不考虑远距离点，点的距离超过最远距离
        if(bFarPoints && pMP->mTrackDepth>thFarPoints)
            continue;

        if(pMP->isBad())
            continue;

        if(pMP->mbTrackInView)
        {
            // 根据距离估算的MP在当前帧投影特征点的金字塔层级
            const int &nPredictedLevel = pMP->mnTrackScaleLevel;

            // The size of the window will depend on the viewing direction
            // 跟踪视差较小时，搜索窗小一些，反之大一些。视差为当前帧与localMap观测帧平均视角之间的角度差
            float r = RadiusByViewingCos(pMP->mTrackViewCos);

            if(bFactor)
                r*=th;

            // 提取(x,y)位置处r范围矩形窗内的特征点
            const vector<size_t> vIndices =
                    F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

            if(!vIndices.empty()){
                // MP代表描述子
                const cv::Mat MPdescriptor = pMP->GetDescriptor();

                int bestDist=256;
                int bestLevel= -1;
                int bestDist2=256;
                int bestLevel2 = -1;
                int bestIdx =-1 ;

                // Get best and second matches with near keypoints
                // 遍历搜索窗中的特征点
                for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                {
                    const size_t idx = *vit;

                    // 特征点已经有对应的MP点了
                    if(F.mvpMapPoints[idx])
                        if(F.mvpMapPoints[idx]->Observations()>0)
                            continue;

                    // 双目
                    if(F.Nleft == -1 && F.mvuRight[idx]>0)
                    {
                        // 在X轴上的投影误差，太大不行
                        const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                        if(er>r*F.mvScaleFactors[nPredictedLevel])
                            continue;
                    }

                    // 描述子
                    const cv::Mat &d = F.mDescriptors.row(idx);
                    // 计算描述子距离
                    const int dist = DescriptorDistance(MPdescriptor,d);

                    // 记录最佳、次佳匹配点，对应层级
                    if(dist<bestDist)
                    {
                        bestDist2=bestDist;
                        bestDist=dist;
                        bestLevel2 = bestLevel;
                        bestLevel = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                    : (idx < F.Nleft) ? F.mvKeys[idx].octave
                                                                      : F.mvKeysRight[idx - F.Nleft].octave;
                        bestIdx=idx;
                    }
                    else if(dist<bestDist2)
                    {
                        bestLevel2 = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                     : (idx < F.Nleft) ? F.mvKeys[idx].octave
                                                                       : F.mvKeysRight[idx - F.Nleft].octave;
                        bestDist2=dist;
                    }
                }

                // Apply ratio to second match (only if best and second are in the same scale level)
                // 最佳距离小于阈值，最佳与次佳不在同一金字塔层级、或者次佳与最佳有一定距离，认为匹配成功
                if(bestDist<=TH_HIGH)
                {
                    if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                        continue;

                    if(bestLevel!=bestLevel2 || bestDist<=mfNNratio*bestDist2){
                        F.mvpMapPoints[bestIdx]=pMP;

                        if(F.Nleft != -1 && F.mvLeftToRightMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                            F.mvpMapPoints[F.mvLeftToRightMatch[bestIdx] + F.Nleft] = pMP;
                            nmatches++;
                            right++;
                        }

                        nmatches++;
                        left++;
                    }
                }
            }
        }

        // 双目右目，同样的做法
        if(F.Nleft != -1 && pMP->mbTrackInViewR){
            const int &nPredictedLevel = pMP->mnTrackScaleLevelR;
            if(nPredictedLevel != -1){
                float r = RadiusByViewingCos(pMP->mTrackViewCosR);
                // 提取(x,y)位置处r范围矩形窗内的特征点
                const vector<size_t> vIndices =
                        F.GetFeaturesInArea(pMP->mTrackProjXR,pMP->mTrackProjYR,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel,true);

                if(vIndices.empty())
                    continue;

                const cv::Mat MPdescriptor = pMP->GetDescriptor();

                int bestDist=256;
                int bestLevel= -1;
                int bestDist2=256;
                int bestLevel2 = -1;
                int bestIdx =-1 ;

                // Get best and second matches with near keypoints
                for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                {
                    const size_t idx = *vit;

                    if(F.mvpMapPoints[idx + F.Nleft])
                        if(F.mvpMapPoints[idx + F.Nleft]->Observations()>0)
                            continue;


                    const cv::Mat &d = F.mDescriptors.row(idx + F.Nleft);

                    const int dist = DescriptorDistance(MPdescriptor,d);

                    if(dist<bestDist)
                    {
                        bestDist2=bestDist;
                        bestDist=dist;
                        bestLevel2 = bestLevel;
                        bestLevel = F.mvKeysRight[idx].octave;
                        bestIdx=idx;
                    }
                    else if(dist<bestDist2)
                    {
                        bestLevel2 = F.mvKeysRight[idx].octave;
                        bestDist2=dist;
                    }
                }

                // Apply ratio to second match (only if best and second are in the same scale level)
                if(bestDist<=TH_HIGH)
                {
                    if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                        continue;

                    if(F.Nleft != -1 && F.mvRightToLeftMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                        F.mvpMapPoints[F.mvRightToLeftMatch[bestIdx]] = pMP;
                        nmatches++;
                        left++;
                    }


                    F.mvpMapPoints[bestIdx + F.Nleft]=pMP;
                    nmatches++;
                    right++;
                }
            }
        }
    }
    return nmatches;
}

/**
 * 角度较小时返回小值
*/
float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}

/**
 * 用基础矩阵计算极线，计算匹配点到极线距离，是否足够小
*/
bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2, const bool b1)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    // 在第二幅图像上的极线方程
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    // kp2到极线距离的平方
    const float dsqr = num*num/den;

    if(!b1)
        // 距离小于1个像素，认为合格
        return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
    else
        return dsqr<6.63*pKF2->mvLevelSigma2[kp2.octave];
}

/**
 * 同上，只是多个阈值系数
*/
bool ORBmatcher::CheckDistEpipolarLine2(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF2, const float unc)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    if(unc==1.f)
        return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
    else
        return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave]*unc;
}

/**
 * 3d-2d，通过词袋树划分特征点到node，在node中计算两帧的特征点描述子距离，寻找当前帧匹配特征点
 * 1、关键帧和当前帧的特征点都划分到了词袋树中不同节点中去了
 * 2、遍历节点集合，相同的节点才计算匹配点
 * 3、在同一节点中，遍历关键帧的特征点，当前帧的特征点，计算描述子距离
 * 4、描述子距离小于阈值，且次佳与最佳有一定差距，认为匹配上了
 * 5、根据特征点angle差值构造的直方图，删除非前三的离群匹配点
 * 6、vpMapPointMatches保存数据，当前帧特征点 - 关键帧MP
*/
int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    // 关键帧特征点-MP集合
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));

    // 关键帧特征点集合
    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches=0;

    // 旋转差统计直方图
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    // FeatureVector: map<NodeId, std::vector<unsigned int> > 这里的NodeId是词袋树中的id
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    // 遍历关键帧、当前帧的node集合
    while(KFit != KFend && Fit != Fend)
    {
        // 同一node才有可能是匹配点
        if(KFit->first == Fit->first)
        {
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF = Fit->second;

            // 遍历关键帧特征点
            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];

                MapPoint* pMP = vpMapPointsKF[realIdxKF];

                if(!pMP)
                    continue;

                if(pMP->isBad())
                    continue;

                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);

                int bestDist1=256;
                int bestIdxF =-1 ;
                int bestDist2=256;

                int bestDist1R=256;
                int bestIdxFR =-1 ;
                int bestDist2R=256;

                // 遍历当前帧特征点
                for(size_t iF=0; iF<vIndicesF.size(); iF++)
                {
                    if(F.Nleft == -1){
                        const unsigned int realIdxF = vIndicesF[iF];

                        if(vpMapPointMatches[realIdxF])
                            continue;

                        const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                        const int dist =  DescriptorDistance(dKF,dF);

                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdxF=realIdxF;
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;
                        }
                    }
                    else{
                        const unsigned int realIdxF = vIndicesF[iF];

                        if(vpMapPointMatches[realIdxF])
                            continue;

                        const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                        const int dist =  DescriptorDistance(dKF,dF);

                        if(realIdxF < F.Nleft && dist<bestDist1){
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdxF=realIdxF;
                        }
                        else if(realIdxF < F.Nleft && dist<bestDist2){
                            bestDist2=dist;
                        }

                        if(realIdxF >= F.Nleft && dist<bestDist1R){
                            bestDist2R=bestDist1R;
                            bestDist1R=dist;
                            bestIdxFR=realIdxF;
                        }
                        else if(realIdxF >= F.Nleft && dist<bestDist2R){
                            bestDist2R=dist;
                        }
                    }

                }

                // 描述子距离小于阈值，且次佳与最佳有一定差距，认为匹配上了，后面再检查特征点方向
                if(bestDist1<=TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMapPointMatches[bestIdxF]=pMP;

                        const cv::KeyPoint &kp =
                                (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :
                                (realIdxKF >= pKF -> NLeft) ? pKF -> mvKeysRight[realIdxKF - pKF -> NLeft]
                                                            : pKF -> mvKeys[realIdxKF];
                        // 检查特征点方向
                        if(mbCheckOrientation)
                        {
                            cv::KeyPoint &Fkp =
                                    (!pKF->mpCamera2 || F.Nleft == -1) ? F.mvKeys[bestIdxF] :
                                    (bestIdxF >= F.Nleft) ? F.mvKeysRight[bestIdxF - F.Nleft]
                                                          : F.mvKeys[bestIdxF];

                            float rot = kp.angle-Fkp.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        nmatches++;
                    }

                    if(bestDist1R<=TH_LOW)
                    {
                        if(static_cast<float>(bestDist1R)<mfNNratio*static_cast<float>(bestDist2R) || true)
                        {
                            vpMapPointMatches[bestIdxFR]=pMP;

                            const cv::KeyPoint &kp =
                                    (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :
                                    (realIdxKF >= pKF -> NLeft) ? pKF -> mvKeysRight[realIdxKF - pKF -> NLeft]
                                                                : pKF -> mvKeys[realIdxKF];

                            if(mbCheckOrientation)
                            {
                                cv::KeyPoint &Fkp =
                                        (!F.mpCamera2) ? F.mvKeys[bestIdxFR] :
                                        (bestIdxFR >= F.Nleft) ? F.mvKeysRight[bestIdxFR - F.Nleft]
                                                               : F.mvKeys[bestIdxFR];

                                float rot = kp.angle-Fkp.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdxFR);
                            }
                            nmatches++;
                        }
                    }
                }

            }

            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }

    // 检查特征点方向，删除方向差不合群的匹配点
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

/**
 * 3d-2d，闭环关键帧与其共视关键帧的MP，投影到当前关键帧中，计算描述子距离，寻找当前帧匹配特征点
 * 1、投影MP到当前关键帧，深度值不得超过估计范围，视差不得超过60°
 * 2、通过深度值估计特征点的层级，搜索框提取fast特征点集合
 * 3、匹配层级上下只允许浮动一层，描述子距离小于阈值即可
 * 4、当前关键帧已有的MP不参与上面的过程
 * @param pKF           当前关键帧
 * @param Scw           当前关键帧位姿
 * @param vpPoints      闭环关键帧与其共视关键帧的地图点
 * @param vpMatched     当前关键帧已经匹配的点
 * @param th            搜索窗大小
 * @param ratioHamming  描述子距离阈值系数
*/
int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints,
                                   vector<MapPoint*> &vpMatched, int th, float ratioHamming)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    // 当前关键帧位姿
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    // 遍历闭环关键帧与其共视关键帧的地图点
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        // 投影到当前相机坐标系
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        const float x = p3Dc.at<float>(0);
        const float y = p3Dc.at<float>(1);
        const float z = p3Dc.at<float>(2);

        // 投影到像素坐标系
        const cv::Point2f uv = pKF->mpCamera->project(cv::Point3f(x,y,z));

        // Point must be inside the image
        if(!pKF->IsInImage(uv.x,uv.y))
            continue;

        // Depth must be inside the scale invariance region of the point
        // 深度值是否在有效范围内
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        // 地图点的观测法矢
        cv::Mat Pn = pMP->GetNormal();

        // 视差不可以超过60°
        if(PO.dot(Pn)<0.5*dist)
            continue;

        // 通过深度值估计特征点的层级
        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        // 搜索范围乘上金字塔尺度
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        // 搜索窗中的候选特征点集合
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x,uv.y,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        // 遍历搜索窗中的候选特征点集合
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            // 层级上下只允许浮动一层
            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);
            // 记录最佳描述子距离
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }
        // 描述子距离小于阈值，认为匹配成功
        if(bestDist<=TH_LOW*ratioHamming)
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }

    return nmatches;
}

/**
 * 同上，记录当前帧2d点 - 闭环Map点对应的参考关键帧
*/
int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, const std::vector<KeyFrame*> &vpPointsKFs,
                       std::vector<MapPoint*> &vpMatched, std::vector<KeyFrame*> &vpMatchedKF, int th, float ratioHamming)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];
        KeyFrame* pKFi = vpPointsKFs[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW*ratioHamming)
        {
            vpMatched[bestIdx] = pMP;
            vpMatchedKF[bestIdx] = pKFi;
            nmatches++;
        }

    }

    return nmatches;
}

/**
 * 2d-2d，两帧搜索匹配特征点
 * 1、F1特征点在F2中，矩形窗范围内搜索匹配点
 * 2、最佳匹配点条件：最佳匹配点要求描述子距离小于阈值，并且要与次佳匹配点拉开距离
 * 3、构建梯度方向差直方图，取数量最多前三，其余的删除匹配
 * [经验]匹配点怎么搜？窗口搜索，描述子距离判断，梯度方向直方图
 * @param F1            前一帧
 * @param F2            当前帧
 * @param vbPrevMatched 前一帧待匹配特征点 inputAndOutput，output没有用到，可以忽略
 * @param vnMatches12   匹配情况，-1表示未匹配上 output
 * @param windowSize    F1特征点在F2中搜索矩形框半径
 * @return              返回匹配数量
*/
int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    // [效率]容器提前初始化好
    int nmatches=0;
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

    // 梯度方向差直方图，30个bin
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    // [效率]频繁除法用乘法代替
    const float factor = 1.0f/HISTO_LENGTH;

    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

    // [效率]for循环size，只计算一次
    // 遍历F1特征点，寻找F2中的匹配点
    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        if(level1>0)
            continue;

        // 提取(x,y)位置处windowSize范围矩形窗内的特征点，保存索引
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

        if(vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        // 遍历F2候选矩形框中的特征点，计算最佳匹配、次佳匹配点
        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            // 描述子距离
            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }

        // [经验]最佳匹配点要求距离小于阈值，并且要与次佳匹配点拉开距离
        if(bestDist<=TH_LOW)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                // [经验]F2中的某个点被F1的多个点匹配到了，删除之前的匹配关系，保留当前匹配
                // 为什么不用匹配距离来判断是否保留当前匹配 todo
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    // 如果特征点一对一匹配，nmatches正好是特征点数量，否则会小于特征点数量
                    nmatches--;
                }
                // 记录匹配关系
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;
                
                // 梯度方向检查
                if(mbCheckOrientation)
                {
                    // 梯度方向差，加入直方图对应bin中
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

    // 梯度方向匹配检查
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        // 找出直方图中bin数量最多的前三个
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        // 删除落入非前三数量bin中的匹配关系，认为这些少量的匹配点是离群点
        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    // 最终匹配点数量
    return nmatches;
}

/**
 * 3d,2d都已知，计算匹配关系；通过bow，对两个关键帧的点进行匹配，用于闭环检测
 * 1、两个关键帧的特征点都划分到了词袋树中不同节点中去了
 * 2、遍历节点集合，相同的节点才计算匹配点
 * 3、在同一节点中，遍历两个关键帧的特征点，计算描述子距离
 * 4、描述子距离小于阈值，且次佳与最佳有一定差距，认为匹配上了
 * 5、根据特征点angle差值构造的直方图，删除非前三的离群匹配点
 * 6、vpMatches12保存数据，当pKF1的特征点 - pKF2的MP
*/
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    // 特征点，特征点词向量，地图点，描述子
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    // 特征点，特征点词向量，地图点，描述子
    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    // 遍历node
    while(f1it != f1end && f2it != f2end)
    {
        // 相同node才有可能匹配
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];
                if(pKF1 -> NLeft != -1 && idx1 >= pKF1 -> mvKeysUn.size()){
                    continue;
                }

                MapPoint* pMP1 = vpMapPoints1[idx1];
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row(idx1);

                int bestDist1=256;
                int bestIdx2 =-1 ;
                int bestDist2=256;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    if(pKF2 -> NLeft != -1 && idx2 >= pKF2 -> mvKeysUn.size()){
                        continue;
                    }

                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    // 已经匹配过的特征点不再参与匹配
                    if(vbMatched2[idx2] || !pMP2)
                        continue;

                    if(pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);

                    int dist = DescriptorDistance(d1,d2);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation)
                        {
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

/**
 * 2d-2d，利用基础矩阵极线约束，bow加速匹配两帧特征点，vMatchedPairs存KF1、KF2特征点id
 * 1、计算KF1相机在KF2下面的像素投影，极点
 * 2、遍历两帧对应的词向量节点，计算描述子距离
 * 3、描述子距离小于阈值，点到极点距离要够大（MP离KF1才足够远），极线约束满足，特征点方向直方图过滤
*/
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
{    
    // 特征点词向量
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    // KF1相机光心在KF2系的坐标
    cv::Mat C2 = R2w*Cw+t2w;

    // KF1相机在KF2下面的像素投影，极点
    cv::Point2f ep = pKF2->mpCamera->project(C2);

    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    cv::Mat R12;
    cv::Mat t12;

    cv::Mat Rll,Rlr,Rrl,Rrr;
    cv::Mat tll,tlr,trl,trr;

    GeometricCamera* pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;

    // 单目，两帧之间的变换
    if(!pKF1->mpCamera2 && !pKF2->mpCamera2){
        R12 = R1w*R2w.t();
        t12 = -R1w*R2w.t()*t2w+t1w;
    }
    else{
        // 双目
        Rll = pKF1->GetRotation() * pKF2->GetRotation().t();
        Rlr = pKF1->GetRotation() * pKF2->GetRightRotation().t();
        Rrl = pKF1->GetRightRotation() * pKF2->GetRotation().t();
        Rrr = pKF1->GetRightRotation() * pKF2->GetRightRotation().t();

        tll = pKF1->GetRotation() * (-pKF2->GetRotation().t() * pKF2->GetTranslation()) + pKF1->GetTranslation();
        tlr = pKF1->GetRotation() * (-pKF2->GetRightRotation().t() * pKF2->GetRightTranslation()) + pKF1->GetTranslation();
        trl = pKF1->GetRightRotation() * (-pKF2->GetRotation().t() * pKF2->GetTranslation()) + pKF1->GetRightTranslation();
        trr = pKF1->GetRightRotation() * (-pKF2->GetRightRotation().t() * pKF2->GetRightTranslation()) + pKF1->GetRightTranslation();
    }

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);
    vector<int> vMatches12(pKF1->N,-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    // 遍历两帧对应的词向量节点
    while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first)
        {
            // 遍历节点下的特征点
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];
                
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);
                
                // If there is already a MapPoint skip
                if(pMP1)
                {
                    continue;
                }

                const bool bStereo1 = (!pKF1->mpCamera2 && pKF1->mvuRight[idx1]>=0);

                if(bOnlyStereo)
                    if(!bStereo1)
                        continue;


                const cv::KeyPoint &kp1 = (pKF1 -> NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                                : (idx1 < pKF1 -> NLeft) ? pKF1 -> mvKeys[idx1]
                                                                                         : pKF1 -> mvKeysRight[idx1 - pKF1 -> NLeft];

                const bool bRight1 = (pKF1 -> NLeft == -1 || idx1 < pKF1 -> NLeft) ? false
                                                                                   : true;
                //if(bRight1) continue;
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
                
                int bestDist = TH_LOW;
                int bestIdx2 = -1;
                
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];
                    
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);
                    
                    // If we have already matched or there is a MapPoint skip
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = (!pKF2->mpCamera2 &&  pKF2->mvuRight[idx2]>=0);

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;
                    
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
                    
                    const int dist = DescriptorDistance(d1,d2);
                    
                    // 描述子距离需要小于阈值
                    if(dist>TH_LOW || dist>bestDist)
                        continue;

                    const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                                    : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                             : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];
                    const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                                       : true;

                    if(!bStereo1 && !bStereo2 && !pKF1->mpCamera2)
                    {
                        // 特征点到极点的距离，如果太近，表示对应MP离KF1太近
                        const float distex = ep.x-kp2.pt.x;
                        const float distey = ep.y-kp2.pt.y;
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                        {
                            continue;
                        }
                    }

                    if(pKF1->mpCamera2 && pKF2->mpCamera2){
                        if(bRight1 && bRight2){
                            R12 = Rrr;
                            t12 = trr;

                            pCamera1 = pKF1->mpCamera2;
                            pCamera2 = pKF2->mpCamera2;
                        }
                        else if(bRight1 && !bRight2){
                            R12 = Rrl;
                            t12 = trl;

                            pCamera1 = pKF1->mpCamera2;
                            pCamera2 = pKF2->mpCamera;
                        }
                        else if(!bRight1 && bRight2){
                            R12 = Rlr;
                            t12 = tlr;

                            pCamera1 = pKF1->mpCamera;
                            pCamera2 = pKF2->mpCamera2;
                        }
                        else{
                            R12 = Rll;
                            t12 = tll;

                            pCamera1 = pKF1->mpCamera;
                            pCamera2 = pKF2->mpCamera;
                        }

                    }

                    
                    // 点到极线距离小于阈值，认为匹配上了
                    if(pCamera1->epipolarConstrain(pCamera2,kp1,kp2,R12,t12,pKF1->mvLevelSigma2[kp1.octave],pKF2->mvLevelSigma2[kp2.octave])||bCoarse) // MODIFICATION_2
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }
                
                // 特征点旋转检查
                if(bestIdx2>=0)
                {
                    const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                                    : (bestIdx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[bestIdx2]
                                                                                                 : pKF2 -> mvKeysRight[bestIdx2 - pKF2 -> NLeft];
                    vMatches12[idx1]=bestIdx2;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    return nmatches;
}

/**
 * 同上，多一个匹配之后进行三角化，vMatchedPoints存KF1特征点id - 三角化点
*/
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                        vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, vector<cv::Mat> &vMatchedPoints)
{
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w*Cw+t2w;

    cv::Point2f ep = pKF2->mpCamera->project(C2);

    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    GeometricCamera* pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;
    cv::Mat Tcw1,Tcw2;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);
    vector<int> vMatches12(pKF1->N,-1);

    vector<cv::Mat> vMatchesPoints12(pKF1 -> N);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();
    int right = 0;
    while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);

                // If there is already a MapPoint skip
                if(pMP1)
                    continue;

                const cv::KeyPoint &kp1 = (pKF1 -> NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                                : (idx1 < pKF1 -> NLeft) ? pKF1 -> mvKeys[idx1]
                                                                                            : pKF1 -> mvKeysRight[idx1 - pKF1 -> NLeft];

                const bool bRight1 = (pKF1 -> NLeft == -1 || idx1 < pKF1 -> NLeft) ? false
                                                                                    : true;


                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                int bestDist = TH_LOW;
                int bestIdx2 = -1;

                cv::Mat bestPoint;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

                    // If we have already matched or there is a MapPoint skip
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

                    const int dist = DescriptorDistance(d1,d2);

                    if(dist>TH_LOW || dist>bestDist){
                        continue;
                    }


                    const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                                    : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                                : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];
                    const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                                        : true;

                    if(bRight1){
                        Tcw1 = pKF1->GetRightPose();
                        pCamera1 = pKF1->mpCamera2;
                    } else{
                        Tcw1 = pKF1->GetPose();
                        pCamera1 = pKF1->mpCamera;
                    }

                    if(bRight2){
                        Tcw2 = pKF2->GetRightPose();
                        pCamera2 = pKF2->mpCamera2;
                    } else{
                        Tcw2 = pKF2->GetPose();
                        pCamera2 = pKF2->mpCamera;
                    }

                    cv::Mat x3D;
                    if(pCamera1->matchAndtriangulate(kp1,kp2,pCamera2,Tcw1,Tcw2,pKF1->mvLevelSigma2[kp1.octave],pKF2->mvLevelSigma2[kp2.octave],x3D)){
                        bestIdx2 = idx2;
                        bestDist = dist;
                        bestPoint = x3D;
                    }

                }

                if(bestIdx2>=0)
                {
                    const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                                    : (bestIdx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[bestIdx2]
                                                                                                    : pKF2 -> mvKeysRight[bestIdx2 - pKF2 -> NLeft];
                    vMatches12[idx1]=bestIdx2;
                    vMatchesPoints12[idx1] = bestPoint;
                    nmatches++;
                    if(bRight1) right++;

                    if(mbCheckOrientation)
                    {
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
        vMatchedPoints.push_back(vMatchesPoints12[i]);
    }
    return nmatches;
}

/**
 * 关键帧与Map点融合，更新MP
 * MP集合投影到KF中，寻找匹配2d点，如果该2d点有自己对应的MP1，那么比较MP与MP1的观测，谁的观测多，谁就留下，另一个被替换；如果没有MP1，就添加一个MP
*/
int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th, const bool bRight)
{
    cv::Mat Rcw,tcw, Ow;
    GeometricCamera* pCamera;

    if(bRight){
        Rcw = pKF->GetRightRotation();
        tcw = pKF->GetRightTranslation();
        Ow = pKF->GetRightCameraCenter();

        pCamera = pKF->mpCamera2;
    }
    else{
        Rcw = pKF->GetRotation();
        tcw = pKF->GetTranslation();
        Ow = pKF->GetCameraCenter();

        pCamera = pKF->mpCamera;
    }

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    int nFused=0;

    const int nMPs = vpMapPoints.size();

    // For debbuging
    int count_notMP = 0, count_bad=0, count_isinKF = 0, count_negdepth = 0, count_notinim = 0, count_dist = 0, count_normal=0, count_notidx = 0, count_thcheck = 0;
    // 遍历MP集合
    for(int i=0; i<nMPs; i++)
    {
        MapPoint* pMP = vpMapPoints[i];

        if(!pMP)
        {
            count_notMP++;
            continue;
        }

        /*if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;*/
        if(pMP->isBad())
        {
            count_bad++;
            continue;
        }
        else if(pMP->IsInKeyFrame(pKF))
        {
            count_isinKF++;
            continue;
        }


        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc = Rcw*p3Dw + tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
        {
            count_negdepth++;
            continue;
        }

        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0);
        const float y = p3Dc.at<float>(1);
        const float z = p3Dc.at<float>(2);

        const cv::Point2f uv = pCamera->project(cv::Point3f(x,y,z));

        // Point must be inside the image
        if(!pKF->IsInImage(uv.x,uv.y))
        {
            count_notinim++;
            continue;
        }

        const float ur = uv.x-bf*invz;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
        {
            count_dist++;
            continue;
        }

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
        {
            count_normal++;
            continue;
        }

        int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x,uv.y,radius,bRight);

        if(vIndices.empty())
        {
            count_notidx++;
            continue;
        }

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            size_t idx = *vit;
            const cv::KeyPoint &kp = (pKF -> NLeft == -1) ? pKF->mvKeysUn[idx]
                                                          : (!bRight) ? pKF -> mvKeys[idx]
                                                                      : pKF -> mvKeysRight[idx];

            const int &kpLevel= kp.octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            if(pKF->mvuRight[idx]>=0)
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = uv.x-kpx;
                const float ey = uv.y-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;
            }
            else
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = uv.x-kpx;
                const float ey = uv.y-kpy;
                const float e2 = ex*ex+ey*ey;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }

            if(bRight) idx += pKF->NLeft;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            // 关键帧的MP
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                {
                    // 谁的观测多，谁就保留
                    if(pMPinKF->Observations()>pMP->Observations())
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else
            {
                // 添加MP与特征点对应关系
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
        else
            count_thcheck++;

    }

    /*cout << "count_notMP = " << count_notMP << endl;
    cout << "count_bad = " << count_bad << endl;
    cout << "count_isinKF = " << count_isinKF << endl;
    cout << "count_negdepth = " << count_negdepth << endl;
    cout << "count_notinim = " << count_notinim << endl;
    cout << "count_dist = " << count_dist << endl;
    cout << "count_normal = " << count_normal << endl;
    cout << "count_notidx = " << count_notidx << endl;
    cout << "count_thcheck = " << count_thcheck << endl;
    cout << "tot fused points: " << nFused << endl;*/
    return nFused;
}

/**
 * 闭环帧及其共视关键帧Map点，与当前关键帧融合，更新MP
*/
int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;

    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // Project into Image
        const float x = p3Dc.at<float>(0);
        const float y = p3Dc.at<float>(1);
        const float z = p3Dc.at<float>(2);

        const cv::Point2f uv = pKF->mpCamera->project(cv::Point3f(x,y,z));

        // Point must be inside the image
        if(!pKF->IsInImage(uv.x,uv.y))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x,uv.y,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

/**
 * 已知两关键帧的变换，计算新匹配点，忽略已经匹配过的点
 * 1、KF1的MP投影到KF2中，搜索窗搜索特征点，计算描述子距离，筛选
 * 2、KF2的MP投影到KF1中，搜索窗搜索特征点，计算描述子距离，筛选
 * 3、综合得到最终匹配结果，存vpMatches12
*/
int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    // 已经匹配的点记录一下
    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = get<0>(pMP->GetIndexInKeyFrame(pKF2));
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

/**
 * 3d-2d，相邻帧跟踪，前一帧MP在当前帧中寻找匹配2d点
 * 1、计算当前帧在前一帧中的位置，判断相对前一帧是向前运动，还是向后运动，用于后面搜索特征点的层级判断
 * 2、前一帧MP投影到当前帧中，根据金字塔层级确定搜索半径，根据前向、后向运动确定搜索特征点层级范围，在当前帧搜索窗中搜索候选特征点
 * 3、描述子距离小于阈值，方向直方图过滤
 * 4、更新当前帧mvpMapPoints，当前帧特征点id，前一帧MP
*/
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat twc = -Rcw.t()*tcw;

    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);

    // 当前帧在前一帧中的位置
    const cv::Mat tlc = Rlw*twc+tlw;

    // [经验]z轴是从相机指向场景的，如果z大于零，表示当前帧在前一帧的z轴正方向上，表示相对于前一帧向靠近场景的方向运动了
    const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono;
    const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;

    // 遍历前一帧的MP
    for(int i=0; i<LastFrame.N; i++)
    {
        MapPoint* pMP = LastFrame.mvpMapPoints[i];
        if(pMP)
        {
            if(!LastFrame.mvbOutlier[i])
            {
                // Project
                // 投影到当前相机坐标系
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                // 深度值判断
                if(invzc<0)
                    continue;

                cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dc);
                
                // 像素边界判断
                if(uv.x<CurrentFrame.mnMinX || uv.x>CurrentFrame.mnMaxX)
                    continue;
                if(uv.y<CurrentFrame.mnMinY || uv.y>CurrentFrame.mnMaxY)
                    continue;
                
                // 前一帧对应特征点的金字塔层级
                // LastFrame.Nleft == -1表示单目，i < LastFrame.Nleft表示双目的左目
                int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                                                                    : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                // Search in a window. Size depends on scale
                // 搜索窗半径，乘上层级尺度
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                vector<size_t> vIndices2;

                /**
                 * 以向前运动为例，看到的图像是放大的，那么之前的某个特征点，在相同的搜索半径下，需要在更高的尺度下（相同的半径，汇聚更多点）才能被辨认出来。
                 * 举个例子，前一时刻我们找到了一个特征点，看上去就是一个点；当前时刻图像放大了，这个点被放大了，夸张一点发散成一块马赛克了，如果想要
                 * 找到之前的那个点，就需要把图像汇聚一下，把马赛克块重新汇聚成一个点，等价于相同的搜索半径下，我们要到更高层级的金字塔图像中去找特征点。
                 * 否则，相同的搜索半径下，还是在当前层级中找，找到的只是马赛克中的某个点。
                */
                if(bForward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave);
                else if(bBackward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, 0, nLastOctave);
                else
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave-1, nLastOctave+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    const size_t i2 = *vit;

                    if(CurrentFrame.mvpMapPoints[i2])
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                            continue;

                    if(CurrentFrame.Nleft == -1 && CurrentFrame.mvuRight[i2]>0)
                    {
                        const float ur = uv.x - CurrentFrame.mbf*invzc;
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)
                            continue;
                    }

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=TH_HIGH)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.mvKeysUn[i]
                                                                    : (i < LastFrame.Nleft) ? LastFrame.mvKeys[i]
                                                                                            : LastFrame.mvKeysRight[i - LastFrame.Nleft];

                        cv::KeyPoint kpCF = (CurrentFrame.Nleft == -1) ? CurrentFrame.mvKeysUn[bestIdx2]
                                                                        : (bestIdx2 < CurrentFrame.Nleft) ? CurrentFrame.mvKeys[bestIdx2]
                                                                                                            : CurrentFrame.mvKeysRight[bestIdx2 - CurrentFrame.Nleft];
                        float rot = kpLF.angle-kpCF.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
                if(CurrentFrame.Nleft != -1){
                    cv::Mat x3Dr = CurrentFrame.mTrl.colRange(0,3).rowRange(0,3) * x3Dc + CurrentFrame.mTrl.col(3);

                    cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dr);

                    int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                                                                        : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                    // Search in a window. Size depends on scale
                    float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                    vector<size_t> vIndices2;

                    if(bForward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave, -1,true);
                    else if(bBackward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, 0, nLastOctave, true);
                    else
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave-1, nLastOctave+1, true);

                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                    {
                        const size_t i2 = *vit;
                        if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft])
                            if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft]->Observations()>0)
                                continue;

                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2 + CurrentFrame.Nleft);

                        const int dist = DescriptorDistance(dMP,d);

                        if(dist<bestDist)
                        {
                            bestDist=dist;
                            bestIdx2=i2;
                        }
                    }

                    if(bestDist<=TH_HIGH)
                    {
                        CurrentFrame.mvpMapPoints[bestIdx2 + CurrentFrame.Nleft]=pMP;
                        nmatches++;
                        if(mbCheckOrientation)
                        {
                            cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.mvKeysUn[i]
                                                                        : (i < LastFrame.Nleft) ? LastFrame.mvKeys[i]
                                                                                                : LastFrame.mvKeysRight[i - LastFrame.Nleft];

                            cv::KeyPoint kpCF = CurrentFrame.mvKeysRight[bestIdx2];

                            float rot = kpLF.angle-kpCF.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2  + CurrentFrame.Nleft);
                        }
                    }

                }
            }
        }
    }

    //Apply rotation consistency
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

/**
 * 3d-2d，关键帧MP在当前帧中寻找匹配2d点
 * 1、投影之后，深度值满足距离范围
 * 2、描述子距离小于阈值，方向直方图过滤
*/
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP)
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dc);

                if(uv.x<CurrentFrame.mnMinX || uv.x>CurrentFrame.mnMaxX)
                    continue;
                if(uv.y<CurrentFrame.mnMinY || uv.y>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                cv::Mat PO = x3Dw-Ow;
                float dist3D = cv::norm(PO);

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                // Search in a window
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x, uv.y, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=ORBdist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

/**
 * 最大前三
*/
void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}

/**
 * 计算描述子距离
 * Bit set count operation from
 * http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
 * 简单的可以是汉明距离
*/
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} //namespace ORB_SLAM
