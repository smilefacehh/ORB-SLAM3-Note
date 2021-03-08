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

#include "Frame.h"

#include "G2oTypes.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "GeometricCamera.h"

#include <thread>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>

namespace ORB_SLAM3
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

//For stereo fisheye matching
cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

Frame::Frame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false)
{
}


//Copy Constructor
/**
 * 拷贝构造函数
*/
Frame::Frame(const Frame &frame)
    :mpcpi(frame.mpcpi),mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
     mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
     mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
     mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
     monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
     mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
     mTlr(frame.mTlr.clone()), mRlr(frame.mRlr.clone()), mtlr(frame.mtlr.clone()), mTrl(frame.mTrl.clone()), mTimeStereoMatch(frame.mTimeStereoMatch), mTimeORB_Ext(frame.mTimeORB_Ext)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++){
            mGrid[i][j]=frame.mGrid[i][j];
            if(frame.Nleft > 0){
                mGridRight[i][j] = frame.mGridRight[i][j];
            }
        }

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);

    if(!frame.mVw.empty())
        mVw = frame.mVw.clone();

    mmProjectPoints = frame.mmProjectPoints;
    mmMatchedInImage = frame.mmMatchedInImage;
}

/**
 * 双目构造函数
 * 1、开辟两个线程提取特征点
 * 2、双目匹配，在右目中寻找左目的匹配特征点
 * 3、将所有特征点分配到网格
*/
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false),
     mpCamera(pCamera) ,mpCamera2(nullptr), mTimeStereoMatch(0), mTimeORB_Ext(0)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    // 开辟两个线程提取特征点
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,0,0);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,0,0);
    threadLeft.join();
    threadRight.join();
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    // 左目特征点数量
    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    // 左目特征点畸变校正
    UndistortKeyPoints();

#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    /**
     * 双目匹配，在右目中寻找左目的匹配特征点
     * 行搜索 - 滑窗对比 - 亚像素修正
     * 1、遍历右目特征点，记录所在行（上下浮动一行），后面左目特征点在相同行找右目候选点
     * 2、遍历左目特征点，得到所在行的右目候选点集合；限制左右目的匹配点在相同行上
     * 3、在行上选取描述子距离最小的点，作为最佳候选点
     * 4、在左目点对应金字塔层级，取半径5的搜索窗；在最佳候选点周围也取半径5的搜索窗，另外搜索窗水平左右滑动5个像素，计算搜索窗相似度
     * 5、选择最佳匹配搜索窗，联合左右计算亚像素（小于1个像素）插值，最后确定右目匹配点的位置
     * 5、对所有左右目匹配点，剔除上面搜索窗匹配精度较低的匹配点
    */
    ComputeStereoMatches();
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif


    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);
    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);



        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF)
    {
        if(!pPrevF->mVw.empty())
            mVw = pPrevF->mVw.clone();
    }
    else
    {
        mVw = cv::Mat::zeros(3,1,CV_32F);
    }

    // 将所有特征点分配到网格
    AssignFeaturesToGrid();

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mTlr = cv::Mat(3,4,CV_32F);
    mTrl = cv::Mat(3,4,CV_32F);
    mvStereo3Dpoints = vector<cv::Mat>(0);
    monoLeft = -1;
    monoRight = -1;
}

/**
 * RGBD构造函数
 * 1、提取orb特征点、描述子
 * 2、提取特征点深度值
 * 3、划分特征点到网格
*/
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false),
     mpCamera(pCamera),mpCamera2(nullptr), mTimeStereoMatch(0), mTimeORB_Ext(0)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    // 提取orb特征点、描述子
    ExtractORB(0,imGray,0,0);
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    // 特征点畸变校正
    UndistortKeyPoints();

    // 深度图像提取特征点深度值
    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mTlr = cv::Mat(3,4,CV_32F);
    mTrl = cv::Mat(3,4,CV_32F);
    mvStereo3Dpoints = vector<cv::Mat>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}

/**
 * 单目构造函数
 * 1、提取orb特征点、描述子；存mvKeys、mDescriptors
 * 2、特征点畸变校正
 * 3、计算图像畸变校正之后的边界，数值是相对原始图像坐标系的
 * 4、将所有特征点分配到网格
*/
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera),
     mpCamera2(nullptr), mTimeStereoMatch(0), mTimeORB_Ext(0)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    // 提取orb特征点、描述子；存mvKeys、mDescriptors
    ExtractORB(0,imGray,0,1000);
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    // 特征点畸变校正
    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);
    mnCloseMPs = 0;

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    // 只在第一帧调用
    if(mbInitialComputations)
    {
        // 图像畸变校正之后的边界，数值是相对原始图像坐标系的
        ComputeImageBounds(imGray);

        // 网格尺寸的逆
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        // 相机内参
        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    // 双目基线
    mb = mbf/fx;

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mTlr = cv::Mat(3,4,CV_32F);
    mTrl = cv::Mat(3,4,CV_32F);
    mvStereo3Dpoints = vector<cv::Mat>(0);
    monoLeft = -1;
    monoRight = -1;

    // 将所有特征点分配到网格
    AssignFeaturesToGrid();

    // mVw = cv::Mat::zeros(3,1,CV_32F);
    // 初始化imu速度
    if(pPrevF)
    {
        if(!pPrevF->mVw.empty())
            mVw = pPrevF->mVw.clone();
    }
    else
    {
        mVw = cv::Mat::zeros(3,1,CV_32F);
    }

    mpMutexImu = new std::mutex();
}

/**
 * 将所有特征点分配到网格
*/
void Frame::AssignFeaturesToGrid()
{
    // Fill matrix with points
    const int nCells = FRAME_GRID_COLS*FRAME_GRID_ROWS;

    // 总共N个点，每个网格提前分配1/2最大值的空间
    int nReserve = 0.5f*N/(nCells);

    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
            mGrid[i][j].reserve(nReserve);
            if(Nleft != -1){
                mGridRight[i][j].reserve(nReserve);
            }
        }


    // 分配到网格
    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
                                                 : (i < Nleft) ? mvKeys[i]
                                                                 : mvKeysRight[i - Nleft];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY)){
            if(Nleft == -1 || i < Nleft)
                mGrid[nGridPosX][nGridPosY].push_back(i);
            else
                mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
        }
    }
}

/**
 * 提取orb特征点、描述子
 * 存mvKeys、mDescriptors；mvKeysRight、mDescriptorsRight
*/
void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1)
{
    vector<int> vLapping = {x0,x1};
    if(flag==0)
        monoLeft = (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors,vLapping);
    else
        monoRight = (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight,vLapping);
}

/**
 * 设置相机位姿
*/
void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    // T计算R、t、O
    UpdatePoseMatrices();
}

/**
 * 相机位姿
*/
void Frame::GetPose(cv::Mat &Tcw)
{
    Tcw = mTcw.clone();
}

/**
 * 设置新的IMU偏置
*/
void Frame::SetNewBias(const IMU::Bias &b)
{
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

/**
 * 设置imu速度
*/
void Frame::SetVelocity(const cv::Mat &Vwb)
{
    mVw = Vwb.clone();
}

/**
 * 设置imu速度、位姿
*/
void Frame::SetImuPoseVelocity(const cv::Mat &Rwb, const cv::Mat &twb, const cv::Mat &Vwb)
{
    mVw = Vwb.clone();
    cv::Mat Rbw = Rwb.t();
    cv::Mat tbw = -Rbw*twb;
    cv::Mat Tbw = cv::Mat::eye(4,4,CV_32F);
    Rbw.copyTo(Tbw.rowRange(0,3).colRange(0,3));
    tbw.copyTo(Tbw.rowRange(0,3).col(3));
    mTcw = mImuCalib.Tcb*Tbw;
    UpdatePoseMatrices();
}

/**
 * 相机T计算R、t、O
*/
void Frame::UpdatePoseMatrices()
{
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

/**
 * imu的世界坐标
*/
cv::Mat Frame::GetImuPosition()
{
    return mRwc*mImuCalib.Tcb.rowRange(0,3).col(3)+mOw;
}

/**
 * imu旋转
*/
cv::Mat Frame::GetImuRotation()
{
    return mRwc*mImuCalib.Tcb.rowRange(0,3).colRange(0,3);
}

/**
 * imu位姿
*/
cv::Mat Frame::GetImuPose()
{
    cv::Mat Twb = cv::Mat::eye(4,4,CV_32F);
    Twb.rowRange(0,3).colRange(0,3) = mRwc*mImuCalib.Tcb.rowRange(0,3).colRange(0,3);
    Twb.rowRange(0,3).col(3) = mRwc*mImuCalib.Tcb.rowRange(0,3).col(3)+mOw;
    return Twb.clone();
}

/**
 * MP投影到当前帧
 * 1、检查深度值大于0
 * 2、检查像素在图像范围内
 * 3、MP与相机距离在最大最小值范围内
 * 4、视差不能太大
*/
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    if(Nleft == -1){
        // cout << "\na";
        pMP->mbTrackInView = false;
        pMP->mTrackProjX = -1;
        pMP->mTrackProjY = -1;

        // 3D in absolute coordinates
        cv::Mat P = pMP->GetWorldPos();

        // cout << "b";

        // 3D in camera coordinates
        const cv::Mat Pc = mRcw*P+mtcw;
        // MP投影到当前相机，与光心距离
        const float Pc_dist = cv::norm(Pc);

        // Check positive depth
        // 深度值
        const float &PcZ = Pc.at<float>(2);
        const float invz = 1.0f/PcZ;
        if(PcZ<0.0f)
            return false;

        const cv::Point2f uv = mpCamera->project(Pc);

        // cout << "c";

        if(uv.x<mnMinX || uv.x>mnMaxX)
            return false;
        if(uv.y<mnMinY || uv.y>mnMaxY)
            return false;

        // cout << "d";
        pMP->mTrackProjX = uv.x;
        pMP->mTrackProjY = uv.y;

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        // MP与相机的距离
        const cv::Mat PO = P-mOw;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            return false;

        // cout << "e";

        // Check viewing angle
        cv::Mat Pn = pMP->GetNormal();

        // cout << "f";

        // 视差
        const float viewCos = PO.dot(Pn)/dist;

        if(viewCos<viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist,this);

        // cout << "g";

        // Data used by the tracking
        pMP->mbTrackInView = true;
        // MP投影到当前帧的像素坐标x
        pMP->mTrackProjX = uv.x;
        pMP->mTrackProjXR = uv.x - mbf*invz;
        // MP投影到当前帧，与相机的距离
        pMP->mTrackDepth = Pc_dist;
        // cout << "h";

        // MP投影到当前帧的像素坐标y
        pMP->mTrackProjY = uv.y;
        // 预测MP在当前帧中匹配2d点的金字塔层级
        pMP->mnTrackScaleLevel= nPredictedLevel;
        // 视差
        pMP->mTrackViewCos = viewCos;

        // cout << "i";

        return true;
    }
    else{
        pMP->mbTrackInView = false;
        pMP->mbTrackInViewR = false;
        pMP -> mnTrackScaleLevel = -1;
        pMP -> mnTrackScaleLevelR = -1;

        pMP->mbTrackInView = isInFrustumChecks(pMP,viewingCosLimit);
        pMP->mbTrackInViewR = isInFrustumChecks(pMP,viewingCosLimit,true);

        return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
}

/**
 * MP投影，畸变校正得到kp
*/
bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    // cout << "c";

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float u_distort, v_distort;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if(mDistCoef.total() == 5)
    {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;


    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

/**
 * not-used
*/
cv::Mat Frame::inRefCoordinates(cv::Mat pCw)
{
    return mRcw*pCw+mtcw;
}

/**
 * 提取(x,y)位置处r范围矩形窗内的特征点
*/
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    // [效率]容器提前分配空间
    vector<size_t> vIndices;
    vIndices.reserve(N);

    // 搜索窗半径
    float factorX = r;
    float factorY = r;

    /*cout << "fX " << factorX << endl;
    cout << "fY " << factorY << endl;*/

    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
    {
        return vIndices;
    }

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    // 遍历搜索窗占据的网格
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]]
                                                         : (!bRight) ? mvKeys[vCell[j]]
                                                                     : mvKeysRight[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                // 保留搜索窗内的特征点
                if(fabs(distx)<factorX && fabs(disty)<factorY)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

/**
 * 计算特征点所在网格位置
*/
bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

/**
 * 描述子计算Bow
*/
void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        // 所有特征点的描述子
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // 转换成词向量
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

/**
 * 特征点畸变校正
*/
void Frame::UndistortKeyPoints()
{
    // 没有畸变
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);

    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    // 像素点畸变校正
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat, static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);


    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }

}

/**
 * 图像畸变校正之后的边界，数值是相对原始图像坐标系的
*/
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        // 四个顶点
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Undistort corners
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

/**
 * 双目匹配，在右目中寻找左目的匹配特征点
 * 行搜索 - 滑窗对比 - 亚像素修正
 * 1、遍历右目特征点，记录所在行（上下浮动一行），后面左目特征点在相同行找右目候选点
 * 2、遍历左目特征点，得到所在行的右目候选点集合；限制左右目的匹配点在相同行上
 * 3、在行上选取描述子距离最小的点，作为最佳候选点
 * 4、在左目点对应金字塔层级，取半径5的搜索窗；在最佳候选点周围也取半径5的搜索窗，另外搜索窗水平左右滑动5个像素，计算搜索窗相似度
 * 5、选择最佳匹配搜索窗，联合左右计算亚像素（小于1个像素）插值，最后确定右目匹配点的位置
 * 5、对所有左右目匹配点，剔除上面搜索窗匹配精度较低的匹配点
*/
void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    // 行数
    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    // 遍历右目特征点，记录所在行（上下浮动一行）
    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        // 理论上，左右目匹配点的y坐标是相同的，在同一行上，但是保不准有误差，上下偏移一行
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);
        // 右目特征点在该行上
        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    // 深度值不得小于基线距离
    const float minZ = mb;
    // ul-ur = mbf/z
    // z无限大的时候，ul=ur
    const float minD = 0;
    // ul-ur = mbf/z
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    // 遍历左目特征点
    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        // 该行对应的右目特征点集合
        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        // 根据双目模型，右目匹配点的x范围
        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        // 左目特征点描述子
        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        // 遍历候选右目匹配点
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            // 层级上下浮动一层
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;
            
            // 计算描述子距离
            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        // 最佳匹配点，描述子距离够小
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            // 右目特征点坐标x
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];

            // 左目特征点尺度缩放后的坐标
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            // 左目该特征点周围半径5的矩形窗
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            //IL.convertTo(IL,CV_32F);
            //IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);
            IL.convertTo(IL,CV_16S);
            // 减去中心点的像素值，归一化，降低亮度变化对相似度计算的影响
            IL = IL - IL.at<short>(w,w);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            // 右目对应匹配点的窗口，不同的是，这个窗口还要沿着x方向左右滑动L距离，下面两个是左右最远边界
            const float iniu = scaleduR0-L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            // 遍历右目匹配点位置处的滑动窗口，计算窗口相似度
            for(int incR=-L; incR<=+L; incR++)
            {
                // 右目匹配点也提取半径5的窗口，并且沿着水平方向左右滑动L距离
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                //IR.convertTo(IR,CV_32F);
                //IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);
                IR.convertTo(IR,CV_16S);
                IR = IR - IR.at<short>(w,w);

                // 计算窗口的像素差SAD
                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            // 最佳匹配窗相似度，前后的匹配窗相似度
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            // 亚像素修正
            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            // 最终的右目匹配点坐标，尺度还原
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            // 视差
            float disparity = (uL-bestuR);

            // 视差满足范围要求
            if(disparity>=minD && disparity<maxD)
            {
                // 视差为0，无穷远点
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                // 深度值
                mvDepth[iL]=mbf/disparity;
                // 右目特征点坐标
                mvuRight[iL] = bestuR;
                // 搜索窗匹配精度 - 左目点索引
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }
    // 特征点按匹配精度从高到低排序
    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    // 删除匹配精度较低的右目匹配点
    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}

/**
 * 深度图像提取特征点深度值
*/
void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

/**
 * 反投影特征点得到MP
*/
cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}


bool Frame::imuIsPreintegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    return mbImuPreintegrated;
}

void Frame::setIntegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    mbImuPreintegrated = true;
}

/**
 * 双目鱼眼相机构造函数
*/
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, cv::Mat& Tlr,Frame* pPrevF, const IMU::Calib &ImuCalib)
        :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2), mTlr(Tlr)
{
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    imgLeft = imLeft.clone();
    imgRight = imRight.clone();
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1]);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1]);
    threadLeft.join();
    threadRight.join();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    Nleft = mvKeys.size();
    Nright = mvKeysRight.size();
    // 双目特征点总数
    N = Nleft + Nright;

    if(N == 0)
        return;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf / fx;

    mRlr = mTlr.rowRange(0,3).colRange(0,3);
    mtlr = mTlr.col(3);

    cv::Mat Rrl = mTlr.rowRange(0,3).colRange(0,3).t();
    cv::Mat trl = Rrl * (-1 * mTlr.col(3));

    cv::hconcat(Rrl,trl,mTrl);

    // 鱼眼相机双目匹配
    ComputeStereoFishEyeMatches();
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    //Put all descriptors in the same matrix
    cv::vconcat(mDescriptors,mDescriptorsRight,mDescriptors);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(nullptr));
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();

    mpMutexImu = new std::mutex();

    UndistortKeyPoints();
    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

    double t_read = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t1 - t0).count();
    double t_orbextract = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
    double t_stereomatches = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t3 - t2).count();
    double t_assign = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t4 - t3).count();
    double t_undistort = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t5 - t4).count();

    /*cout << "Reading time: " << t_read << endl;
    cout << "Extraction time: " << t_orbextract << endl;
    cout << "Matching time: " << t_stereomatches << endl;
    cout << "Assignment time: " << t_assign << endl;
    cout << "Undistortion time: " << t_undistort << endl;*/

}

/**
 * 鱼眼相机双目匹配
*/
void Frame::ComputeStereoFishEyeMatches() {
    //Speed it up by matching keypoints in the lapping area
    vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
    vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());

    // 左目、右目描述子
    cv::Mat stereoDescLeft = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
    cv::Mat stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

    mvLeftToRightMatch = vector<int>(Nleft,-1);
    mvRightToLeftMatch = vector<int>(Nright,-1);
    mvDepth = vector<float>(Nleft,-1.0f);
    mvuRight = vector<float>(Nleft,-1);
    mvStereo3Dpoints = vector<cv::Mat>(Nleft);
    mnCloseMPs = 0;

    //Perform a brute force between Keypoint in the left and right image
    vector<vector<cv::DMatch>> matches;

    // 对每个左目描述子，在右目中找到2个候选最佳匹配
    BFmatcher.knnMatch(stereoDescLeft,stereoDescRight,matches,2);

    int nMatches = 0;
    int descMatches = 0;

    //Check matches using Lowe's ratio
    for(vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it){
        // 次佳与最佳拉开距离
        if((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7){
            //For every good match, check parallax and reprojection error to discard spurious matches
            cv::Mat p3D;
            descMatches++;
            float sigma1 = mvLevelSigma2[mvKeys[(*it)[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[mvKeysRight[(*it)[0].trainIdx + monoRight].octave];
            // 三角化
            float depth = static_cast<KannalaBrandt8*>(mpCamera)->TriangulateMatches(mpCamera2,mvKeys[(*it)[0].queryIdx + monoLeft],mvKeysRight[(*it)[0].trainIdx + monoRight],mRlr,mtlr,sigma1,sigma2,p3D);
            if(depth > 0.0001f){
                mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft] = (*it)[0].trainIdx + monoRight;
                mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
                mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft] = p3D.clone();
                mvDepth[(*it)[0].queryIdx + monoLeft] = depth;
                nMatches++;
            }
        }
    }
}

/**
 * MP投影到当前帧
 * 1、检查深度值大于0
 * 2、检查像素在图像范围内
 * 3、MP与相机距离在最大最小值范围内
 * 4、视差不能太大
*/
bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight) {
    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos();

    cv::Mat mR, mt, twc;
    if(bRight){
        cv::Mat Rrl = mTrl.colRange(0,3).rowRange(0,3);
        cv::Mat trl = mTrl.col(3);
        mR = Rrl * mRcw;
        mt = Rrl * mtcw + trl;
        twc = mRwc * mTlr.rowRange(0,3).col(3) + mOw;
    }
    else{
        mR = mRcw;
        mt = mtcw;
        twc = mOw;
    }

    // 3D in camera coordinates
    cv::Mat Pc = mR*P+mt;
    const float Pc_dist = cv::norm(Pc);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    cv::Point2f uv;
    if(bRight) uv = mpCamera2->project(Pc);
    else uv = mpCamera->project(Pc);

    if(uv.x<mnMinX || uv.x>mnMaxX)
        return false;
    if(uv.y<mnMinY || uv.y>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-twc;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    if(bRight){
        pMP->mTrackProjXR = uv.x;
        pMP->mTrackProjYR = uv.y;
        pMP->mnTrackScaleLevelR= nPredictedLevel;
        pMP->mTrackViewCosR = viewCos;
        pMP->mTrackDepthR = Pc_dist;
    }
    else{
        pMP->mTrackProjX = uv.x;
        pMP->mTrackProjY = uv.y;
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;
        pMP->mTrackDepth = Pc_dist;
    }

    return true;
}

/**
 * 鱼眼相机反投影得到MP
*/
cv::Mat Frame::UnprojectStereoFishEye(const int &i){
    return mRwc*mvStereo3Dpoints[i]+mOw;
}

} //namespace ORB_SLAM
