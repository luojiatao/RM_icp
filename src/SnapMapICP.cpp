#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <sensor_msgs/LaserScan.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <laser_geometry/laser_geometry.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <boost/thread/mutex.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

'''
定义了一个名为 scan_callback_mutex 的互斥锁mutex,它属于 Boost 库中的 boost::mutex 类型。
互斥锁是一种线程同步机制，可以确保同一时间只有一个线程能够访问共享资源，避免多个线程同时修改一个共享资源而导致的数据竞争问题。
在这个例子中,scan_callback_mutex 的作用是确保在扫描回调函数运行时，不会有其他线程同时对回调函数进行修改。
'''
boost::mutex scan_callback_mutex;

//ICP算法匹配的最大残差阈值，如果匹配残差超过这个阈值，则认为匹配失败。
double ICP_FITNESS_THRESHOLD = 100.1;// =  0.025;
//激光雷达匹配时两个点之间的距离阈值，如果两个点的距离超过这个阈值，则认为它们不匹配。
double DIST_THRESHOLD = 0.05;
//激光雷达匹配时两个点之间的角度差阈值，如果两个点之间的角度差超过这个阈值，则认为它们不匹配。
double ANGLE_THRESHOLD = 0.01;
//激光雷达匹配时两个点之间的最大角度差阈值，如果两个点之间的角度差超过这个阈值，则认为它们不匹配。
double ANGLE_UPPER_THRESHOLD = M_PI / 6;
//一个点云数据在地图中的最大存活时间，如果超过这个时间，则认为它不再有效。
double AGE_THRESHOLD = 1;
//地图中的一个点云数据在更新之前最小需要保持的时间，如果不满足这个条件，则不会更新它。
double UPDATE_AGE_THRESHOLD = 1;
//ICP算法中的内点比例阈值，如果内点比例低于这个阈值，则认为匹配失败。
double ICP_INLIER_THRESHOLD = 0.9;
//ICP算法中用于计算内点的距离阈值，如果一个点的距离大于这个阈值，则不会将它作为内点。
double ICP_INLIER_DIST = 0.1;
//位姿估计协方差矩阵中位置分量的系数。
double POSE_COVARIANCE_TRANS = 1.5;
//ICP算法的最大迭代次数。
double ICP_NUM_ITER = 250;
//激光雷达的扫描频率。
double SCAN_RATE = 2;

std::string BASE_LASER_FRAME = "/hokuyo_laser_link";
std::string ODOM_FRAME = "/odom";
//指向ros::NodeHandle对象的指针，用于与ROS系统进行通信。
ros::NodeHandle *nh = 0;
//用于发布处理后的点云数据。
ros::Publisher pub_output_;
//用于发布处理后的激光扫描数据。
ros::Publisher pub_output_scan;
//用于发布经过坐标变换后的激光扫描数据。
ros::Publisher pub_output_scan_transformed;
//用于发布与激光雷达配准相关的信息。
ros::Publisher pub_info_;

'''
laser_geometry::LaserProjection 是一个 ROS package 中的类，用于将激光雷达数据转换成点云数据。
projector_ 是一个指向 LaserProjection 对象的指针，它被用来调用 projectLaser() 方法将激光雷达数据转换成点云数据。
'''
laser_geometry::LaserProjection *projector_ = 0;
//tf::TransformListener 是一个用于监听和存储在ROS中广泛使用的坐标变换的类,监听 ROS 中两个坐标系之间的变换
tf::TransformListener *listener_ = 0;
//cloud2存储的是原始的点云数据，cloud2transformed存储的是经过坐标变换后的点云数据。
sensor_msgs::PointCloud2 cloud2;
sensor_msgs::PointCloud2 cloud2transformed;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudT;

//output_cloud 保存的是转换后的点云数据，它将被发布到 /velodyne_points 话题上，而 scan_cloud 保存的是激光雷达扫描数据，它将被发布到 /scan 话题上
boost::shared_ptr< sensor_msgs::PointCloud2> output_cloud = boost::shared_ptr<sensor_msgs::PointCloud2>(new sensor_msgs::PointCloud2());
boost::shared_ptr< sensor_msgs::PointCloud2> scan_cloud = boost::shared_ptr<sensor_msgs::PointCloud2>(new sensor_msgs::PointCloud2());

bool we_have_a_map = false;
bool we_have_a_scan = false;
bool we_have_a_scan_transformed = false;
bool use_sim_time = true;
//lastScan 记录了上一次接收到的扫描数据的序号，actScan 记录当前接收到的扫描数据的序号。
int lastScan = 0;
int actScan = 0;

// 定义了一个名为transform_map_baselink的tf::Transform类型变量，用于存储从地图坐标系到机器人base_link坐标系的变换。
tf::Transform transform_map_baselink;

'''
inline void是一种函数定义方式,使用inline关键字可以让函数在被调用时直接将函数体嵌入到调用的位置,
从而减少函数调用的开销。这种函数定义方式适用于函数体较小的函数，通常用于需要频繁调用的函数，以提高程序的执行效率。
'''
'''
bt 是 tf::Transform 类型,代表了一个旋转和平移的变换矩阵。虽然它也包含了4x4的矩阵,但它的内部结构不同于Eigen::Matrix4f,
而是由旋转矩阵和平移向量组成。通过这个函数将一个Eigen::Matrix4f类型的矩阵转换为tf::Transform类型的变换,
'''
inline void
matrixAsTransfrom (const Eigen::Matrix4f &out_mat,  tf::Transform& bt)
{
    double mv[12];

    mv[0] = out_mat (0, 0) ;
    mv[4] = out_mat (0, 1);
    mv[8] = out_mat (0, 2);
    mv[1] = out_mat (1, 0) ;
    mv[5] = out_mat (1, 1);
    mv[9] = out_mat (1, 2);
    mv[2] = out_mat (2, 0) ;
    mv[6] = out_mat (2, 1);
    mv[10] = out_mat (2, 2);

    tf::Matrix3x3 basis;
    basis.setFromOpenGLSubMatrix(mv);
    tf::Vector3 origin(out_mat (0, 3),out_mat (1, 3),out_mat (2, 3));

    ROS_DEBUG("origin %f %f %f", origin.x(), origin.y(), origin.z());

    bt = tf::Transform(basis,origin);
}


boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > cloud_xyz;
//pcl::KdTree<pcl::PointXYZ> 是一个用于查找点云中最近邻点的数据结构,mapTree是一个指向pcl::KdTree<pcl::PointXYZ>类型对象的指针。
pcl::KdTree<pcl::PointXYZ>::Ptr mapTree;

//函数返回指向tree的智能指针，这样可以将其传递给其他需要进行空间查询和距离查询的函数。
pcl::KdTree<pcl::PointXYZ>::Ptr getTree(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudb)
{
    pcl::KdTree<pcl::PointXYZ>::Ptr tree;
    tree.reset (new pcl::KdTreeFLANN<pcl::PointXYZ>);

    tree->setInputCloud (cloudb);
    return tree;
}

'''
ROS的回调函数,用于接收一个名为msg的占用栅格地图,并将地图中的障碍物转换成点云数据结构(PCL类型)进行处理。
依次遍历地图中的每一个像素,若该像素值为100(表示该处为障碍物)，则计算该像素对应的点在地图中的位置，
并将该点添加到cloud_xyz点云数据结构中(PCL)。最后，通过调用getTree函数生成点云的KDTree结构，
用于加速点云数据的空间查询和距离查询。最终，该函数将cloud_xyz转换成ROS格式的消息，发布到名为output_cloud的话题中。
'''
void mapCallback(const nav_msgs::OccupancyGrid& msg)
{
    ROS_INFO("I heard frame_id: [%s]", msg.header.frame_id.c_str());

    //msg.info包含了地图的一些基本信息，如分辨率、宽度、高度等，而msg.info.origin则包含了地图的原点位置信息，是一个geometry_msgs::Pose类型的消息。
    float resolution = msg.info.resolution;
    float width = msg.info.width;
    float height = msg.info.height;
    float posx = msg.info.origin.position.x;
    float posy = msg.info.origin.position.y;

    cloud_xyz = boost::shared_ptr< pcl::PointCloud<pcl::PointXYZ> >(new pcl::PointCloud<pcl::PointXYZ>());

    //height设置为1是因为这是一个无序点云，不需要考虑高度（height）这一维度。
    //is_dense设置为false表示这个点云中可能有缺失的点，即有些点的数据为NaN或者INF，而不是所有点都有完整的数据。
    cloud_xyz->height   = 1;
    cloud_xyz->is_dense = false;

    std_msgs::Header header;
    header.stamp = ros::Time(0.0);
    header.frame_id = "map";

    cloud_xyz->header = pcl_conversions::toPCL(header);

    pcl::PointXYZ point_xyz;

    //for (unsigned int i = 0; i < cloud_xyz->width ; i++)
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            //@TODO
            if (msg.data[x + y * width] == 100)
            {
                point_xyz.x = (.5f + x) * resolution + posx;
                point_xyz.y = (.5f + y) * resolution + posy;
                point_xyz.z = 0;
                //将当前循环中满足条件的点添加到点云数据
                cloud_xyz->points.push_back(point_xyz);
            }
        }
    cloud_xyz->width = cloud_xyz->points.size();

    mapTree = getTree(cloud_xyz);

    //将PointCloud(pcl::PointXYZ类型)的cloud_xyz点云数据转换为ROS中的PointCloud2类型，存储到output_cloud中。
    pcl::toROSMsg (*cloud_xyz, *output_cloud);
    ROS_INFO("Publishing PointXYZ cloud with %ld points in frame %s", cloud_xyz->points.size(),output_cloud->header.frame_id.c_str());

    we_have_a_map = true;
}


int lastTimeSent = -1000;

int count_sc_ = 0;

bool getTransform(tf::StampedTransform &trans , const std::string parent_frame, const std::string child_frame, const ros::Time stamp)
{
    bool gotTransform = false;

    ros::Time before = ros::Time::now();
    if (!listener_->waitForTransform(parent_frame, child_frame, stamp, ros::Duration(0.5)))
    {
        ROS_WARN("DIDNT GET TRANSFORM %s %s IN c at %f", parent_frame.c_str(), child_frame.c_str(), stamp.toSec());
        return false;
    }
    //ROS_INFO("waited for transform %f", (ros::Time::now() - before).toSec());

    try
    {
        gotTransform = true;
        listener_->lookupTransform(parent_frame,child_frame,stamp , trans);
    }
    catch (tf::TransformException ex)
    {
        gotTransform = false;
        ROS_WARN("DIDNT GET TRANSFORM %s %s IN B", parent_frame.c_str(), child_frame.c_str());
    }

    return gotTransform;
}

void initialpose_callback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg){
    ROS_INFO("initialpose : %f",msg->pose.pose.position.x);

    //-----------------------------------
    tf::StampedTransform odom_to_base;
    if (!getTransform(odom_to_base, ODOM_FRAME, "base_link", ros::Time(0)))
    {
        ROS_WARN("Did not get base pose at now");
        scan_callback_mutex.unlock();

        return;
    }

    // 创建tf的广播器
    static tf::TransformBroadcaster br;
    // 转为tf数据，并广播
    transform_map_baselink.setOrigin( tf::Vector3(msg->pose.pose.position.x, msg->pose.pose.position.y, 0.0) );
    tf::Quaternion q;
    q.setRPY(0, 0, tf::getYaw(msg->pose.pose.orientation));
    transform_map_baselink.setRotation(q);

    
    transform_map_baselink = transform_map_baselink * odom_to_base.inverse();

    // 更新广播map与base_link坐标系之间的tf数据
    br.sendTransform(tf::StampedTransform(transform_map_baselink, ros::Time::now(), "map", ODOM_FRAME));

}
/// ////////////////////////////////////////////////////////////////
ros::Time last_processed_scan;

void scanCallback (const sensor_msgs::LaserScan::ConstPtr& scan_in)
{
    if (!we_have_a_map)
    {
        ROS_INFO("SnapMapICP waiting for map to be published");
        return;
    }

    ros::Time scan_in_time = scan_in->header.stamp;
    ros::Time time_received = ros::Time::now();

    if ( scan_in_time - last_processed_scan < ros::Duration(1.0f / SCAN_RATE) )
    {
        ROS_DEBUG("rejected scan, last %f , this %f", last_processed_scan.toSec() ,scan_in_time.toSec());
        return;
    }


    //projector_.transformLaserScanToPointCloud("base_link",*scan_in,cloud,listener_);
    if (!scan_callback_mutex.try_lock())
        return;

    ros::Duration scan_age = ros::Time::now() - scan_in_time;

    //check if we want to accept this scan, if its older than 1 sec, drop it
    if (!use_sim_time)
        if (scan_age.toSec() > AGE_THRESHOLD)
        {
            //ROS_WARN("SCAN SEEMS TOO OLD (%f seconds, %f threshold)", scan_age.toSec(), AGE_THRESHOLD);
            ROS_WARN("SCAN SEEMS TOO OLD (%f seconds, %f threshold) scan time: %f , now %f", scan_age.toSec(), AGE_THRESHOLD, scan_in_time.toSec(),ros::Time::now().toSec() );
            scan_callback_mutex.unlock();

            return;
        }

    count_sc_++;
    //ROS_INFO("count_sc %i MUTEX LOCKED", count_sc_);

    //if (count_sc_ > 10)
    //if (count_sc_ > 10)
    {
        count_sc_ = 0;

        // ------------------------
        tf::StampedTransform odom_to_base_before;
        if (!getTransform(odom_to_base_before, ODOM_FRAME, "base_link", scan_in_time))
        {
            ROS_WARN("Did not get base pose at laser scan time");
            scan_callback_mutex.unlock();

            return;
        }


        sensor_msgs::PointCloud cloud;
        sensor_msgs::PointCloud cloudInMap;

        projector_->projectLaser(*scan_in,cloud);

        we_have_a_scan = false;
        bool gotTransform = false;

        if (!listener_->waitForTransform("/map", cloud.header.frame_id, cloud.header.stamp, ros::Duration(0.05)))
        {
            scan_callback_mutex.unlock();
            ROS_WARN("SnapMapICP no map to cloud transform found MUTEX UNLOCKED");
            return;
        }

        if (!listener_->waitForTransform("/map", "/base_link", cloud.header.stamp, ros::Duration(0.05)))
        {
            scan_callback_mutex.unlock();
            ROS_WARN("SnapMapICP no map to base transform found MUTEX UNLOCKED");
            return;
        }


        while (!gotTransform && (ros::ok()))
        {
            try
            {
                gotTransform = true;
                listener_->transformPointCloud ("/map",cloud,cloudInMap);
            }
            catch (...)
            {
                gotTransform = false;
                ROS_WARN("DIDNT GET TRANSFORM IN A");
            }
        }

        for (size_t k =0; k < cloudInMap.points.size(); k++)
        {
            cloudInMap.points[k].z = 0;
        }


        gotTransform = false;
        // ---------------------------------------
        tf::StampedTransform oldPose;
        while (!gotTransform && (ros::ok()))
        {
            try
            {
                gotTransform = true;
                listener_->lookupTransform("map", "base_link",
                                           cloud.header.stamp , oldPose);
            }
            catch (tf::TransformException ex)
            {
                gotTransform = false;
                ROS_WARN("DIDNT GET TRANSFORM IN B");
            }
        }
        if (we_have_a_map && gotTransform)
        {
            sensor_msgs::convertPointCloudToPointCloud2(cloudInMap,cloud2);
            we_have_a_scan = true;

            actScan++;

            //pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> reg;
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> reg;
            reg.setTransformationEpsilon (1e-6);
            // Set the maximum distance between two correspondences (src<->tgt) to 10cm
            // Note: adjust this based on the size of your datasets
            reg.setMaxCorrespondenceDistance(5);
            reg.setMaximumIterations (ICP_NUM_ITER);
            // Set the point representation

            //ros::Time bef = ros::Time::now();

            PointCloudT::Ptr myMapCloud (new PointCloudT());
            PointCloudT::Ptr myScanCloud (new PointCloudT());

            pcl::fromROSMsg(*output_cloud,*myMapCloud);
            pcl::fromROSMsg(cloud2,*myScanCloud);

            reg.setInputSource(myScanCloud);
            reg.setInputTarget(myMapCloud);

            PointCloudT unused;
            int i = 0;

            reg.align (unused);

            Eigen::Matrix4f transf = reg.getFinalTransformation();
            std::cout << "transf : " << std::endl;
            std::cout << transf << std::endl;

            const Eigen::Matrix3f rotation_matrix = transf.block(0,0,3,3);  //从欧拉矩阵获取旋转矩阵
            std::cout << "rotation_matrix : " << std::endl;
            std::cout << rotation_matrix << std::endl;

            Eigen::Vector3f eulerAngle = rotation_matrix.eulerAngles(0,1,2);    //旋转矩阵转欧拉角
            eulerAngle(2) *= 0.1;
            std::cout << "eulerAngle : " << std::endl;
            std::cout << eulerAngle << std::endl;

            Eigen::Matrix3f rotation_matri_after;   //欧拉角转旋转矩阵
            rotation_matri_after = Eigen::AngleAxisf(eulerAngle[0], Eigen::Vector3f::UnitX()) *
                            Eigen::AngleAxisf(eulerAngle[1], Eigen::Vector3f::UnitY()) *
                            Eigen::AngleAxisf(eulerAngle[2], Eigen::Vector3f::UnitZ());
            std::cout << "rotation_matri_after =\n" << rotation_matri_after << std::endl; 


            transf.block(0,0,3,3) = rotation_matri_after;
            // transf(0,3) *= 0.2;
            // transf(1,3) *= 0.2;
            std::cout << "transf : " << std::endl;
            std::cout << transf << std::endl;



            tf::Transform t_result;     
            matrixAsTransfrom(transf,t_result);
            // std::cout << "t_result : " << std::endl;
            // std::cout << &t_result << std::endl;

            //ROS_ERROR("proc time %f", (ros::Time::now() - bef).toSec());

            we_have_a_scan_transformed = false;
            PointCloudT transformedCloud;
            pcl::transformPointCloud (*myScanCloud, transformedCloud, reg.getFinalTransformation());

            double inlier_perc = 0;
            {
                // count inliers
                std::vector<int> nn_indices (1);
                std::vector<float> nn_sqr_dists (1);

                size_t numinliers = 0;

                for (size_t k = 0; k < transformedCloud.points.size(); ++k )
                {
                    if (mapTree->radiusSearch (transformedCloud.points[k], ICP_INLIER_DIST, nn_indices,nn_sqr_dists, 1) != 0)
                        numinliers += 1;
                }
                if (transformedCloud.points.size() > 0)
                {
                    //ROS_INFO("Inliers in dist %f: %zu of %zu percentage %f (%f)", ICP_INLIER_DIST, numinliers, transformedCloud.points.size(), (double) numinliers / (double) transformedCloud.points.size(), ICP_INLIER_THRESHOLD);
                    inlier_perc = (double) numinliers / (double) transformedCloud.points.size();
                }
            }

            last_processed_scan = scan_in_time;

            pcl::toROSMsg (transformedCloud, cloud2transformed);
            we_have_a_scan_transformed = true;

            double dist = sqrt((t_result.getOrigin().x() * t_result.getOrigin().x()) + (t_result.getOrigin().y() * t_result.getOrigin().y()));
            double angleDist = t_result.getRotation().getAngle();
            tf::Vector3 rotAxis  = t_result.getRotation().getAxis();
            t_result =  t_result * oldPose;

            //-----------------------------------
            tf::StampedTransform odom_to_base_after;
            if (!getTransform(odom_to_base_after, ODOM_FRAME, "base_link", ros::Time(0)))
            {
                ROS_WARN("Did not get base pose at now");
                scan_callback_mutex.unlock();

                return;
            }
            else
            {
                tf::Transform rel = odom_to_base_before.inverseTimes(odom_to_base_after);
                ROS_DEBUG("relative motion of robot while doing icp: %fcm %fdeg", rel.getOrigin().length(), rel.getRotation().getAngle() * 180 / M_PI);
                t_result= t_result * rel;
                t_result = t_result * odom_to_base_after.inverse();
        
            }
            //ROS_INFO("dist %f angleDist %f",dist, angleDist);

            //ROS_INFO("SCAN_AGE seems to be %f", scan_age.toSec());
            char msg_c_str[2048];
            sprintf(msg_c_str,"INLIERS %f (%f) scan_age %f (%f age_threshold) dist %f angleDist %f axis(%f %f %f) fitting %f (icp_fitness_threshold %f)",inlier_perc, ICP_INLIER_THRESHOLD, scan_age.toSec(), AGE_THRESHOLD ,dist, angleDist, rotAxis.x(), rotAxis.y(), rotAxis.z(),reg.getFitnessScore(), ICP_FITNESS_THRESHOLD );
            std_msgs::String strmsg;
            strmsg.data = msg_c_str;

            //ROS_INFO("%s", msg_c_str);

            double cov = POSE_COVARIANCE_TRANS;

            //if ((actScan - lastTimeSent > UPDATE_AGE_THRESHOLD) && ((dist > DIST_THRESHOLD) || (angleDist > ANGLE_THRESHOLD)) && (angleDist < ANGLE_UPPER_THRESHOLD))
            //  if ( reg.getFitnessScore()  <= ICP_FITNESS_THRESHOLD )
	    //	    std::cerr << "actScan - lastTimeSent: " << actScan - lastTimeSent << " " << "dist: " << dist << " " << "angleDist: " << angleDist << " inlier_perc: " << inlier_perc << std::endl;
            if ((actScan - lastTimeSent > UPDATE_AGE_THRESHOLD) && ((dist > DIST_THRESHOLD) || (angleDist > ANGLE_THRESHOLD)) && (inlier_perc > ICP_INLIER_THRESHOLD) && (angleDist < ANGLE_UPPER_THRESHOLD))
            {
                // 创建tf的广播器
                static tf::TransformBroadcaster br;

                // 转为tf数据，并广播
                transform_map_baselink.setOrigin( tf::Vector3(t_result.getOrigin().x(), t_result.getOrigin().y(), 0.0) );
                tf::Quaternion q;
                q.setRPY(0, 0, tf::getYaw(t_result.getRotation()));
                transform_map_baselink.setRotation(q);
                ROS_INFO("update pose!!!!");
                ROS_INFO("i %i converged %i score: %f", i,  reg.hasConverged (),  reg.getFitnessScore());
                ROS_INFO("publish new pose: dist:%f angleDist:%f pose(x,y):(%.3f,%.3f)",dist, angleDist, t_result.getOrigin().x(), t_result.getOrigin().y());


                // 更新广播map与base_link坐标系之间的tf数据
                br.sendTransform(tf::StampedTransform(transform_map_baselink, ros::Time::now(), "map", ODOM_FRAME));
                ROS_INFO("processing time : %f", (ros::Time::now() - time_received).toSec());
            }

            // ROS_INFO("processing time : %f", (ros::Time::now() - time_received).toSec());

            pub_info_.publish(strmsg);
        }
    }
    scan_callback_mutex.unlock();
}


ros::Time paramsWereUpdated;


void updateParams()
{
    paramsWereUpdated = ros::Time::now();
    // nh.param<std::string>("default_param", default_param, "default_value");
    nh->param<bool>("/snap_map_icp/USE_SIM_TIME", use_sim_time, true);
    nh->param<double>("/snap_map_icp/icp_fitness_threshold", ICP_FITNESS_THRESHOLD, 100 );
    nh->param<double>("/snap_map_icp/age_threshold", AGE_THRESHOLD, 1);   //scan与匹配的最大时间间隔
    nh->param<double>("/snap_map_icp/angle_upper_threshold", ANGLE_UPPER_THRESHOLD, 1);   //最大变换角度
    nh->param<double>("/snap_map_icp/angle_threshold", ANGLE_THRESHOLD, 0.01);    //最小变换角度
    nh->param<double>("/snap_map_icp/update_age_threshold", UPDATE_AGE_THRESHOLD, 1); //更新间隔
    nh->param<double>("/snap_map_icp/dist_threshold", DIST_THRESHOLD, 0.01);  //最小变换距离
    nh->param<double>("/snap_map_icp/icp_inlier_threshold", ICP_INLIER_THRESHOLD, 0.6);  //匹配的百分比，大于此百分比修正定位
    nh->param<double>("/snap_map_icp/icp_inlier_dist", ICP_INLIER_DIST, 0.1);
    nh->param<double>("/snap_map_icp/icp_num_iter", ICP_NUM_ITER, 250);   //ICP中的迭代次数
    nh->param<double>("/snap_map_icp/pose_covariance_trans", POSE_COVARIANCE_TRANS, 0.5); //平移姿态协方差，初始姿态被发送
    nh->param<double>("/snap_map_icp/scan_rate", SCAN_RATE, 2);
    if (SCAN_RATE < .001)
        SCAN_RATE  = .001;
    //ROS_INFO("PARAM UPDATE");
    // std::cerr << "ICP_INLIER_THRESHOLD: " << ICP_INLIER_THRESHOLD << std::endl;
}


int main(int argc, char** argv)
{

// Init the ROS node
    ros::init(argc, argv, "snapmapicp");
    ros::NodeHandle nh_;
    nh = &nh_;

    nh->param<std::string>("/snap_map_icp/odom_frame", ODOM_FRAME, "/odom");
    nh->param<std::string>("/snap_map_icp/base_laser_frame", BASE_LASER_FRAME, "/hokuyo_laser_link");

    last_processed_scan = ros::Time::now();

    projector_ = new laser_geometry::LaserProjection();
    tf::TransformListener listener;
    listener_ = &listener;

    pub_info_ =  nh->advertise<std_msgs::String> ("SnapMapICP", 1);
    pub_output_ = nh->advertise<sensor_msgs::PointCloud2> ("map_points", 1);
    pub_output_scan = nh->advertise<sensor_msgs::PointCloud2> ("scan_points", 1);
    pub_output_scan_transformed = nh->advertise<sensor_msgs::PointCloud2> ("scan_points_transformed", 1);
    // pub_pose = nh->advertise<geometry_msgs::PoseWithCovarianceStamped>("initialpose", 1);

    ros::Subscriber subMap = nh_.subscribe("map", 1, mapCallback);
    ros::Subscriber subScan = nh_.subscribe("scan", 1, scanCallback);
    ros::Subscriber subInitialpose = nh_.subscribe("initialpose",10,initialpose_callback);

    ros::Rate loop_rate(40);


    // listener_->waitForTransform("/base_link", "/map",
    //                             ros::Time(0), ros::Duration(30.0));

    // listener_->waitForTransform(BASE_LASER_FRAME, "/map",
    //                             ros::Time(0), ros::Duration(30.0));

    ros::AsyncSpinner spinner(1);
    spinner.start();

    updateParams();

    ROS_INFO("SnapMapICP running......");
    // 初始化坐标点
    transform_map_baselink.setOrigin( tf::Vector3(0.0, 0.0, 0.0) );
    tf::Quaternion q;
    q.setRPY(0, 0, 1.57);
    transform_map_baselink.setRotation(q);


    while (ros::ok())
    {
        //-----------------------------------------
        // 创建tf的广播器
        static tf::TransformBroadcaster br;

        // 广播map与base_link坐标系之间的tf数据
        br.sendTransform(tf::StampedTransform(transform_map_baselink, ros::Time::now(), "map", ODOM_FRAME));
        //法ROS_INFO("send tranform from map to base_link......");
        //-------------------------------------------

        if (actScan > lastScan)
        {
            lastScan = actScan;
            // publish map as a pointcloud2
            if (we_have_a_map)
              pub_output_.publish(output_cloud);
            // publish scan as seen as a pointcloud2
            if (we_have_a_scan)
               pub_output_scan.publish(cloud2);
            // publish icp transformed scan
            if (we_have_a_scan_transformed)
                pub_output_scan_transformed.publish(cloud2transformed);
        }
        loop_rate.sleep();
        ros::spinOnce();

        if (ros::Time::now() - paramsWereUpdated > ros::Duration(1))
            updateParams();
    }

}
