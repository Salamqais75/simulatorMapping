#include <thread>
#include <future>
#include <queue>

#include <pangolin/pangolin.h>
#include <pangolin/geometry/geometry.h>
#include <pangolin/gl/glsl.h>
#include <pangolin/gl/glvbo.h>

#include <pangolin/utils/file_utils.h>
#include <pangolin/geometry/glgeometry.h>

#include "include/run_model/TextureShader.h"
#include "include/Auxiliary.h"

#include "ORBextractor.h"

#include <Eigen/SVD>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>



void  applyForwardToModelCam(pangolin::OpenGlRenderState& cam, double value) {
       auto camMatrix = pangolin::ToEigen<double>(cam.GetModelViewMatrix());
           camMatrix(2, 3) += value;
           cam.SetModelViewMatrix(camMatrix);
}

    int main()
    {


        std::string settingPath = Auxiliary::GetGeneralSettingsPath();
        std::ifstream programData(settingPath); // inpute file for reading the data from JSON file 
        nlohmann::json data;
        programData >> data;
        programData.close();   // Closing file read 



        std::string configPath = data["DroneYamlPathSlam"];//retrieves the path to another configuration file .
        cv::FileStorage fSettings(configPath, cv::FileStorage::READ);//opens the YAML configuration file for reading.

        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];  ///put the values from the DroneYamlPathSlam by using the read file
        float cy = fSettings["Camera.cy"];
        float viewpointX = fSettings["RunModel.ViewpointX"];
        float viewpointY = fSettings["RunModel.ViewpointY"];
        float viewpointZ = fSettings["RunModel.ViewpointZ"];

        Eigen::Matrix3d K; // declares an Eigen matrix (K) to store the camera intrinsic matrix.

        K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;  //t initializes the intrinsic matrix K using the extracted fx, fy, cx, and cy values.
        cv::Mat K_cv = (cv::Mat_<float>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0); //This line creates a corresponding OpenCV matrix (K_cv) using the same camera intrinsic parameters.

        cv::Mat img; //declares an OpenCV matrix (img) to store images captured from the OpenGL buffer.

        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        ORB_SLAM2::ORBextractor* orbExtractor = new ORB_SLAM2::ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST); //creates a new instance of the ORB_SLAM2::ORBextractor class , 

        // Options
        bool show_bounds = false; // show bounding boxes around the rendered geometry.
        bool show_axis = false;// show the axis coordinate system (X, Y, Z axes) on the 
        bool show_x0 = false;//  show the planes representing the X=0
        bool show_y0 = false;// show the planes representing the y=0
        bool show_z0 = false;//  show the planes representing the z=0
        bool cull_backfaces = false;// enable backface culling. Backface culling is a technique used in 3D rendering to improve performance by not rendering triangles facing away from the camera.

        // Create Window for rendering
        pangolin::CreateWindowAndBind("Main", 640, 480);
        glEnable(GL_DEPTH_TEST);

        // Define Projection and initial ModelView matrix
        pangolin::OpenGlRenderState cam(
            pangolin::ProjectionMatrix(640, 480, K(0, 0), K(1, 1), K(0, 2), K(1, 2), NEAR_PLANE, FAR_PLANE),
            pangolin::ModelViewLookAt(viewpointX, viewpointY, viewpointZ, 0, 0, 0, 0.0, -1.0, pangolin::AxisY)
        );


        double nagvation_speed = 0.1;
        
        Eigen::Vector3d wanted_point(1, 1, 1);


        while (!pangolin::ShouldQuit()) {
            // Clear the screen and activate the camera view
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            cam.Activate();

            // Move the camera towards the wanted_point
            Eigen::Vector3d camera_position = cam.GetModelViewMatrix().topRightCorner(3, 1);

            Eigen::Vector3d direction = (wanted_point - camera_position).normalized();
            double distance_to_target = (wanted_point - camera_position).norm();

            // If the camera hasn't reached the wanted_point, move it forward along the direction vector
            if (distance_to_target > nagvation_speed) {
                applyForwardToModelCam(cam, nagvation_speed * direction.z());
            }
            else {
                // The camera has reached the wanted_point
                std::cout << "Camera reached the wanted_point: " << wanted_point.transpose() << std::endl;
                break;
            }

            // Update the Pangolin display
            pangolin::FinishFrame();
        }
    }