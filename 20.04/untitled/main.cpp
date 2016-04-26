#include <iostream>
#include <fstream>
#include <string>
#include <ctype.h>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/imgproc/imgproc.hpp"


using namespace std;
using namespace cv;
using namespace cv::detail;


int main(){

    srand (time(NULL));
    initModule_nonfree();
    vector<string> img_names;
    img_names.push_back("Y3.JPG");    img_names.push_back("Y2.JPG");  //   img_names.push_back("Y3.JPG");     img_names.push_back("Y4.JPG");

    //задаем количество и названия
    int num_images = img_names.size();

    //читаем
    vector<Mat> full_images(num_images);
    for (int i=0; i<num_images; ++i)
        full_images[i] = imread(img_names[i]);

    //ресайзим
    vector<Mat> images(num_images);
    for (int i=0; i<num_images; ++i)
        resize(full_images[i], images[i], Size(), 0.1, 0.1);

    //находим ключи
    vector<ImageFeatures> features(num_images);
    Ptr<FeaturesFinder> finder = new SurfFeaturesFinder(300, 3, 2);
    //уровень отсева, количество октав, колличество слоев,
    for (int i=0; i<num_images; ++i)
        (*finder)(images[i], features[i]);

    //находим соответствия
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);
    //gpu, порог принятия, минимальное кооличество соотнесений для 2D проективного прообразования на ???
    matcher(features, pairwise_matches);

//вывод ключей
//    for (int i=0; i<num_images; ++i)
//    {
//        Mat output;
//        drawKeypoints(images[i], features[i].keypoints, output );
//        imwrite("key"+to_string(i)+".jpg", output);
//    }

//вывод матчей
//    for (int i=0; i<pairwise_matches.size(); ++i)
//    {
//        if (pairwise_matches[i].src_img_idx >= 0)
//        {
//            Mat out;
//            drawMatches(images[pairwise_matches[i].src_img_idx], features[pairwise_matches[i].src_img_idx].keypoints, images[pairwise_matches[i].dst_img_idx], features[pairwise_matches[i].dst_img_idx].keypoints, pairwise_matches[i].matches, out);
//            imwrite("match"+to_string(i)+".jpg", out);
//        }
//    }


    for (int i=0; i<pairwise_matches.size(); ++i)
    {
        if (pairwise_matches[i].matches.size() >=  4)
        {
            int src_img_idx = pairwise_matches[i].src_img_idx;
            int dst_img_idx = pairwise_matches[i].dst_img_idx;
            Mat mask_src(images[src_img_idx].size(), CV_8UC1, cv::Scalar(0));
            Mat mask_dst(images[dst_img_idx].size(), CV_8UC1, cv::Scalar(0));
            vector<Point> points_src;
            vector<Point> points_dst;
            //нахождение 4 случайных
            for (int j = 0; j<4; ++j)
            {
                int number = rand() % pairwise_matches[i].matches.size();
                points_src.push_back(features[src_img_idx].keypoints[pairwise_matches[i].matches[number].queryIdx].pt);
                points_dst.push_back(features[dst_img_idx].keypoints[pairwise_matches[i].matches[number].trainIdx].pt);
            }
            //построение маски
            vector<vector<Point>> contours_src = {points_src};
            vector<vector<Point>> contours_dst = {points_dst};

            drawContours(mask_src, contours_src, 0, cv::Scalar(255), CV_FILLED);
            drawContours(mask_dst, contours_dst, 0, cv::Scalar(255), CV_FILLED);

            Mat src_out;
            images[src_img_idx].copyTo(src_out, mask_src);
            imwrite(to_string(i)+"src.jpg",src_out);

            Mat dst_out;
            images[dst_img_idx].copyTo(dst_out, mask_dst);

            //приведение к необходимому виду
            Point2f contours_dst1[points_src.size()];
            Point2f contours_src1[points_src.size()];
            for (int j=0; j<points_src.size(); ++j)
            {
                contours_dst1[j]=Point2f(points_dst[j].x, points_dst[j].y );
                contours_src1[j]=Point2f(points_src[j].x, points_src[j].y );
            }
            //преобразование
            Mat H =  getPerspectiveTransform(contours_dst1, contours_src1);
            warpPerspective(dst_out, dst_out, H, Size());
            imwrite(to_string(i)+"dsta.jpg",dst_out);

            imwrite(to_string(i)+"de.jpg", dst_out-src_out);
            std::cout << norm(dst_out-src_out)<<std::endl;
//            Mat src1;
//            bitwise_not(mask,mask);
//            images[i].copyTo(src1,mask);
//            imwrite("src1.jpg",src1);

        }
    }


    waitKey();
}

