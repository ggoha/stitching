#include <iostream>
#include <fstream>
#include <string>
#include <ctype.h>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
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


using namespace std;
using namespace cv;
using namespace cv::detail;


int main(){

    srand (time(NULL));
    initModule_nonfree();
    vector<string> img_names;
    img_names.push_back("IMG_1156.JPG");    img_names.push_back("IMG_1157.JPG");   img_names.push_back("IMG_1158.JPG");     img_names.push_back("IMG_1159.JPG");
    //задаем количество и названия
    int num_images = img_names.size();

    //читаем
    vector<Mat> full_images(num_images);
    for (int i=0; i<num_images; ++i)
        full_images[i] = imread(img_names[i]);

    //ресайзим
    vector<Mat> images(num_images);
    for (int i=0; i<num_images; ++i)
    {
        resize(full_images[i], images[i],Size(), 0.5, 0.5);
    }

    //находим ключи
    vector<ImageFeatures> features(num_images);
    Ptr<FeaturesFinder> finder = new SurfFeaturesFinder(100, 3, 2);
    //уровень отсева, количество октав, колличество слоев,
    for (int i=0; i<num_images; ++i)
        (*finder)(full_images[i], features[i]);
    finder->collectGarbage();

    vector<Point2f> test;
    for (int j=0; j<100; ++j)
    {
        test.push_back(Point2f(3,3));
    }
    Mat result;
    kmeans(test, 2, result, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 1, 10.0), 5, KMEANS_PP_CENTERS);
    //    //находим соответствия
    //    vector<MatchesInfo> pairwise_matches;
    //    BestOf2NearestMatcher matcher(false, 0.4f, 6, 6);
    //    //gpu, порог принятия, минимальное кооличество соотнесений для 2D проективного прообразования на ???
    //    matcher(features, pairwise_matches);

    //    //вывод ключей
    //    for (int i=0; i<num_images; ++i)
    //    {
    //        Mat output;
    //        drawKeypoints(images[i], features[i].keypoints, output );
    //        imwrite("key"+to_string(i)+"_"+to_string(features[i].keypoints.size())+".jpg", output);
    //    }

    //    //вывод матчей
    //    for (int i=0; i<pairwise_matches.size(); ++i)
    //    {
    //        if (pairwise_matches[i].src_img_idx >= 0)
    //        {
    //            Mat out;
    //            drawMatches(images[pairwise_matches[i].src_img_idx], features[pairwise_matches[i].src_img_idx].keypoints, images[pairwise_matches[i].dst_img_idx], features[pairwise_matches[i].dst_img_idx].keypoints, pairwise_matches[i].matches, out);
    //            imwrite("match"+to_string(i)+".jpg", out);
    //        }
    //    }


    //    for (int i=0; i<pairwise_matches.size(); ++i)
    //    {
    //        int src_img_idx = pairwise_matches[i].src_img_idx;
    //        int dst_img_idx = pairwise_matches[i].dst_img_idx;
    //        if (pairwise_matches[i].src_img_idx >= 0)
    //        {
    //            bool flag = true;
    //            while (flag)
    //            {
    //                vector<Point2f> all_points_src, all_points_dst;
    //                for (int j=0; j<pairwise_matches[i].matches.size(); ++j)
    //                {
    //                    all_points_src.push_back(features[src_img_idx].keypoints[pairwise_matches[i].matches[j].queryIdx].pt);
    //                    all_points_dst.push_back(features[dst_img_idx].keypoints[pairwise_matches[i].matches[j].trainIdx].pt);
    //                }
    //                vector<vector<Point>> contours_src(1), contours_dst(1);
    //                vector<Point2f> _src, _dst;
    //                //выделяем 4 случайные
    //                for (int j = 0; j<4; ++j)
    //                {
    //                    int number = rand() % pairwise_matches[i].matches.size();
    //                    contours_src[0].push_back(features[src_img_idx].keypoints[pairwise_matches[i].matches[number].queryIdx].pt);
    //                    contours_dst[0].push_back(features[dst_img_idx].keypoints[pairwise_matches[i].matches[number].trainIdx].pt);
    //                    _src.push_back(features[src_img_idx].keypoints[pairwise_matches[i].matches[number].queryIdx].pt);
    //                    _dst.push_back(features[dst_img_idx].keypoints[pairwise_matches[i].matches[number].trainIdx].pt);
    //                }
    //                //построение маски
    //                Mat mask_src(images[src_img_idx].size(), CV_8UC1, cv::Scalar(0));
    //                Mat mask_dst(images[dst_img_idx].size(), CV_8UC1, cv::Scalar(0));

    //                drawContours(mask_src, contours_src, -1, cv::Scalar(255), CV_FILLED);
    //                drawContours(mask_dst, contours_dst, -1, cv::Scalar(255), CV_FILLED);

    //                Mat src_out;
    //                images[src_img_idx].copyTo(src_out, mask_src);

    //                Mat dst_out;
    //                images[dst_img_idx].copyTo(dst_out, mask_dst);

    //                Mat H = getPerspectiveTransform(_src, _dst);
    //                perspectiveTransform(all_points_src, all_points_src, H);
    //                warpPerspective(src_out, src_out, H, Size());
    //                Mat out = src_out-dst_out;
    //                int res=0, count = 0;

    //                for (int j=0; j<all_points_src.size(); ++j)
    //                {
    //                    if (pointPolygonTest(contours_dst[0], all_points_dst[j], false) > 0)
    //                    {
    //                        circle(out, all_points_dst[j], 3, Scalar(255,0,0));
    //                        circle(out, all_points_src[j], 3, Scalar(255,0,0));
    //                        line(out, all_points_dst[j], all_points_src[j], Scalar(0,0,255));
    //                        res += norm(all_points_src[j]-all_points_dst[j]);
    //                        count += 1;
    //                    }
    //                }

    //                if (count >= 4)
    //                {
    //                    double crit = min(sqrt(contourArea(contours_src[0])), sqrt(contourArea(contours_dst[0])));
    //                    std::cout<<i<<endl<<"point: "<<res<<endl<<"count: "<<count<<endl<<"res/count: "<<res/count<<endl<<"sqrt(s): "<<crit<<endl<<"crit1: "<<(res/count)/crit<<endl;
    //                    if (res/count < crit*0.1)
    //                    {
    //                        imwrite(to_string(i)+" "+to_string((res/count)/crit)+"d1.jpg",dst_out);
    //                        imwrite(to_string(i)+" "+to_string((res/count)/crit)+"s2.jpg",src_out);
    //                        imwrite(to_string(i)+" "+to_string((res/count)/crit)+"c.jpg", out);
    //    //                    std::cout <<"norm: "<< norm(src_out-dst_out)<<endl<<"s1: "<<contourArea(contours_src[0])<<endl<<"s2: "<<contourArea(contours_dst[0])<<endl<<"crit2: "<<norm(src_out-dst_out)/crit<<std::endl;
    //                        flag = false;
    //                   }
    //                   else
    //                    {
    //                        imwrite(to_string(i)+" "+to_string((res/count)/crit)+"d1.jpg",dst_out);
    //                        imwrite(to_string(i)+" "+to_string((res/count)/crit)+"s2.jpg",src_out);
    //                        imwrite(to_string(i)+" "+to_string((res/count)/crit)+"c.jpg", out);
    //    //                    std::cout <<"norm: "<< norm(src_out-dst_out)<<endl<<"s1: "<<contourArea(contours_src[0])<<endl<<"s2: "<<contourArea(contours_dst[0])<<endl<<"crit2: "<<norm(src_out-dst_out)/crit<<std::endl;
    //                   }
    //               }
    //            }
    //        }
    //    }
    //    waitKey();
}


  // #include <iostream>
  // #include <fstream>
  // #include <string>
  // #include <ctype.h>
  // #include "opencv2/opencv_modules.hpp"
  // #include "opencv2/highgui/highgui.hpp"
  // #include "opencv2/imgproc/imgproc.hpp"
  // #include "opencv2/calib3d/calib3d.hpp"
  // #include "opencv2/nonfree/nonfree.hpp"
  // #include "opencv2/nonfree/features2d.hpp"
  // #include "opencv2/stitching/detail/autocalib.hpp"
  // #include "opencv2/stitching/detail/blenders.hpp"
  // #include "opencv2/stitching/detail/camera.hpp"
  // #include "opencv2/stitching/detail/exposure_compensate.hpp"
  // #include "opencv2/stitching/detail/matchers.hpp"
  // #include "opencv2/stitching/detail/motion_estimators.hpp"
  // #include "opencv2/stitching/detail/seam_finders.hpp"
  // #include "opencv2/stitching/detail/util.hpp"
  // #include "opencv2/stitching/detail/warpers.hpp"
  // #include "opencv2/stitching/warpers.hpp"


  // using namespace std;
  // using namespace cv;
  // using namespace cv::detail;


  // int main(){

  //     Scalar colorTab[] =
  //     {
  //         Scalar(0, 0, 255),
  //         Scalar(0,255,0),
  //         Scalar(255,100,100),
  //         Scalar(255,0,255),
  //         Scalar(0,255,255)
  //     };

  //     srand (time(NULL));
  //     initModule_nonfree();
  //     vector<string> img_names;
  //     img_names.push_back("IMG_1156.JPG");    img_names.push_back("IMG_1157.JPG");   img_names.push_back("IMG_1158.JPG");     img_names.push_back("IMG_1159.JPG");
  //     //задаем количество и названия
  //     int num_images = img_names.size();

  //     //читаем
  //     vector<Mat> full_images(num_images);
  //     for (int i=0; i<num_images; ++i)
  //         full_images[i] = imread(img_names[i]);

  //     //ресайзим
  //     vector<Mat> images(num_images);
  //     for (int i=0; i<num_images; ++i)
  //     {
  //         resize(full_images[i], images[i],Size(), 0.5, 0.5);
  //     }

  //     //находим ключи
  //     vector<ImageFeatures> features(num_images);
  //     Ptr<FeaturesFinder> finder = new SurfFeaturesFinder(100, 3, 2);
  //     //уровень отсева, количество октав, колличество слоев,
  //     for (int i=0; i<num_images; ++i)
  //         (*finder)(full_images[i], features[i]);
  // //    finder->collectGarbage();

  //     //находим соответствия
  //     vector<MatchesInfo> pairwise_matches;
  //     BestOf2NearestMatcher matcher(false, 0.4f, 6, 6);
  //     //gpu, порог принятия, минимальное кооличество соотнесений для 2D проективного прообразования на ???
  //     matcher(features, pairwise_matches);

  //     //вывод ключей
  //     for (int i=0; i<num_images; ++i)
  //     {
  //         Mat output;
  //         drawKeypoints(images[i], features[i].keypoints, output );
  //         imwrite("key"+to_string(i)+"_"+to_string(features[i].keypoints.size())+".jpg", output);
  //     }

  //     //вывод матчей
  //     for (int i=0; i<pairwise_matches.size(); ++i)
  //     {
  //         if (pairwise_matches[i].src_img_idx >= 0)
  //         {
  //             Mat out;
  //             drawMatches(images[pairwise_matches[i].src_img_idx], features[pairwise_matches[i].src_img_idx].keypoints, images[pairwise_matches[i].dst_img_idx], features[pairwise_matches[i].dst_img_idx].keypoints, pairwise_matches[i].matches, out);
  //             imwrite("match"+to_string(i)+".jpg", out);
  //         }
  //     }
      
      
  //     for (int i=0; i<pairwise_matches.size(); ++i)
  //     {
  //         int src_img_idx = pairwise_matches[i].src_img_idx;
  //         int dst_img_idx = pairwise_matches[i].dst_img_idx;
  //         if (pairwise_matches[i].src_img_idx >= 0)
  //         {
  //             bool flag = true;
  //             while (flag)
  //             {
  //                 vector<Point2f> all_points_src, all_points_dst;
  //                 for (int j=0; j<pairwise_matches[i].matches.size(); ++j)
  //                 {
  //                     all_points_src.push_back(features[src_img_idx].keypoints[pairwise_matches[i].matches[j].queryIdx].pt);
  //                     all_points_dst.push_back(features[dst_img_idx].keypoints[pairwise_matches[i].matches[j].trainIdx].pt);
  //                 }
  //                 vector<vector<Point>> contours_src(1), contours_dst(1);
  //                 vector<Point2f> _src, _dst;
  //                 //выделяем 4 случайные
  //                 for (int j = 0; j<4; ++j)
  //                 {
  //                     int number = rand() % pairwise_matches[i].matches.size();
  //                     contours_src[0].push_back(features[src_img_idx].keypoints[pairwise_matches[i].matches[number].queryIdx].pt);
  //                     contours_dst[0].push_back(features[dst_img_idx].keypoints[pairwise_matches[i].matches[number].trainIdx].pt);
  //                     _src.push_back(features[src_img_idx].keypoints[pairwise_matches[i].matches[number].queryIdx].pt);
  //                     _dst.push_back(features[dst_img_idx].keypoints[pairwise_matches[i].matches[number].trainIdx].pt);
  //                 }
  //                 //построение маски
  //                 Mat mask_src(images[src_img_idx].size(), CV_8UC1, cv::Scalar(0));
  //                 Mat mask_dst(images[dst_img_idx].size(), CV_8UC1, cv::Scalar(0));

  //                 drawContours(mask_src, contours_src, -1, cv::Scalar(255), CV_FILLED);
  //                 drawContours(mask_dst, contours_dst, -1, cv::Scalar(255), CV_FILLED);

  //                 Mat src_out;
  //                 images[src_img_idx].copyTo(src_out, mask_src);

  //                 Mat dst_out;
  //                 images[dst_img_idx].copyTo(dst_out, mask_dst);

  //                 Mat H = getPerspectiveTransform(_src, _dst);
  //                 perspectiveTransform(all_points_src, all_points_src, H);
  //                 warpPerspective(src_out, src_out, H, Size());
  //                 Mat out = src_out-dst_out;
  //                 int res=0, count = 0;

  //                 Mat points(0, 1, CV_32FC3), points2(0, 1, CV_32FC2), labels;
  //                 for (int j=0; j<all_points_src.size(); ++j)
  //                 {
  //                     if (pointPolygonTest(contours_dst[0], all_points_dst[j], false) > 0)
  //                     {
  //                         circle(out, all_points_dst[j], 3, Scalar(255,0,0));
  //                         circle(out, all_points_src[j], 3, Scalar(255,0,0));
  //                         line(out, all_points_dst[j], all_points_src[j], Scalar(0,0,255));
  //                         res += norm(all_points_src[j]-all_points_dst[j]);
  //                         count += 1;
  //                         points.push_back(Point3f(all_points_dst[j].x, all_points_dst[j].y, norm(all_points_src[j]-all_points_dst[j])));
  //                         points2.push_back(all_points_src[j]);
  //                     }
  //                 }

  //                 if (count >= 4)
  //                 {
  //                     double crit = min(sqrt(contourArea(contours_src[0])), sqrt(contourArea(contours_dst[0])));
  //                     std::cout<<i<<endl<<"point: "<<res<<endl<<"count: "<<count<<endl<<"res/count: "<<res/count<<endl<<"sqrt(s): "<<crit<<endl<<"crit1: "<<(res/count)/crit<<endl;
  //                     if (res/count < crit*0.1)
  //                     {
  //                           kmeans(points, 2, labels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS);
  // //                          for (int j=0; j<points.rows; ++j)
  // //                          {
  // //                                circle(out, points2.at<Point2f>(j), 3, colorTab[labels.at<int>(j)]);
  // //                              circle(out, points.at<Point2f>(j), 3, colorTab[labels.at<int>(j)]);
  // //                                line(out, points.at<Point2f>(j), points2.at<Point2f>(j), colorTab[labels.at<int>(j)]);
  // //                          }
  //                         imwrite(to_string(i)+" "+to_string((res/count)/crit)+"d1.jpg",dst_out);
  //                         imwrite(to_string(i)+" "+to_string((res/count)/crit)+"s2.jpg",src_out);
  //                         imwrite(to_string(i)+" "+to_string((res/count)/crit)+"c.jpg", out);
  //     //                    std::cout <<"norm: "<< norm(src_out-dst_out)<<endl<<"s1: "<<contourArea(contours_src[0])<<endl<<"s2: "<<contourArea(contours_dst[0])<<endl<<"crit2: "<<norm(src_out-dst_out)/crit<<std::endl;
  //                         flag = false;


  //                    }
  //                    else
  //                     {
  //                         int a=0;
  // //                        imwrite(to_string(i)+" "+to_string((res/count)/crit)+"d1.jpg",dst_out);
  // //                        imwrite(to_string(i)+" "+to_string((res/count)/crit)+"s2.jpg",src_out);
  // //                        imwrite(to_string(i)+" "+to_string((res/count)/crit)+"c.jpg", out);
  //     //                    std::cout <<"norm: "<< norm(src_out-dst_out)<<endl<<"s1: "<<contourArea(contours_src[0])<<endl<<"s2: "<<contourArea(contours_dst[0])<<endl<<"crit2: "<<norm(src_out-dst_out)/crit<<std::endl;
  //                    }
  //                }
  //             }
  //         }
  //     }

  // //    const int MAX_CLUSTERS = 5;
  // //    Scalar colorTab[] =
  // //    {
  // //        Scalar(0, 0, 255),
  // //        Scalar(0,255,0),
  // //        Scalar(255,100,100),
  // //        Scalar(255,0,255),
  // //        Scalar(0,255,255)
  // //    };
  // //    Mat img(500, 500, CV_8UC3);
  // //    RNG rng(12345);
  // //    for(;;)
  // //    {
  // //        int k, clusterCount = rng.uniform(2, MAX_CLUSTERS+1);
  // //        int i, sampleCount = rng.uniform(1, 1001);
  // //        Mat points(sampleCount, 1, CV_32FC3), labels;
  // //        clusterCount = MIN(clusterCount, sampleCount);
  // //        Mat centers;
  // //        /* generate random sample from multigaussian distribution */
  // //        for( k = 0; k < clusterCount; k++ )
  // //        {
  // //            Point3f center;
  // //            center.x = rng.uniform(0, img.cols);
  // //            center.y = rng.uniform(0, img.rows);
  // //            center.z = rng.uniform(0, 10);
  // //            Mat pointChunk = points.rowRange(k*sampleCount/clusterCount,
  // //                                             k == clusterCount - 1 ? sampleCount :
  // //                                             (k+1)*sampleCount/clusterCount);
  // //            rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y, center.z), Scalar(img.cols*0.05, img.rows*0.05, 5));
  // //        }
  // //        randShuffle(points, 1, &rng);
  // //        kmeans(points, clusterCount, labels,
  // //            TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
  // //               3, KMEANS_PP_CENTERS, centers);
  // //        img = Scalar::all(0);
  // //        for( i = 0; i < sampleCount; i++ )
  // //        {
  // //            int clusterIdx = labels.at<int>(i);
  // //            Point ipt = points.at<Point2f>(i);
  // //            circle( img, ipt, 2, colorTab[clusterIdx]);
  // //        }
  // //        imshow("clusters", img);
  // //        char key = (char)waitKey();
  // //        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
  // //            break;
  // //    }
  //     waitKey();
  // }



#include <iostream>
#include <fstream>
#include <string>
#include <ctype.h>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
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
#include "opencv2/video/tracking.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // Load two images and allocate other structures
    Mat imgA = imread("/home/ggoha/Документы/Учеба/stitching/1505/build-opticalflow-Desktop-Debug/IMG_1158.JPG", CV_LOAD_IMAGE_GRAYSCALE);
    Mat imgB = imread("/home/ggoha/Документы/Учеба/stitching/1505/build-opticalflow-Desktop-Debug/IMG_1159.JPG", CV_LOAD_IMAGE_GRAYSCALE);

    Size img_sz = imgA.size();
    Mat imgC(img_sz,1);

    int win_size = 15;
    int maxCorners = 20;
    double qualityLevel = 0.05;
    double minDistance = 5.0;
    int blockSize = 3;
    double k = 0.04;
    std::vector<cv::Point2f> cornersA;
    cornersA.reserve(maxCorners);
    std::vector<cv::Point2f> cornersB;
    cornersB.reserve(maxCorners);


    goodFeaturesToTrack( imgA,cornersA,maxCorners,qualityLevel,minDistance,cv::Mat());
    goodFeaturesToTrack( imgB,cornersB,maxCorners,qualityLevel,minDistance,cv::Mat());

    cornerSubPix( imgA, cornersA, Size( win_size, win_size ), Size( -1, -1 ),
                  TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03 ) );

    cornerSubPix( imgB, cornersB, Size( win_size, win_size ), Size( -1, -1 ),
                  TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03 ) );

    // Call Lucas Kanade algorithm

    CvSize pyr_sz = Size( img_sz.width+8, img_sz.height/3 );

    std::vector<uchar> features_found;
    features_found.reserve(maxCorners);
    std::vector<float> feature_errors;
    feature_errors.reserve(maxCorners);

    calcOpticalFlowPyrLK( imgA, imgB, cornersA, cornersB, features_found, feature_errors ,
        Size( win_size, win_size ), 5,
         cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 ), 0 );

    // Make an image of the results

    for( int i=0; i < features_found.size(); i++ ){
            cout<<"Error is "<<feature_errors[i]<<endl;
            //continue;

        cout<<"Got it"<<endl;
        Point p0( ceil( cornersA[i].x ), ceil( cornersA[i].y ) );
        Point p1( ceil( cornersB[i].x ), ceil( cornersB[i].y ) );
        line( imgC, p0, p1, CV_RGB(255,255,255), 2 );
    }

    namedWindow( "ImageA", 0 );
    namedWindow( "ImageB", 0 );
    namedWindow( "LKpyr_OpticalFlow", 0 );

    imshow( "ImageA", imgA );
    imshow( "ImageB", imgB );
    imshow( "LKpyr_OpticalFlow", imgC );

    cvWaitKey(0);

    return 0;
}


                        for (int j=0; j<clusterCount; ++j)
                        {
                            vector<Point2f> src_hul, dst_hul;
                            convexHull(src_clusters[j], src_hul);
                            convexHull(dst_clusters[j], dst_hul);
                            vector<vector<Point>> src_hul_contur(1), dst_hul_contur(1);
                            for (int z=0; z<src_hul.size(); ++z)
                                src_hul_contur[0].push_back(Point(src_hul[z].x, src_hul[z].y));
                            for (int z=0; z<dst_hul.size(); ++z)
                                dst_hul_contur[0].push_back(Point(dst_hul[z].x, dst_hul[z].y));
//                            if (src_clusters[j].size()>=4 && dst_clusters[j].size()>=4)
//                            {
//                                Mat H = findHomography(src_clusters[j], dst_clusters[j]);
//                                //построение маски
//                                Mat mask_src(images[src_img_idx].size(), CV_8UC1, cv::Scalar(0));
//                                Mat mask_dst(images[dst_img_idx].size(), CV_8UC1, cv::Scalar(0));

//                                drawContours(mask_src, src_hul_contur, -1, cv::Scalar(255), CV_FILLED);
//                                drawContours(mask_dst, dst_hul_contur, -1, cv::Scalar(255), CV_FILLED);

//                                Mat src_mask_out;
//                                images[src_img_idx].copyTo(src_mask_out, mask_src);
//                                warpPerspective(src_mask_out, src_mask_out, H, Size());
//                                imwrite(to_string(i)+"m"+to_string(j)+"s.jpg",src_mask_out);

//                                Mat dst_mask_out;
//                                images[dst_img_idx].copyTo(dst_mask_out, mask_dst);
//                                imwrite(to_string(i)+"m"+to_string(j)+"d.jpg",dst_mask_out);
//                            }


                        }