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
#include "opencv/ml.h"


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
        resize(full_images[i], images[i], Size(), 0.5, 0.5);
    }

    //находим ключи
    vector<ImageFeatures> features(num_images);
    Ptr<FeaturesFinder> finder = new SurfFeaturesFinder(20, 3, 2);
    //уровень отсева, количество октав, колличество слоев,
    for (int i=0; i<num_images; ++i)
        (*finder)(images[i], features[i]);
    finder->collectGarbage();

        //находим соответствия
        vector<MatchesInfo> pairwise_matches;
        BestOf2NearestMatcher matcher(false, 0.4f, 6, 6);
        //gpu, порог принятия, минимальное кооличество соотнесений для 2D проективного прообразования на ???
        matcher(features, pairwise_matches);

        //вывод ключей
        for (int i=0; i<num_images; ++i)
        {
            Mat output;
            drawKeypoints(images[i], features[i].keypoints, output );
            imwrite("key"+to_string(i)+"_"+to_string(features[i].keypoints.size())+".jpg", output);
        }

        //вывод матчей
        for (int i=0; i<pairwise_matches.size(); ++i)
        {
            if (pairwise_matches[i].src_img_idx >= 0)
            {
                Mat out;
                drawMatches(images[pairwise_matches[i].src_img_idx], features[pairwise_matches[i].src_img_idx].keypoints, images[pairwise_matches[i].dst_img_idx], features[pairwise_matches[i].dst_img_idx].keypoints, pairwise_matches[i].matches, out);
                imwrite("match"+to_string(i)+".jpg", out);
            }
        }


//        for (int i=0; i<pairwise_matches.size(); ++i)
//        {
//            int src_img_idx = pairwise_matches[i].src_img_idx;
//            int dst_img_idx = pairwise_matches[i].dst_img_idx;
//            if (pairwise_matches[i].src_img_idx >= 0)
//            {
//                bool flag = true;
//                while (flag)
//                {
//                    vector<Point2f> all_points_src, all_points_dst;
//                    for (int j=0; j<pairwise_matches[i].matches.size(); ++j)
//                    {
//                        all_points_src.push_back(features[src_img_idx].keypoints[pairwise_matches[i].matches[j].queryIdx].pt);
//                        all_points_dst.push_back(features[dst_img_idx].keypoints[pairwise_matches[i].matches[j].trainIdx].pt);
//                    }
//                    vector<vector<Point>> contours_src(1), contours_dst(1);
//                    vector<Point2f> _src, _dst;
//                    //выделяем 4 случайные
//                    for (int j = 0; j<4; ++j)
//                    {
//                        int number = rand() % pairwise_matches[i].matches.size();
//                        contours_src[0].push_back(features[src_img_idx].keypoints[pairwise_matches[i].matches[number].queryIdx].pt);
//                        contours_dst[0].push_back(features[dst_img_idx].keypoints[pairwise_matches[i].matches[number].trainIdx].pt);
//                        _src.push_back(features[src_img_idx].keypoints[pairwise_matches[i].matches[number].queryIdx].pt);
//                        _dst.push_back(features[dst_img_idx].keypoints[pairwise_matches[i].matches[number].trainIdx].pt);
//                    }
//                    //построение маски
//                    Mat mask_src(images[src_img_idx].size(), CV_8UC1, cv::Scalar(0));
//                    Mat mask_dst(images[dst_img_idx].size(), CV_8UC1, cv::Scalar(0));

//                    drawContours(mask_src, contours_src, -1, cv::Scalar(255), CV_FILLED);
//                    drawContours(mask_dst, contours_dst, -1, cv::Scalar(255), CV_FILLED);

//                    Mat src_out;
//                    images[src_img_idx].copyTo(src_out, mask_src);

//                    Mat dst_out;
//                    images[dst_img_idx].copyTo(dst_out, mask_dst);

//                    Mat H = getPerspectiveTransform(_src, _dst);
//                    perspectiveTransform(all_points_src, all_points_src, H);
//                    warpPerspective(src_out, src_out, H, Size());
//                    Mat out = src_out-dst_out;
//                    int res=0, count = 0;
//                    Mat points(0, 1, CV_32FC3), points2(0, 1, CV_32FC2), labels;
//                    for (int j=0; j<all_points_src.size(); ++j)
//                    {
//                        if (pointPolygonTest(contours_dst[0], all_points_dst[j], false) > 0)
//                        {
////                            circle(out, all_points_dst[j], 3, Scalar(255,0,0));
////                            circle(out, all_points_src[j], 3, Scalar(255,0,0));
////                            line(out, all_points_dst[j], all_points_src[j], Scalar(0,0,255));
//                            res += norm(all_points_src[j]-all_points_dst[j]);
//                            count += 1;
//                            points.push_back(Point3f(all_points_dst[j].x, all_points_dst[j].y, 10*norm(all_points_src[j]-all_points_dst[j])));
//                            points2.push_back(all_points_src[j]);
//                        }
//                    }

//                    if (count >= 8)
//                    {
//                        double crit = min(sqrt(contourArea(contours_src[0])), sqrt(contourArea(contours_dst[0])));
//                        std::cout<<i<<endl<<"point: "<<res<<endl<<"count: "<<count<<endl<<"res/count: "<<res/count<<endl<<"sqrt(s): "<<crit<<endl<<"crit1: "<<(res/count)/crit<<endl;
//                        if (res/count < crit*0.1)
//                        {
//                            kmeans(points, points.rows/8, labels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS);
//                           for (int j=0; j<points.rows; ++j)
//                           {
//                                 circle(out, points2.at<Point2f>(j), 3, colorTab[labels.at<int>(j)]);
//                                 circle(out, Point2f(points.at<Point3f>(j).x, points.at<Point3f>(j).y), 3, colorTab[labels.at<int>(j)]);
//                                 line(out, Point2f(points.at<Point3f>(j).x, points.at<Point3f>(j).y), points2.at<Point2f>(j), colorTab[labels.at<int>(j)]);
//                           }
//                            imwrite(to_string(i)+" "+to_string((res/count)/crit)+"d1.jpg",dst_out);
//                            imwrite(to_string(i)+" "+to_string((res/count)/crit)+"s2.jpg",src_out);
//                            imwrite(to_string(i)+" "+to_string((res/count)/crit)+"c.jpg", out);
//        //                    std::cout <<"norm: "<< norm(src_out-dst_out)<<endl<<"s1: "<<contourArea(contours_src[0])<<endl<<"s2: "<<contourArea(contours_dst[0])<<endl<<"crit2: "<<norm(src_out-dst_out)/crit<<std::endl;
//                            flag = false;
//                       }
//                       else
//                        {
////                            imwrite(to_string(i)+" "+to_string((res/count)/crit)+"d1.jpg",dst_out);
////                            imwrite(to_string(i)+" "+to_string((res/count)/crit)+"s2.jpg",src_out);
////                            imwrite(to_string(i)+" "+to_string((res/count)/crit)+"c.jpg", out);
//        //                    std::cout <<"norm: "<< norm(src_out-dst_out)<<endl<<"s1: "<<contourArea(contours_src[0])<<endl<<"s2: "<<contourArea(contours_dst[0])<<endl<<"crit2: "<<norm(src_out-dst_out)/crit<<std::endl;
//                       }
//                   }
//                }
//            }
//        }
                for (int i=0; i<pairwise_matches.size(); ++i)
                {
                    int src_img_idx = pairwise_matches[i].src_img_idx;
                    int dst_img_idx = pairwise_matches[i].dst_img_idx;
                    if (pairwise_matches[i].src_img_idx >= 0)
                    {
                        vector<Point2f> all_points_src, all_points_src_copy, all_points_dst;

                        Mat src_out;
                        images[src_img_idx].copyTo(src_out);

                        Mat dst_out;
                        images[dst_img_idx].copyTo(dst_out);

                        //ищем преобраование
                        //гомография для src
                        //сдвиг для dst
                        Mat H, C;
                        bool flag = true;
                        //углы
                        vector<Point2f> c, c1;
                        while (flag)
                        {
                            for (int j=0; j<pairwise_matches[i].matches.size(); ++j)
                            {
                                all_points_src.push_back(features[src_img_idx].keypoints[pairwise_matches[i].matches[j].queryIdx].pt);
                                all_points_src_copy.push_back(features[src_img_idx].keypoints[pairwise_matches[i].matches[j].queryIdx].pt);
                                all_points_dst.push_back(features[dst_img_idx].keypoints[pairwise_matches[i].matches[j].trainIdx].pt);
                            }
                            H = findHomography(all_points_src, all_points_dst);
                            //ищем сдвиг,чтобы вошли обе картинки
                            c.clear();
                            c.push_back(Point2f(0,0));c.push_back(Point2f(src_out.size().width,0));c.push_back(Point2f(0,src_out.size().height));c.push_back(Point2f(src_out.size().width,src_out.size().width));
                            perspectiveTransform(c,c1,H);
                            C = (Mat_<double>(3,3) << 1, 0, -boundingRect(c1).tl().x, 0, 1, -boundingRect(c1).tl().y, 0, 0, 1);

                            perspectiveTransform(all_points_src, all_points_src, C*H);
                            perspectiveTransform(all_points_src_copy, all_points_src_copy, C);
                            perspectiveTransform(all_points_dst, all_points_dst, C);

                            int res=0, count=0;
                            for (int j=0; j<all_points_src.size(); ++j)
                            {
                                        res += norm(all_points_src[j]-all_points_dst[j]);
                                        count += 1;
                            }
                            if (res/count < 1000)
                                flag = false;
                        }

                        Size a = Size(max(boundingRect(c1).size().width, dst_out.size().width-boundingRect(c1).tl().x),max(boundingRect(c1).size().height, dst_out.size().height-boundingRect(c1).tl().y));
                        warpPerspective(src_out, src_out, C*H, a);
                        warpPerspective(dst_out, dst_out, C, a);
                        imwrite("test1.jpg", src_out);
                        imwrite("test2.jpg", dst_out);
                        Mat points_dst(0, 1, CV_32FC3), points_src(0, 1, CV_32FC2), points_src_copy(0, 1, CV_32FC2), labels;
                        //преобразуем для обучения
                        Mat out_test = src_out - dst_out;
                        for (int j=0; j<all_points_src.size(); ++j)
                        {
                                    circle(out_test, all_points_dst[j], 3, Scalar(0, 0, 255));
                                    circle(out_test, all_points_src[j], 3, Scalar(0, 0, 255));
                                    line(out_test, all_points_src[j], all_points_dst[j], Scalar(0, 0, 255));
                                    points_dst.push_back(Point3f(all_points_dst[j].x, all_points_dst[j].y, 100*norm(all_points_src[j]-all_points_dst[j])));
                                    points_src.push_back(all_points_src[j]);
                                    points_src_copy.push_back(all_points_src_copy[j]);
                        }
                        imwrite(to_string(i)+"o.jpg",out_test);
                        Mat out = src_out - dst_out;
                        int clusterCount = 20;
                        vector<vector<Point2f>> src_clusters(clusterCount), dst_clusters(clusterCount);
                        kmeans(points_dst, clusterCount, labels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS);
                        for (int j=0; j<points_dst.rows; ++j)
                        {
                             Scalar color = Scalar(255*(labels.at<int>(j)%2),
                                                   255/2*(labels.at<int>(j)%3),
                                                   255/4*(labels.at<int>(j)%5));
                             circle(out, points_src.at<Point2f>(j), 3, color);
                             circle(out, Point2f(points_dst.at<Point3f>(j).x, points_dst.at<Point3f>(j).y), 3, color);
                             line(out, Point2f(points_dst.at<Point3f>(j).x, points_dst.at<Point3f>(j).y), points_src.at<Point2f>(j), color);
                             src_clusters[labels.at<int>(j)].push_back(points_src_copy.at<Point2f>(j));
                             dst_clusters[labels.at<int>(j)].push_back(Point2f(points_dst.at<Point3f>(j).x, points_dst.at<Point3f>(j).y));
                        }

                        Mat dst_trainData = Mat(points_dst.rows, 2, CV_32FC1);
                        for(int z=0; z<points_dst.rows; ++z)
                        {
                            dst_trainData.at<float>(z, 0) = all_points_dst[z].x;
                            dst_trainData.at<float>(z, 1) = all_points_dst[z].y;
                            cout<<all_points_dst[z].x<<" "<<all_points_dst[z].y<<endl;
                        }
                        Mat dst_trainClasses = Mat(0, 1, CV_32FC1 );
                        for(int z=0; z<labels.rows; ++z)
                            dst_trainClasses.push_back(float(labels.at<int>(z)));
                        CvKNearest dst_knn( dst_trainData, dst_trainClasses);

                        float _sample[2];
                        CvMat sample = cvMat( 1, 2, CV_32FC1, _sample );
                        vector<Mat> dst_masks;
                        for (int j=0; j<clusterCount; ++j)
                        {
                            dst_masks.push_back(Mat (dst_out.size(), CV_8UC1, Scalar(0)));
                        }
                        for( int q = 0; q < dst_out.cols; q++ )
                        {
                            for( int w = 0; w < dst_out.rows; w++ )
                            {
                                sample.data.fl[0] = (float)q;
                                sample.data.fl[1] = (float)w;

                                int test = dst_knn.find_nearest(&sample,10);
                                Scalar color = Scalar(255*(test%2),
                                                      255/2*(test%3),
                                                      255/4*(test%5));
                                //cout<<i<<" "<<response<<" "<<test<<" "<<w<<" "<<q<<endl;
                                circle(dst_out, Point2f(q, w), 1, color);
                                dst_masks[test].at<uchar>(w, q) = 255;
                            }
                        }

                        Mat src_trainData = Mat(points_src_copy.rows, 2, CV_32FC1);
                        for(int z=0; z<points_src.rows; ++z)
                        {
                            src_trainData.at<float>(z, 0) = all_points_src_copy[z].x;
                            src_trainData.at<float>(z, 1) = all_points_src_copy[z].y;
                            cout<<all_points_dst[z].x<<" "<<all_points_dst[z].y<<endl;
                        }
                        Mat src_trainClasses = Mat(0, 1, CV_32FC1 );
                        for(int z=0; z<labels.rows; ++z)
                            src_trainClasses.push_back(float(labels.at<int>(z)));
                        CvKNearest src_knn( src_trainData, src_trainClasses);

                        vector<Mat> src_masks;
                        for (int j=0; j<clusterCount; ++j)
                        {
                            src_masks.push_back(Mat (src_out.size(), CV_8UC1, Scalar(0)));
                        }
                        for( int q = 0; q < src_out.cols; q++ )
                        {
                            for( int w = 0; w < src_out.rows; w++ )
                            {
                                sample.data.fl[0] = (float)q;
                                sample.data.fl[1] = (float)w;

                                int test = src_knn.find_nearest(&sample,10);
                                Scalar color = Scalar(255*(test%2),
                                                      255/2*(test%3),
                                                      255/4*(test%5));
                                //cout<<i<<" "<<response<<" "<<test<<" "<<w<<" "<<q<<endl;
                                circle(src_out, Point2f(q, w), 1, color);
                                src_masks[test].at<uchar>(w, q) = 255;
                            }
                        }


                        imwrite(to_string(i)+"d.jpg",images[dst_img_idx]);
                        imwrite(to_string(i)+"d1.jpg",dst_out);
                        imwrite(to_string(i)+"s.jpg",images[src_img_idx]);
                        imwrite(to_string(i)+"s1.jpg",src_out);
                        imwrite(to_string(i)+"c.jpg", out);
                        //Mat res(images[dst_img_idx].size(), CV_32FC2, Scalar(0));;
                            Mat dst_mask_out0, src_mask_out0;
                            images[dst_img_idx].copyTo(dst_mask_out0, dst_masks[0]);
                            //imwrite(to_string(i)+"m"+to_string(0)+"m2.jpg", dst_masks[0]);
                            //imwrite(to_string(i)+"m"+to_string(0)+"md.jpg", dst_mask_out0);

                            images[src_img_idx].copyTo(src_mask_out0, src_masks[0]);
                            //imwrite(to_string(i)+"m"+to_string(0)+"m1.jpg", src_masks[0]);
                            //imwrite(to_string(i)+"m"+to_string(0)+"ms2.jpg", src_mask_out0);

                            if (src_clusters[0].size()>=4 && dst_clusters[0].size()>=4)
                            {
                                Mat H = findHomography(src_clusters[0], dst_clusters[0]);
                                warpPerspective(src_masks[0], src_masks[0], H, Size());
                                warpPerspective(src_mask_out0, src_mask_out0, H, Size());
                                //imwrite(to_string(i)+"m"+to_string(0)+"ms1.jpg", src_mask_out0);
                                //res += src_mask_out;
                            }

                        for (int j=1; j<clusterCount; ++j)
                        {
                            Mat dst_mask_out, src_mask_out;
                            images[dst_img_idx].copyTo(dst_mask_out, dst_masks[j]);
                            //imwrite(to_string(i)+"m"+to_string(j)+"m2.jpg", dst_masks[j]);
                            //imwrite(to_string(i)+"m"+to_string(j)+"md.jpg", dst_mask_out);

                            images[src_img_idx].copyTo(src_mask_out, src_masks[j]);
                            //imwrite(to_string(i)+"m"+to_string(j)+"m1.jpg", src_masks[j]);
                            //imwrite(to_string(i)+"m"+to_string(j)+"ms2.jpg", src_mask_out);

                            if (src_clusters[j].size()>=4 && dst_clusters[j].size()>=4)
                            {
                                Mat H = findHomography(src_clusters[j], dst_clusters[j]);
                                warpPerspective(src_masks[j], src_masks[j], H, Size());
                                warpPerspective(src_mask_out, src_mask_out, H, Size());
                                //imwrite(to_string(i)+"m"+to_string(j)+"ms1.jpg", src_mask_out);
                                src_mask_out0 += src_mask_out;
                            }

                        }
                        imwrite(to_string(i)+"res.jpg", src_mask_out0);
                    }
                }

        waitKey();
}
