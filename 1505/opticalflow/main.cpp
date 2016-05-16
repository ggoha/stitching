#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
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

int main( int argc, char** argv )
{
    Mat img_warped1 = Mat::ones(Size(100, 100), CV_16SC3);
    line(img_warped1, Point(0,0), Point(100,100), Scalar(255, 0, 9));
    img_warped1 = img_warped1 * 200;

    Mat img_warped2 = Mat::ones(Size(100, 100), CV_16SC3);
    line(img_warped2, Point(0,0), Point(100,100), Scalar(255, 0, 9));
    img_warped2 = img_warped2 * 60000;
    vector<Point2f> c, c1, c2;
    c.push_back(Point2f(0,0));c.push_back(Point2f(100,0));c.push_back(Point2f(0,100));c.push_back(Point2f(100,100));

    Point2f inputQuad[4];
    Point2f outputQuad[4];
    inputQuad[0] = Point2f( -30,-60 );
    inputQuad[1] = Point2f( +50,-50);
    inputQuad[2] = Point2f( +100,+50);
    inputQuad[3] = Point2f( -50,+50  );
    outputQuad[0] = Point2f( -60,-12 );
    outputQuad[1] = Point2f( +100,-100);
    outputQuad[2] = Point2f( +200,+100);
    outputQuad[3] = Point2f( -100,+100  );
    Mat rot_mat = getPerspectiveTransform( inputQuad, outputQuad );


    perspectiveTransform(c,c1,rot_mat);
    Mat C = (Mat_<double>(3,3) << 1, 0, -boundingRect(c1).tl().x, 0, 1, -boundingRect(c1).tl().y, 0, 0, 1);
    warpPerspective(img_warped2, img_warped2, C*rot_mat, boundingRect(c1).size());

    vector<Point> corners; corners.push_back(Point(0,-0)); corners.push_back(Point(boundingRect(c1).tl().x,boundingRect(c1).tl().y));
    vector<Size> sizes;

    sizes.push_back(img_warped1.size());
    sizes.push_back(boundingRect(c1).size());

    Ptr<Blender> blender;
    blender = Blender::createDefault(Blender::MULTI_BAND);
    blender->prepare(corners, sizes);
    blender->feed(img_warped1, Mat::ones(img_warped1.size(), CV_8UC1), corners[0]);
    blender->feed(img_warped2, Mat::ones(img_warped2.size(), CV_8UC1), corners[1]);

    Mat result, result_mask;
    blender->blend(result, result_mask);
    imshow("test", result);
    waitKey();
    return 0;
}
