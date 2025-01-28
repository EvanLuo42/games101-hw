#include "rasterizer.hpp"
#include <Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Matrix4f get_view_matrix(Vector3f eye_pos)
{
    Matrix4f view = Matrix4f::Identity();

    Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1,
        -eye_pos[2], 0, 0, 0, 1;

    view = translate * view;

    return view;
}

Matrix4f get_model_matrix(const float rotation_angle)
{
    Matrix4f model = Matrix4f::Identity();

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.
    model << cos(rotation_angle), -sin(rotation_angle), 0, 0, sin(rotation_angle), cos(rotation_angle), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

    return model;
}

Matrix4f get_projection_matrix(
    const float eye_fov,
    const float aspect_ratio,
    const float zNear,
    const float zFar)
{
    // Students will implement this function

    Matrix4f projection = Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.
    const auto n = -zNear;
    const auto f = -zFar;
    const auto t = zNear * tan(eye_fov / 2);
    const auto b = -t;
    const auto r = t * aspect_ratio;
    const auto l = -r;
    projection <<
        2 * n / (r - l), 0, 0, 0,
        0, 2 * n / (t - b), 0, 0,
        0, 0, (n + f) / (f - n), -(2 * n * f) / (f - n),
        0, 0, 1, 0;

    return projection;
}

auto main(const int argc, const char **argv) -> int {
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);

    const Vector3f eye_pos = {0, 0, 5};

    const std::vector<Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    const std::vector<Vector3i> ind{{0, 1, 2}};

    const auto pos_id = r.load_positions(pos);
    const auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}
