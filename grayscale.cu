#include <opencv2/opencv.hpp>
#include <vector>

__global__ void grayscale_pixels(
    unsigned char* pixels_rgb_directive, 
    unsigned char* gray_pixels_directive, 
    int cols, int rows
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < cols && y < rows) {
        int red = pixels_rgb_directive[3 * (y * cols + x)];
        int green = pixels_rgb_directive[3 * (y * cols + x) + 1];
        int blue = pixels_rgb_directive[3 * (y * cols + x) + 2];

        gray_pixels_directive[y * cols + x] = (307 * red + 604 * green + 113 * blue) / 1024;
    }
}


int main() {
    cv::Mat input_image = cv::imread("in.jpg", cv::IMREAD_COLOR);

    unsigned char* pixels_rgb = input_image.data;
    std::vector<unsigned char> gray_pixels(input_image.rows * input_image.cols);
    cv::Mat output_image(input_image.rows, input_image.cols, CV_8UC1, gray_pixels.data());
    
    unsigned char* pixels_rgb_directive;
    unsigned char* gray_pixels_directive;

    cudaMalloc(&pixels_rgb_directive, input_image.rows * input_image.cols * 3);
    cudaMalloc(&gray_pixels_directive, input_image.rows * input_image.cols);
    cudaMemcpy(pixels_rgb_directive, pixels_rgb, 3 * input_image.rows * input_image.cols, cudaMemcpyHostToDevice);

    dim3 nb_threads(32, 32);
    dim3 nb_blocks(input_image.cols / nb_threads.x + 1, input_image.rows / nb_threads.y + 1);

    grayscale_pixels<<<nb_blocks, nb_threads>>>(pixels_rgb_directive, gray_pixels_directive, input_image.cols, input_image.rows);

    cudaMemcpy(gray_pixels.data(), gray_pixels_directive, input_image.rows * input_image.cols, cudaMemcpyDeviceToHost);

    cv::imwrite("out.jpg", output_image);

    cudaFree(pixels_rgb_directive);
    cudaFree(gray_pixels_directive);

    return 0;
}
