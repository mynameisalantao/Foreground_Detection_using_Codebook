// import library
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <math.h>
# include<opencv2/opencv.hpp>  // getStructuringElement

// namespace
using namespace cv;
using namespace std;

// Function prototype
Mat read_image(int);
void codebook_update(Mat,int);
Mat Foreground_Detection(Mat);
void codebook_parameter(int, int);
void codebook_cache_parameter(int, int);
void save_image(Mat, int);
Mat median_filter(Mat);
Mat modify(Mat);

// Define
#define pi 3.1415926


// Picture parameter
const int max_codeword = 40;                       // 容許最多codeword的數量
const int image_row = 480;                         // 圖片長度
const int image_col = 640;                         // 圖片寬度
const float epsilon = 20;                          // cache能夠容許的color distortion
const int T_add = 250;                              // 在codebook_cache當中的codeword累計出現幾次以上就放到codebook
const int T_delete = 250;                           // 在codebook或在codebook_cache中的codebook連續幾次沒出現就丟掉
int copy_image[image_row + 2][image_col + 2] = { 0 };  // padding矩陣


// Parameter matrix
float codebook[image_row][image_col][max_codeword][9] = {0};         // v=[R,G,B],aux=<I_min,I_max,f,lamda,p,q> 存放背景
float codebook_cache[image_row][image_col][max_codeword][9] = {0};   // v=[R,G,B],aux=<I_min,I_max,f,lamda,p,q> 存放快取


// main
int main() {


	cout << "Update 20191107..." << endl;

	cout << "請確認輸入圖片有在路徑./input/當中..."<<endl;
	

	
	// 進行迭代運算
	for (int iter2 = 0; iter2 < 1; iter2++) {
		for (int iter = 1; iter < 530; iter++) {

			Mat image = read_image(iter);                 // input new data
			Mat output = Foreground_Detection(image);     // 前景偵測
			Mat stage1 = median_filter(output);           // median filter
			Mat stage2 = modify(stage1);
	

			save_image(stage2, iter);                     // 儲存結果
			codebook_update(image,iter);                  // 更新參數
			
		}
	}
	
	


	cout << "ok!";
	waitKey(0); // 等待一次按鍵，程式結束
	system("pause");
	return 0;

}

// 讀取圖片
Mat read_image(int data_num) {
	// parameter
	//char picture_name[70] = "C:\\Users\\alan\\Desktop\\HW1_dataset\\input\\in000";
	char picture_name[70] = "./input/in000";
	char data_type[5] = ".png";
	char graph[10];
	char add_zero[5] = "0";      // 對於1~9的檔案要補00
	char add_2zero[5] = "00";    // 對於10~99的檔案要補0
	// 建立檔名
	sprintf_s(graph, "%d", data_num);                   // 把int轉為string
	if (data_num < 10) {
		strcat_s(add_2zero, graph);
		strcat_s(picture_name, add_2zero);                      // 把兩個string合併
		strcat_s(picture_name, data_type);
	}
	else if (data_num < 100) {
		strcat_s(add_zero, graph);
		strcat_s(picture_name, add_zero);                      // 把兩個string合併
		strcat_s(picture_name, data_type);
	}
	else {
		strcat_s(picture_name, graph);                        // 把兩個string合併
		strcat_s(picture_name, data_type);
	}

	cout << picture_name;
	cout << " ";

	Mat image;
	image = imread(picture_name);//讀入圖片資料
	if (image.empty()) {
		cout << "Can't read image, please check the input file path..."<<endl;
	}
	else {
		cout << "Read picture=" << data_num << endl;
	}


	/*
	cout << "The column is " << image.cols << endl;        // 長度(行數)
	cout << "The row is " << image.rows << endl;           // 寬度(列數)
	cout << "The data type is " << image.type() << endl;   // Data type
	*/

	/*Data type*/
	/*
				C1	C2	C3	C4
		CV_8U	0	8	16	24
		CV_8S	1	9	17	25
		CV_16U	2	10	18	26
		CV_16S	3	11	19	27
		CV_32S	4	12	20	28
		CV_32F	5	13	21	29
		CV_64F	6	14	22	30
	*/
	return image;
}

// 儲存圖片
void save_image(Mat image,int data_num) {
	char data_type[5] = ".png";
	char graph[10];
	char add_5zero[20] = "00000";       // 對於1~9的檔案要補00
	char add_4zero[20] = "0000";        // 對於10~99的檔案要補0
	char add_3zero[20] = "000";         // 對於100~529的檔案要補0

	// 建立檔名
	sprintf_s(graph, "%d", data_num);                   // 把int轉為string
	if (data_num < 10) {
		strcat_s(add_5zero, graph);
		strcat_s(add_5zero, data_type);
		//cout << add_5zero << endl;
		imwrite(add_5zero, image);
	}
	else if (data_num < 100) {
		strcat_s(add_4zero, graph);
		strcat_s(add_4zero, data_type);
		//cout << add_4zero << endl;
		imwrite(add_4zero, image);
	}
	else {
		strcat_s(add_3zero, graph);
		strcat_s(add_3zero, data_type);
		//cout << add_3zero << endl;
		imwrite(add_3zero, image);
	}
	cout << "Save image=" << data_num << endl;

}



// Codebook參數更新
void codebook_update(Mat image, int times) {
	// parameter
	float sum_of_dot = 0;                // 暫存新資料與某cache的內積總和
	float sum_of_square_BGR = 0;         // 暫存新資料的BGR平方總和
	float brightness = 0;                // 暫存新資料的亮度
	float square_of_v = 0;               // 暫存某cache與原點的距離平方
	float distortion = 0;                // 暫存新資料與某cache的顏色差異
	float max_bound = 0;                 // 暫存某cache能容忍的最大亮度
	float min_bound = 0;                 // 暫存某cache能容忍的最小亮度 
	int find_flag = 0;                   // 有無在codebook或codebook_cache當中找到他的codeword，{0=沒有,1=有}

	cout << "Codebook updating..."<<endl;

	// 依序處理每個pixel
	for (int row = 0; row < image_row; row++) {
		for (int col = 0; col < image_col; col++) {

			// 初始化參數
			find_flag = 0;

			// 在codebook當中尋找新資料的codeword
			for (int num_codeword = 0; num_codeword < max_codeword; num_codeword++) {
				// 初始化參數
				sum_of_dot = 0;
				sum_of_square_BGR = 0;
				square_of_v = 0;
				// 若該codeword為空的，則換找下一個
				if (codebook[row][col][num_codeword][0] == 0 && codebook[row][col][num_codeword][1] == 0 && codebook[row][col][num_codeword][2] == 0)
					continue;
				// 若有東西，則計算與他的距離
				else {
					//依序計算BGR
					for (int BGR = 0; BGR < 3; BGR++) {
						sum_of_dot += float(image.at<cv::Vec3b>(row, col)[BGR]) * codebook[row][col][num_codeword][BGR];
						sum_of_square_BGR += pow(float(image.at<cv::Vec3b>(row, col)[BGR]),2);
						square_of_v += pow(codebook[row][col][num_codeword][BGR],2);
					}
					// 統計參數 [I_max*0.5<新資料亮度<min(I_max*1.2,I_min*2)]
					brightness = sqrt(sum_of_square_BGR);
					distortion = sqrt(sum_of_square_BGR - (pow(sum_of_dot, 2) / square_of_v));
					min_bound = 0.5 * codebook[row][col][num_codeword][4];
					max_bound = 1.2 * codebook[row][col][num_codeword][4];
					if (2 * codebook[row][col][num_codeword][3] < max_bound)
						max_bound = 2 * codebook[row][col][num_codeword][3];
		
					// 若[distortion小於epsilon]且[min_bound<新資料亮度<max_bound]，則屬於該codework
					if (distortion<= epsilon && min_bound< brightness && brightness< max_bound) {
						find_flag = 1;          // 找到了

						// 更新codebook參數  v=[R,G,B],aux=<I_min,I_max,f,lamda,p,q> 
						//依序計算BGR
						for (int BGR = 0; BGR < 3; BGR++) {
							codebook[row][col][num_codeword][BGR] = (codebook[row][col][num_codeword][5] * codebook[row][col][num_codeword][BGR] + float(image.at<cv::Vec3b>(row, col)[BGR])) / (codebook[row][col][num_codeword][5] + 1);
						}
						// 累計出現次數
						codebook[row][col][num_codeword][5] += 1;       
						// 更新最大亮度
						if (brightness > codebook[row][col][num_codeword][4])
							codebook[row][col][num_codeword][4] = brightness;
						// 更新最小亮度
						if (brightness < codebook[row][col][num_codeword][3])
							codebook[row][col][num_codeword][3] = brightness;
						// 更新最久沒出現的次數
						if (times - codebook[row][col][num_codeword][8] > codebook[row][col][num_codeword][6])
							codebook[row][col][num_codeword][6] = times - codebook[row][col][num_codeword][8];
						// 更新最後一次出現的時間點
						codebook[row][col][num_codeword][8] = times;
						break;
					}
				}
			}

			// 若沒有在codebook當中找到，則改在codebook_cache當中尋找新資料的codeword
			if (find_flag == 0) {

				for (int num_codeword = 0; num_codeword < max_codeword; num_codeword++) {
					// 初始化參數
					sum_of_dot = 0;
					sum_of_square_BGR = 0;
					square_of_v = 0;
					// 若該codeword為空的，則換找下一個
					if (codebook_cache[row][col][num_codeword][0] == 0 && codebook_cache[row][col][num_codeword][1] == 0 && codebook_cache[row][col][num_codeword][2] == 0)
						continue;
					// 若有東西，則計算與他的距離
					else {


						//依序計算BGR
						for (int BGR = 0; BGR < 3; BGR++) {
							sum_of_dot += float(image.at<cv::Vec3b>(row, col)[BGR]) * codebook_cache[row][col][num_codeword][BGR];
							sum_of_square_BGR += pow(float(image.at<cv::Vec3b>(row, col)[BGR]), 2);
							square_of_v += pow(codebook_cache[row][col][num_codeword][BGR], 2);
						}
						// 統計參數 [I_max*0.5<新資料亮度<min(I_max*1.2,I_min*2)]
						brightness = sqrt(sum_of_square_BGR);
						distortion = sqrt(sum_of_square_BGR-(pow(sum_of_dot,2) / square_of_v));
						min_bound = 0.5 * codebook_cache[row][col][num_codeword][4];
						max_bound = 1.2 * codebook_cache[row][col][num_codeword][4];
						if (2 * codebook_cache[row][col][num_codeword][3] < max_bound)
							max_bound = 2 * codebook_cache[row][col][num_codeword][3];


						// 若[distortion小於epsilon]且[min_bound<新資料亮度<max_bound]，則屬於該codework
						if (distortion <= epsilon && min_bound < brightness && brightness < max_bound) {
							find_flag = 1;          // 找到了


							// 更新codebook參數  v=[R,G,B],aux=<I_min,I_max,f,lamda,p,q> 
							//依序計算BGR
							for (int BGR = 0; BGR < 3; BGR++) {
								codebook_cache[row][col][num_codeword][BGR] = (codebook_cache[row][col][num_codeword][5] * codebook_cache[row][col][num_codeword][BGR] + float(image.at<cv::Vec3b>(row, col)[BGR])) / (codebook_cache[row][col][num_codeword][5] + 1);
							}
							// 累計出現次數
							codebook_cache[row][col][num_codeword][5] += 1;
							// 更新最大亮度
							if (brightness > codebook_cache[row][col][num_codeword][4])
								codebook_cache[row][col][num_codeword][4] = brightness;
							// 更新最小亮度
							if (brightness < codebook_cache[row][col][num_codeword][3])
								codebook_cache[row][col][num_codeword][3] = brightness;
							// 更新最久沒出現的次數
							if (times - codebook_cache[row][col][num_codeword][8] > codebook_cache[row][col][num_codeword][6])
								codebook_cache[row][col][num_codeword][6] = times - codebook_cache[row][col][num_codeword][8];
							// 更新最後一次出現的時間點
							codebook_cache[row][col][num_codeword][8] = times;
							break;
						}
					}
				}
			}


			// 若都沒有找到屬於新資料的codeword，則要在codebook_cache當中建立新的
			if (find_flag == 0) {

				// 初始化參數
				sum_of_square_BGR = 0;
				// 找每個codeword，尋找空的位置
				for (int num_codeword = 0; num_codeword < max_codeword; num_codeword++) {
					// 若該codeword為空的，則使用他，更新參數  v=[R,G,B],aux=<I_min,I_max,f,lamda,p,q> 
					if (codebook_cache[row][col][num_codeword][0] == 0 && codebook_cache[row][col][num_codeword][1] == 0 && codebook_cache[row][col][num_codeword][2] == 0) {
						find_flag = 1;          // 找到了

						//依序處理BGR
						for (int BGR = 0; BGR < 3; BGR++) {
							codebook_cache[row][col][num_codeword][BGR] = float(image.at<cv::Vec3b>(row, col)[BGR]);
							sum_of_square_BGR += pow(float(image.at<cv::Vec3b>(row, col)[BGR]),2);
						}
						brightness = sqrt(sum_of_square_BGR);
						codebook_cache[row][col][num_codeword][3] = brightness;
						codebook_cache[row][col][num_codeword][4] = brightness;
						codebook_cache[row][col][num_codeword][5] = 1;                // 第1次出現
						codebook_cache[row][col][num_codeword][6] = times - 1;        // 之前都沒出現
						codebook_cache[row][col][num_codeword][7] = times;
						codebook_cache[row][col][num_codeword][8] = times;
						break;
					}
				}
			}

			// 如果已經找不到空白的codeword
			if (find_flag == 0) {
				cout << "Sorry.. out of memory for pixel (" << row << "," << col << ")" << endl;
			}


			// 對於在codebook當中練續T_delete次沒出現的codeword，將他丟掉
			for (int num_codeword = 0; num_codeword < max_codeword; num_codeword++) {
				// 檢查該codeword是否存在
				if (codebook[row][col][num_codeword][0] != 0 && codebook[row][col][num_codeword][1] != 0 && codebook[row][col][num_codeword][2] != 0) {
					// 是否多次沒出現
					if (times - codebook[row][col][num_codeword][8] > T_delete) {

						// 將其參數全部清空
						for (int para_num = 0; para_num < 9; para_num++)
							codebook[row][col][num_codeword][para_num] = 0;
					}
				}
			}




			// 對於在codebook_cache當中練續T_delete次沒出現的codeword，將他丟掉
			for (int num_codeword = 0; num_codeword < max_codeword; num_codeword++) {
				// 檢查該codeword是否存在
				if (codebook_cache[row][col][num_codeword][0] != 0 && codebook_cache[row][col][num_codeword][1] != 0 && codebook_cache[row][col][num_codeword][2] != 0) {
					// 是否多次沒出現
					if (times - codebook_cache[row][col][num_codeword][8] > T_delete) {

						// 將其參數全部清空
						for (int para_num = 0; para_num < 9; para_num++)
							codebook_cache[row][col][num_codeword][para_num] = 0;
					}
				}
			}

			// 對於在codebook_cache當中累計出現夠多次的codeword，把它放到codebook
			// (num_codeword會指到能夠升級的codebook_cache編號，而num_codeword會指到空的codebook編號)
			for (int num_codeword = 0; num_codeword < max_codeword; num_codeword++) {
				// 檢查該codeword是否存在
				if (codebook_cache[row][col][num_codeword][0] != 0 && codebook_cache[row][col][num_codeword][1] != 0 && codebook_cache[row][col][num_codeword][2] != 0) {
					// 累計出現次數是否足夠
					if (codebook_cache[row][col][num_codeword][5] > T_add) {
						// 在codebook當中尋找空的位置，將其放入
						for (int num_codeword2 = 0; num_codeword2 < max_codeword; num_codeword2++) {
							// 找空的位置
							if (codebook[row][col][num_codeword2][0] == 0 && codebook[row][col][num_codeword2][1] == 0 && codebook[row][col][num_codeword2][2] == 0) {

								// 將其參數填入
								for (int para_num = 0; para_num < 9; para_num++)
									codebook[row][col][num_codeword2][para_num] = codebook_cache[row][col][num_codeword][para_num];
								break;
							}

						}
						// 將其參數全部清空
						for (int para_num = 0; para_num < 9; para_num++)
							codebook_cache[row][col][num_codeword][para_num] = 0;
					}
				}
			}


		}
	}
}





// 前景偵測
Mat Foreground_Detection(Mat image) {
	// parameter
	int find_flag = 0;                                   // 有無在codebook當中找到屬於他的codeword

	float sum_of_dot = 0;                // 暫存新資料與某cache的內積總和
	float sum_of_square_BGR = 0;         // 暫存新資料的BGR平方總和
	float brightness = 0;                // 暫存新資料的亮度
	float square_of_v = 0;               // 暫存某cache與原點的距離平方
	float distortion = 0;                // 暫存新資料與某cache的顏色差異
	float max_bound = 0;                 // 暫存某cache能容忍的最大亮度
	float min_bound = 0;                 // 暫存某cache能容忍的最小亮度 
	Mat image2(image_row, image_col, CV_8UC3, Scalar(255, 255, 255));   // 輸出圖形

	// 依序處理每個pixel
	for (int row = 0; row < image_row; row++) {
		for (int col = 0; col < image_col; col++) {
			// 初始化參數
			find_flag = 0;
			// 檢查每個codeword
			for (int num_codeword = 0; num_codeword < max_codeword; num_codeword++) {
				// 初始化參數
				sum_of_dot = 0;
				sum_of_square_BGR = 0;
				square_of_v = 0;
				// 若該codeword為空的，則換找下一個
				if (codebook[row][col][num_codeword][0] == 0 && codebook[row][col][num_codeword][1] == 0 && codebook[row][col][num_codeword][2] == 0)
					continue;
				// 若有東西，則計算與他的距離
				else {
					//依序計算BGR
					for (int BGR = 0; BGR < 3; BGR++) {
						sum_of_dot += float(image.at<cv::Vec3b>(row, col)[BGR]) * codebook[row][col][num_codeword][BGR];
						sum_of_square_BGR += pow(float(image.at<cv::Vec3b>(row, col)[BGR]), 2);
						square_of_v += pow(codebook[row][col][num_codeword][BGR], 2);
					}
					// 統計參數 [I_max*0.5<新資料亮度<min(I_max*1.2,I_min*2)]
					brightness = sqrt(sum_of_square_BGR);
					distortion = sqrt(sum_of_square_BGR - (pow(sum_of_dot, 2) / square_of_v));
					min_bound = 0.5 * codebook[row][col][num_codeword][4];
					max_bound = 1.2 * codebook[row][col][num_codeword][4];
					if (2 * codebook[row][col][num_codeword][3] < max_bound)
						max_bound = 2 * codebook[row][col][num_codeword][3];

					// 若[distortion小於epsilon]且[min_bound<新資料亮度<max_bound]，則屬於該codework
					if (distortion <= epsilon && min_bound < brightness && brightness < max_bound) {
						find_flag = 1;          // 找到了
						break;
					}
				}
			}


			// 若沒在codebook中找到，表示為前景(白色)
			if (find_flag == 0) {
				for (int BGR = 0; BGR < 3; BGR++) {
					image2.at<cv::Vec3b>(row, col)[BGR] = 255;
				}
			}
			// 後景(黑色)
			else {
				for (int BGR = 0; BGR < 3; BGR++) 
					image2.at<cv::Vec3b>(row, col)[BGR] = 0;
			}



		}
	}
	return image2;
}


void codebook_parameter(int pixel_x, int pixel_y) {
	for (int num_codeword = 0; num_codeword < max_codeword; num_codeword++) {
		cout << "The codeword No." << num_codeword << " : ";
		for (int num_para = 0; num_para < 9; num_para++)
			cout << codebook[pixel_x][pixel_y][num_codeword][num_para] << ", ";
		cout << endl;
	}
}

void codebook_cache_parameter(int pixel_x, int pixel_y) {
	for (int num_codeword = 0; num_codeword < max_codeword; num_codeword++) {
		cout << "The codeword_cache No." << num_codeword << " : ";
		for (int num_para = 0; num_para < 9; num_para++)
			cout << codebook_cache[pixel_x][pixel_y][num_codeword][num_para] << ", ";
		cout << endl;
	}
}


Mat median_filter(Mat image) {
	Mat image2(image_row, image_col, CV_8UC3, Scalar(0, 0, 0));   // 輸出圖形
	int num_of_white = 0;                                               // 這個filter掃到的白色(255)點數

	
	// 填充copy image
	for (int row = 0; row < image_row; row++) {
		for (int col = 0; col < image_col; col++) {
			if (int(image.at<cv::Vec3b>(row, col)[0]) == 255)       // 若為白色
				copy_image[row + 1][col + 1] = 1;                   // 則紀錄為1
			else
				copy_image[row + 1][col + 1] = 0;                   // 則紀錄為0
		}
	}

	
	for (int row = 0; row < image_row; row++) {
		for (int col = 0; col < image_col; col++) {
			num_of_white = copy_image[row][col]+ copy_image[row][col+1]+ copy_image[row][col+2];
			num_of_white += copy_image[row+1][col] + copy_image[row+1][col+1] + copy_image[row+1][col+2];
			num_of_white += copy_image[row+2][col] + copy_image[row+2][col+1] + copy_image[row+2][col+2];
			if (num_of_white >= 5) {    // 則選擇使用白色
				//依序處理BGR
				for (int BGR = 0; BGR < 3; BGR++) 
					image2.at<cv::Vec3b>(row, col)[BGR] = 255;

			}

		}
	}
	
	return image2;
}

Mat modify(Mat image) {
	Mat after;
	medianBlur(image, after, 13);
	Mat element = getStructuringElement(MORPH_RECT, Size(2, 2)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
	dilate(after, after, element);    //膨胀操作
	medianBlur(after, after, 7);
	medianBlur(after, after, 3);
	return after;
}

