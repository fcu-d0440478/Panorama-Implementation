使用 python 撰寫全景影像的拼接(不使用套件)
Write panorama image stitching using Python (without using packages).

以下是 PanoramaFor5.py 程式碼的介紹

# Image Stitching Project

這個專案主要是用來進行圖像拼接，包括特徵點的檢測、匹配和仿射變換。

## 程式碼結構

主要的程式碼包含以下幾個部分：

1. `translateToBlackImg(x, y)`：這個函數用來將座標轉換到一張大的黑色背景圖像上。

2. `SIFT(imgA, imgB)`：這個函數用來進行 SIFT 特徵點檢測和描述子計算。

3. `findNearest(target_des, second_des_arr)`：這個函數用來找出與目標描述子最接近的描述子。

4. `get_suitable_point(kp1, des1, kp2, des2)`：這個函數用來從所有的特徵點中選擇出可靠的特徵點。

5. `get_random_points(reliable_points, kp1 ,kp2)`：這個函數用來從可靠的特徵點中隨機選擇四個點。

6. `calculate_affine_matrix(source_point, destination_point)`：這個函數用來計算仿射變換矩陣。

7. `stich_image(imgA, imgB)`：這個函數用來進行圖像拼接。它首先使用 SIFT 來檢測和匹配特徵點，然後選擇可靠的特徵點，並計算仿射變換矩陣。

8. 圖像讀取：程式碼會讀取指定路徑下的圖像檔案。

9. 圖像拼接：程式碼會將讀取的圖像拼接到一個大的黑色背景圖像上。

## 使用方法

1. 確保您的電腦已經安裝了必要的 Python 套件，包括 `numpy`、`cv2` 和 `matplotlib.pyplot`。

2. 將您想要拼接的圖像檔案放到指定的路徑下。

3. 在命令提示字元中，切換到程式碼所在的目錄，並執行程式碼。

## 注意事項

這個程式碼目前只支援兩張圖像的拼接。如果您想要拼接更多的圖像，您可能需要修改程式碼以支援這種功能。

此外，這個程式碼並未包含錯誤處理機制。如果在執行過程中遇到問題，程式可能會直接終止。請確保您的圖像檔案和程式碼都是正確的，並且您的電腦已經正確安裝了所有必要的套件。
