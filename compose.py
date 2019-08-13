# coding=utf-8
# 头像特效合成
import cv2


def compose():
    # OpenCV人脸识别分类器
    classifier = cv2.CascadeClassifier(
        "D:\program\python\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
    )

    img = cv2.imread("img/zhuyilong.jpg")  # 读取图片
    imgCompose = cv2.imread("img/compose/maozi-1.png")

    #rgb色是3通道（shape中的第三个字段 channel），灰色是单通道，shape没有第三个字段，
    # rgb到灰色的转换，大概是有这么个函数 f（x,y,z）
    # 转换灰色
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray) 灰色系的图片
    # rgb里面的绿色
    color = (0, 255, 0)  # 定义绘制颜色
    # 调用识别人脸
    #返回长方形 有几张脸就取几个
    faceRects = classifier.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects):  # 大于0则检测到人脸
        #给大家的脸戴帽子
        for faceRect in faceRects:
            # 坐标  宽度 高度 是下面这样的关系
            # x y
            #
            #       (x+w,y+h)
            x, y, w, h = faceRect
            #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            sp = imgCompose.shape
            # sp[0]是高度，sp[1]是宽度
            imgComposeSizeH = int(sp[0] / sp[1] * w*2)
            # 如果按比例得出的高度大于到顶部的距离，那帽子的高度就是到顶部的距离，例如，在脸上面留白太少的情况下，会出现这个问题
            if imgComposeSizeH > (y - 20):
                imgComposeSizeH = (y - 20)
            imgComposeResized = cv2.resize(imgCompose, (w, imgComposeSizeH), interpolation=cv2.INTER_NEAREST)
            top = (y - imgComposeSizeH - 20)
            if top <= 0:
                top = 0
            rows, cols, channels = imgComposeResized.shape
            roi = img[top:top + rows, x:x + cols]

            #下面开始抠图，要把帽子抠下来， 还要把要抠图出来的矩形除帽子外的部分抠下来
            # Now create a mask of logo and create its inverse mask also
            #必须是灰色图像
            img2gray = cv2.cvtColor(imgComposeResized, cv2.COLOR_RGB2GRAY)
            # The function applies fixed-level thresholding to a multiple-channel array. The function is typically
            # .   used to get a bi-level (binary) image out of a grayscale image ( #compare could be also used for
            # .   this purpose) or for removing a noise, that is, filtering out pixels with too small or too large
            # .   values. There are several types of thresholding supported by the function. They are determined by
            # .   type parameter.
            # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
            #dst(x,y)={ if src(x,y)>thresh  maxval else 0
            # mask 周围是0，帽子是255 看THRESH_BINARY的含义就知道了
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            #周围是白的 255，帽子是黑的 0
            mask_inv = cv2.bitwise_not(mask)
            #cv2.imshow("mask_inv", mask_inv)

            # Now black-out the area of logo in ROI 把要放帽子的这个矩形 帽子周边留下，帽子黑掉 0掩码都为0,255掩码之后保留本色
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            cv2.imshow("img1_bg", img1_bg)

        # Take only region of logo from logo image.  只取帽子部分
            img2_fg = cv2.bitwise_and(imgComposeResized, imgComposeResized, mask=mask)
            cv2.imshow("img2_fg", img2_fg)


        # Put logo in ROI and modify the main image
            dst = cv2.add(img1_bg, img2_fg)
            cv2.imshow("dst", dst)
            img[top:top + rows, x:x + cols] = dst

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    compose()
