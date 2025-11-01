import math
import random
import cv2
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector

# 1. 初始化摄像头（1280x720分辨率）与手部检测器（单只手检测，平衡精度与速度）
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# 2. 贪吃蛇游戏核心类（适配Python 3.10：解决坐标类型错误、提升健壮性）
class SnakeGameClass:
    def __init__(self, pathFood):
        # 蛇身属性：坐标均为整数类型，避免绘图报错
        self.points = []            # 蛇身所有坐标点
        self.lengths = []           # 相邻点距离
        self.currentLength = 0      # 当前总长度
        self.allowedLength = 150    # 初始允许最大长度
        self.previousHead = (0, 0)  # 上一帧蛇头坐标

        # 食物属性：兼容图片缺失场景（缺失时生成红色方块）
        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        if self.imgFood is None:
            self.imgFood = np.zeros((50, 50, 4), dtype=np.uint8)
            self.imgFood[:, :, 2] = 255  # R通道（红色）
            self.imgFood[:, :, 3] = 255  # Alpha通道（不透明）
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = (0, 0)
        self.randomFoodLocation()

        # 游戏状态
        self.score = 0
        self.gameOver = False

    def randomFoodLocation(self):
        # 限制食物在屏幕可视范围内（避免超出1280x720界面）
        self.foodPoint = (random.randint(100, 1000), random.randint(100, 600))

    def update(self, imgMain, currentHead):
        # 关键：强制蛇头为整数元组，解决cv2.pointPolygonTest参数错误
        currentHead = (int(currentHead[0]), int(currentHead[1]))
        cx, cy = currentHead

        if self.gameOver:
            # 游戏结束提示
            cvzone.putTextRect(imgMain, "Game Over", [300, 400], scale=7, thickness=5, offset=20, textColor=(255,255,255), colorR=(255,0,0))
            cvzone.putTextRect(imgMain, f'Your Score: {self.score}', [300, 550], scale=7, thickness=5, offset=20, textColor=(255,255,0), colorR=(0,0,0))
        else:
            px, py = self.previousHead
            self.points.append([int(cx), int(cy)])  # 存储整数坐标
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            self.currentLength += distance
            self.previousHead = currentHead

            # 蛇身长度收缩：超过允许长度时，从尾部删除点
            if self.currentLength > self.allowedLength:
                for i in range(len(self.lengths)):
                    self.currentLength -= self.lengths[i]
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.currentLength <= self.allowedLength:
                        break

            # 食物碰撞检测：基于食物中心±半宽/高判定
            rx, ry = self.foodPoint
            if (rx - self.wFood//2 < cx < rx + self.wFood//2) and (ry - self.hFood//2 < cy < ry + self.hFood//2):
                self.randomFoodLocation()
                self.allowedLength += 50  # 吃食物后蛇身延长
                self.score += 1
                print(f"当前分数：{self.score}")

            # 绘制蛇身（线段）与蛇头（紫色圆）
            if self.points:
                for i in range(1, len(self.points)):
                    prev_pt = (self.points[i-1][0], self.points[i-1][1])
                    curr_pt = (self.points[i][0], self.points[i][1])
                    cv2.line(imgMain, prev_pt, curr_pt, (0, 0, 255), 20)
                head_pt = (self.points[-1][0], self.points[-1][1])
                cv2.circle(imgMain, head_pt, 20, (200, 0, 200), cv2.FILLED)

            # 绘制食物（透明叠加）与分数
            imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood//2, ry - self.hFood//2))
            cvzone.putTextRect(imgMain, f'Your Score: {self.score}', [50, 80], scale=3, thickness=5, offset=10, textColor=(255,255,255), colorR=(0,255,0))

            # 蛇身碰撞检测：蛇身长度>5时才检测，避免初始误判
            if len(self.points) > 5:
                # 转换为OpenCV要求的np.int32格式
                pts = np.array(self.points[:-2], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(imgMain, [pts], False, (0, 200, 0), 3)
                minDist = cv2.pointPolygonTest(pts, (cx, cy), True)
                if -1 <= minDist <= 1:  # 距离在[-1,1]表示碰撞
                    self.gameOver = True
                    # 重置游戏状态
                    self.points, self.lengths, self.currentLength = [], [], 0
                    self.allowedLength, self.previousHead = 150, (0, 0)
                    self.randomFoodLocation()

        return imgMain

# 3. 游戏初始化与主循环
if __name__ == "__main__":
    # 初始化游戏：兼容食物图片缺失场景
    try:
        game = SnakeGameClass("donut.png")
    except Exception as e:
        print(f"初始化游戏失败：{e}，使用默认食物图片")
        game = SnakeGameClass("default_food.png")

    while True:
        # 读取摄像头图像：失败时重新连接
        success, img = cap.read()
        if not success:
            print("摄像头读取失败，重试...")
            cap = cv2.VideoCapture(0)
            continue

        img = cv2.flip(img, 1)  # 水平翻转，操作更直观
        hands, img = detector.findHands(img, flipType=False)

        # 检测到手时，用食指指尖控制蛇头
        if hands:
            lmList = hands[0]['lmList']
            pointIndex = (lmList[8][0], lmList[8][1])  # 食指指尖（索引8）
            img = game.update(img, pointIndex)
        else:
            # 未检测到手时显示提示
            cvzone.putTextRect(img, "请伸出食指控制蛇头", [400, 350], scale=3, thickness=3, offset=10, textColor=(255,255,255), colorR=(0,0,255))

        cv2.imshow("Hand-Controlled Snake Game", img)

        # 按键控制：R重置游戏，Q/ESC退出
        key = cv2.waitKey(1)
        if key == ord('r'):
            game.gameOver = False
            game.score, game.points, game.lengths = 0, [], []
            game.currentLength, game.allowedLength = 0, 150
            game.previousHead = (0, 0)
            game.randomFoodLocation()
        elif key == ord('q') or key == 27:
            break

    # 释放资源，避免内存泄漏
    cap.release()
    cv2.destroyAllWindows()

