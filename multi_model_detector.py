import cv2
import numpy as np
import onnxruntime as ort
import os
from tqdm import tqdm
from typing import Tuple, List

class LineIntersector:
    def __init__(self):
        """初始化交点计算器"""
        pass

    def get_line_params(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[float, float]:
        """
        计算直线参数 (k, b)，y = kx + b
        使用图像坐标系（左上角为原点，向右为x正方向，向下为y正方向）
        """
        x1, y1 = p1
        x2, y2 = p2
        
        # 处理垂直线的情况
        if x2 - x1 == 0:
            return float('inf'), x1
        
        # 计算斜率和截距
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        return k, b

    def find_intersection(self, line1: Tuple[Tuple[int, int], Tuple[int, int]], 
                         line2: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[int, int]:
        """计算两条直线的交点"""
        k1, b1 = self.get_line_params(*line1)
        k2, b2 = self.get_line_params(*line2)
        
        # 处理平行线的情况
        if k1 == k2:
            return None
        
        # 处理垂直线的情况
        if k1 == float('inf'):
            x = b1
            y = k2 * x + b2
        elif k2 == float('inf'):
            x = b2
            y = k1 * x + b1
        else:
            # 计算交点
            x = (b2 - b1) / (k1 - k2)
            y = k1 * x + b1
        
        return (int(x), int(y))

    def draw_intersection(self, img: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
        """绘制直线和交点"""
        if len(points) != 4:
            raise ValueError("需要4个点的坐标")
        
        # 获取两条直线
        line1 = (points[0], points[1])
        line2 = (points[2], points[3])
        
        # 计算交点
        intersection = self.find_intersection(line1, line2)
        if intersection is None:
            print("直线平行，无交点")
            return img
        
        # 绘制延长线
        img_height, img_width = img.shape[:2]
        
        # 计算直线1的延长线端点
        k1, b1 = self.get_line_params(*line1)
        if k1 != float('inf'):
            x1, x2 = 0, img_width
            y1 = int(k1 * x1 + b1)
            y2 = int(k1 * x2 + b1)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
        else:
            x = line1[0][0]
            cv2.line(img, (x, 0), (x, img_height), (0, 255, 255), 1)
        
        # 计算直线2的延长线端点
        k2, b2 = self.get_line_params(*line2)
        if k2 != float('inf'):
            x1, x2 = 0, img_width
            y1 = int(k2 * x1 + b2)
            y2 = int(k2 * x2 + b2)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
        else:
            x = line2[0][0]
            cv2.line(img, (x, 0), (x, img_height), (0, 255, 255), 1)
        
        # 绘制交点
        cv2.circle(img, intersection, 5, (0, 0, 255), -1)
        cv2.putText(img, f"Intersection({intersection[0]},{intersection[1]})", 
                   (intersection[0]+10, intersection[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return img

class MultiModelDetector:
    def __init__(self, model_configs):
        """
        初始化多模型检测器
        Args:
            model_configs: 模型配置列表
        """
        self.models = []
        self.model_info = {}  # 存储模型信息
        
        print("正在加载模型...")
        # 加载所有模型并提取信息
        for i, config in enumerate(model_configs):
            print(f"\n加载模型 {i+1}:")
            print(f"路径: {config['model_path']}")
            print(f"类别: {config['class_names']}")
            print(f"关键点数量: {len(config['point_colors'])}")
            
            model = {
                'session': ort.InferenceSession(
                    config['model_path'],
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"] 
                    if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
                ),
                'class_names': config['class_names'],
                'confidence_thres': config.get('confidence_thres', 0.7),
                'iou_thres': config.get('iou_thres', 0.45),
                'point_colors': config.get('point_colors', 
                    [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
                ),
                'color_palette': np.random.uniform(0, 255, size=(len(config['class_names']), 3))
            }
            
            # 获取模型输入尺寸
            model_inputs = model['session'].get_inputs()
            input_shape = model_inputs[0].shape
            model['input_width'] = input_shape[2]
            model['input_height'] = input_shape[3]
            
            # 存储模型信息
            self.model_info[i] = {
                'type': list(config['class_names'].values())[0],
                'num_points': len(config['point_colors']),
                'input_shape': input_shape,
            }
            
            self.models.append(model)
        
        print("\n模型加载完成!")
        print("\n模型信息汇总:")
        for idx, info in self.model_info.items():
            print(f"模型 {idx}:")
            print(f"  类型: {info['type']}")
            print(f"  关键点数量: {info['num_points']}")
            print(f"  输入尺寸: {info['input_shape']}")

    def add_black_border(self, img, target_width=802, target_height=530):
        """添加黑色边框"""
        h, w = img.shape[:2]
        black_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x_offset = (target_width - w) // 2
        y_offset = (target_height - h) // 2
        black_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
        return black_img, (x_offset, y_offset)

    def preprocess(self, image_path, model):
        """图像预处理"""
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        self.img, (self.x_offset, self.y_offset) = self.add_black_border(self.img)
        self.img_height, self.img_width = self.img.shape[:2]
        
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img, self.ratio, (self.dw, self.dh) = self.letterbox(
            img, 
            new_shape=(model['input_width'], model['input_height'])
        )
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), 
                 auto=False, scaleFill=False, scaleup=True):
        """调整图像大小并填充"""
        shape = img.shape[:2]  # 当前形状 [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # 只缩小，不放大
            r = min(r, 1.0)
        
        # 计算新的未填充尺寸
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # 分割填充以保持中心
        dh /= 2
        
        # 调整图像大小
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        # 添加边框填充
        top, bottom = int(np.floor(dh)), int(np.ceil(dh))
        left, right = int(np.floor(dw)), int(np.ceil(dw))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, (r, r), (dw, dh)

    def draw_detections(self, img, box, score, class_id, keypoints, model):
        """绘制检测结果"""
        x1, y1, w, h = box
        color = model['color_palette'][class_id]
        
        # 绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        
        # 绘制类别标签
        label = f"{model['class_names'][class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(
            img, (label_x, label_y - label_height),
            (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )
        cv2.putText(
            img, label, (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

        # 绘制关键点和标签
        for i, kpt in enumerate(keypoints):
            x, y, conf = kpt
            if conf > model['confidence_thres']:
                x, y = int(x), int(y)
                cv2.circle(img, (x, y), 4, model['point_colors'][i], -1)
                cv2.putText(
                    img, f"{i+1}({x},{y})", (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, model['point_colors'][i], 1
                )

    def process_image(self, image_path):
        """处理单张图片并返回最佳检测结果"""
        best_result = None
        best_confidence = -1
        best_model_index = -1
        all_results = {}  # 存储所有模型的检测结果
        
        # 对每个模型进行检测
        for i, model in enumerate(self.models):
            try:
                img_data = self.preprocess(image_path, model)
                outputs = model['session'].run(
                    None, 
                    {model['session'].get_inputs()[0].name: img_data}
                )
                
                # 获取检测结果的置信度
                output_data = np.transpose(np.squeeze(outputs[0]))
                if len(output_data) > 0:
                    max_confidence = np.max(output_data[:, 4:5])
                    
                    # 获取检测结果和坐标
                    result_image, coordinates = self.postprocess(self.img.copy(), outputs, model)
                    
                    # 存储当前模型的检测结果
                    all_results[i] = {
                        'confidence': max_confidence,
                        'result': result_image,
                        'coordinates': coordinates
                    }
                    
                    # 更新最佳结果
                    if max_confidence > best_confidence:
                        best_confidence = max_confidence
                        best_model_index = i
                        best_result = result_image
                
            except Exception as e:
                print(f"模型 {i} 处理出错: {str(e)}")
                continue
        
        return best_result, best_model_index, best_confidence, all_results

    def process_directory(self, dir_path):
        """处理目录中的所有图片"""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(dir_path) 
                      if f.lower().endswith(image_extensions)]
        print(f"找到 {len(image_files)} 个图片文件")

        # 创建输出目录（只用于存放CSV文件）
        output_dir = os.path.join(dir_path, 'detected_best')
        os.makedirs(output_dir, exist_ok=True)

        # 创建CSV文件
        csv_files = []
        for i, model in enumerate(self.models):
            model_type = self.model_info[i]['type']
            num_points = self.model_info[i]['num_points']
            csv_path = os.path.join(output_dir, f'{model_type}_coordinates.csv')
            # 创建CSV文件并写入表头
            with open(csv_path, 'w', encoding='utf-8') as f:
                headers = ['image_name']
                for p in range(num_points):
                    headers.extend([f'point{p+1}_x', f'point{p+1}_y'])
                # 如果是A类型，添加交点坐标列
                if model_type == 'A':
                    headers.extend(['intersection_x', 'intersection_y'])
                f.write(','.join(headers) + '\n')
            csv_files.append(csv_path)

        # 创建LineIntersector实例
        intersector = LineIntersector()
        
        # 处理每张图片
        for img_file in tqdm(image_files, desc="处理进度", unit="张"):
            try:
                image_path = os.path.join(dir_path, img_file)
                
                # 获取所有检测结果
                best_result, best_model_index, best_confidence, all_results = self.process_image(image_path)
                
                if best_result is not None:
                    # 只处理最佳匹配模型的结果
                    intersection = None
                    if self.model_info[best_model_index]['type'] == 'A':
                        coordinates = all_results[best_model_index].get('coordinates', [])
                        if len(coordinates) == 4:
                            try:
                                intersection = intersector.find_intersection(
                                    (coordinates[0], coordinates[1]),
                                    (coordinates[2], coordinates[3])
                                )
                            except Exception as e:
                                print(f"\n处理图片 {img_file} 的交点时出错: {str(e)}")
                    
                    # 只记录最佳匹配模型的结果到对应的CSV文件
                    coordinates = all_results[best_model_index].get('coordinates', [])
                    if coordinates:
                        csv_path = csv_files[best_model_index]
                        with open(csv_path, 'a', encoding='utf-8') as f:
                            values = [img_file]
                            for x, y in coordinates:
                                values.extend([str(x), str(y)])
                            # 如果是A类型且有交点，添加交点坐标
                            if self.model_info[best_model_index]['type'] == 'A':
                                if intersection:
                                    values.extend([str(intersection[0]), str(intersection[1])])
                                else:
                                    values.extend(['', ''])
                            f.write(','.join(values) + '\n')
                
            except Exception as e:
                print(f"\n处理图片 {img_file} 时出错: {str(e)}")
                continue

        print(f"\n处理完成! CSV文件保存在:")
        for csv_file in csv_files:
            print(f"  - {csv_file}")

    def extract_coordinates(self, result_image):
        """从结果图像中提取坐标点信息"""
        try:
            # 使用OCR或其他方法从图像中提取坐标
            # 这里需要实现具体的提取逻辑
            # 返回格式: [(x1,y1), (x2,y2), ...]
            return []
        except Exception as e:
            print(f"提取坐标时出错: {str(e)}")
            return None

    def postprocess(self, input_image, output, model):
        """后处理检测结果"""
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []
        keypoints_list = []
        coordinates = []  # 存储坐标信息

        # 获取模型的关键点数量
        num_keypoints = len(model['point_colors'])

        for i in range(rows):
            classes_scores = outputs[i][4:5]
            max_score = np.amax(classes_scores)
            
            if max_score >= model['confidence_thres']:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0:4]
                
                # 调整边界框坐标
                x -= self.dw
                y -= self.dh
                x /= self.ratio[0]
                y /= self.ratio[1]
                w /= self.ratio[0]
                h /= self.ratio[1]
                
                left = int(x - w / 2)
                top = int(y - h / 2)
                width = int(w)
                height = int(h)
                
                # 处理关键点
                kpts = []
                point_coords = []  # 存储当前检测的坐标
                for j in range(num_keypoints):
                    kx = outputs[i][5 + j * 3]
                    ky = outputs[i][6 + j * 3]
                    kconf = outputs[i][7 + j * 3]
                    
                    # 调整关键点坐标
                    kx = (kx - self.dw) / self.ratio[0]
                    ky = (ky - self.dh) / self.ratio[1]
                    
                    kpts.append([kx, ky, kconf])
                    point_coords.append((int(kx), int(ky)))
                
                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)
                keypoints_list.append(kpts)
                coordinates.append(point_coords)

        indices = cv2.dnn.NMSBoxes(boxes, scores, model['confidence_thres'], model['iou_thres'])
        
        final_coords = []
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            keypoints = keypoints_list[i]
            final_coords = coordinates[i]  # 保存最终使用的坐标
            self.draw_detections(input_image, box, score, class_id, keypoints, model)
        
        return input_image, final_coords

if __name__ == "__main__":
    # 模型配置示例
    model_configs = [
        {
            'model_path': r"pt/A.onnx",
            'class_names': {0: 'A'},
            'confidence_thres': 0.7,
            'iou_thres': 0.45,
            'point_colors': [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # 4个关键点
        },
        {
            'model_path': r"pt/C.onnx",
            'class_names': {0: 'C'},
            'confidence_thres': 0.7,
            'iou_thres': 0.45,
            'point_colors': [(0, 255, 0), (255, 0, 0)]  # 2个关键点
        },
        {
            'model_path': r"pt/D.onnx",
            'class_names': {0: 'D'},
            'confidence_thres': 0.7,
            'iou_thres': 0.45,
            'point_colors': [(0, 255, 0), (255, 0, 0)]  # 2个关键点
        },
        {
            'model_path': r"pt/JB.onnx",  # 新增的JB模型
            'class_names': {0: 'JB'},
            'confidence_thres': 0.7,
            'iou_thres': 0.45,
            'point_colors': [(0, 255, 0), (255, 0, 0)]  # 2个关键点
        }
    ]
    
    # 创建检测器实例
    detector = MultiModelDetector(model_configs)
    
    # 修改为当前目录下的test文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(current_dir, 'test')
    
    # 确保test文件夹存在
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        print(f"创建测试图片目录: {image_dir}")
    
    detector.process_directory(image_dir) 