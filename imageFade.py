import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

class SimpleFadeApp:
    def __init__(self, root, filePath):
        self.root = root
        self.root.title("红色渐变编辑器")
        self.root.geometry("600x500")
        self.filePath = filePath

        # 初始值
        self.intensity = 0  # 默认50%
        self.target_red = None  # 将从图片四角获取
        
        self.create_widgets()
        
        if not os.path.isfile(self.filePath):
            messagebox.showerror("错误", f"文件不存在: {self.filePath}")
            self.root.destroy()
            return
        try:
            self.open_image()
        except Exception as e:
            messagebox.showerror("错误", f"无法打开图片: {str(e)}")
            self.root.destroy()
            return

    def create_widgets(self):
        # 顶部按钮
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(pady=10)
        
        # self.open_btn = tk.Button(self.top_frame, text="打开图片", command=self.open_image)
        # self.open_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = tk.Button(self.top_frame, text="保存图片", command=self.save_image, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # 强度控制
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady=10)
        
        tk.Label(self.control_frame, text="渐变强度:").pack(side=tk.LEFT, padx=5)
        
        self.intensity_label = tk.Label(self.control_frame, text=f"{self.intensity}%", width=5)
        self.intensity_label.pack(side=tk.LEFT, padx=5)
        
        self.slider = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, 
                               command=self.update_intensity, length=300)
        self.slider.set(self.intensity)
        self.slider.pack(pady=10)
        
        # 图片显示区域
        self.image_frame = tk.Frame(self.root, width=400, height=300, bg="gray")
        self.image_frame.pack(pady=20)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()
        
    def open_image(self):
        # file_path = filedialog.askopenfilename(
        #     filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        # )
        
        # if not file_path:
        #     return
            
        try:
            # 加载图片
            self.original_img = Image.open(self.filePath).convert('RGB')
            self.original_array = np.array(self.original_img)
            
            # 从四角获取目标红色（简单平均）
            h, w = self.original_array.shape[:2]
            corners = [
                self.original_array[0, 0],      # 左上
                self.original_array[0, w-1],    # 右上
                self.original_array[h-1, 0],    # 左下
                self.original_array[h-1, w-1]   # 右下
            ]
            
            # 计算平均红色值
            red_values = [corner[0] for corner in corners]
            self.target_red = int(np.mean(red_values))
            
            # 显示初始图像
            self.update_image()
            self.save_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("错误", f"无法打开图片: {str(e)}")
    
    def update_intensity(self, value):
        self.intensity = int(value)
        self.intensity_label.config(text=f"{value}%")
        if hasattr(self, 'original_img'):
            self.update_image()
    
    def update_image(self):
        if not hasattr(self, 'original_array'):
            return
            
        # 将强度从0-100转换为0-1
        t = self.intensity / 100.0
        
        # 转换为浮点数以便计算
        img_array = self.original_array.astype(np.float32)
        
        # 计算新图像
        # 红色通道：向目标红色过渡
        new_r = img_array[:, :, 0] * (1 - t) + self.target_red * t
        
        # 绿色和蓝色通道：逐渐减少
        new_g = img_array[:, :, 1] * (1 - t)
        new_b = img_array[:, :, 2] * (1 - t)
        
        # 组合并限制值范围
        result_array = np.stack([new_r, new_g, new_b], axis=2)
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        
        # 转换为PIL图像
        result_img = Image.fromarray(result_array)
        
        # 调整大小以适应显示区域
        display_img = result_img.copy()
        display_img.thumbnail((380, 280))
        
        # 转换为Tkinter兼容格式
        self.tk_image = ImageTk.PhotoImage(display_img)
        self.image_label.config(image=self.tk_image)
        
        # 保存当前处理结果
        self.current_result = result_img
    
    def save_image(self):
        if not hasattr(self, 'current_result'):
            messagebox.showwarning("警告", "请先打开图片")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.current_result.save(file_path)
                messagebox.showinfo("成功", f"图片已保存到:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")

# 运行应用
if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleFadeApp(root, r'E:\Unity\LEDRingTest\Assets\Resources\Backgrounds\backgroundWith4pattern.png')
    root.mainloop()