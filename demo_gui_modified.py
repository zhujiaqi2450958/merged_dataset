# -*- coding: utf-8 -*-
# demo_gui.py
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================== RetinexNet Model ======================
class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3, padding=4, padding_mode='replicate')
        self.net1_convs = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(), 
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'), nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'), nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'), nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'), nn.ReLU()
        )
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size, padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)
        feats0 = self.net1_conv0(input_img)
        featss = self.net1_convs(feats0)
        outs = self.net1_recon(featss)
        R = torch.sigmoid(outs[:, 0:3, :, :])
        L = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L


class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()
        self.relu = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(4, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')

        self.net2_deconv1_1 = nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_deconv1_2 = nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_deconv1_3 = nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='replicate')

        self.net2_fusion = nn.Conv2d(channel*3, channel, kernel_size=1, padding=1, padding_mode='replicate')
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=0)

    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1)
        out0 = self.net2_conv0_1(input_img)
        out1 = self.relu(self.net2_conv1_1(out0))
        out2 = self.relu(self.net2_conv1_2(out1))
        out3 = self.relu(self.net2_conv1_3(out2))

        out3_up = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))
        deconv1 = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        deconv1_up = F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))
        deconv2 = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2_up = F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))
        deconv3 = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))

        deconv1_rs = F.interpolate(deconv1, size=(input_R.size()[2], input_R.size()[3]))
        deconv2_rs = F.interpolate(deconv2, size=(input_R.size()[2], input_R.size()[3]))
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        feats_fus = self.net2_fusion(feats_all)
        output = self.net2_output(feats_fus)
        return output


class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()
        self.DecomNet = DecomNet()
        self.RelightNet = RelightNet()

    def load(self, ckpt_dir):
        try:
            for phase in ['Decom', 'Relight']:
                load_dir = os.path.join(ckpt_dir, phase)
                if os.path.exists(load_dir):
                    ckpts = [f for f in os.listdir(load_dir) if f.endswith('.tar')]
                    if ckpts:
                        ckpts.sort()
                        ckpt_path = os.path.join(load_dir, ckpts[-1])
                        state_dict = torch.load(ckpt_path, map_location='cpu')
                        if phase == 'Decom':
                            self.DecomNet.load_state_dict(state_dict)
                        else:
                            self.RelightNet.load_state_dict(state_dict)
                        print(f"Loaded {phase} weights")
            return True
        except Exception as e:
            print(f"Weight load failed: {e}")
            return False

    def predict(self, img_path, output_path):
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img, dtype="float32") / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))
            input_low = np.expand_dims(img_np, axis=0)

            input_tensor = torch.FloatTensor(input_low)

            with torch.no_grad():
                R_low, I_low = self.DecomNet(input_tensor)
                I_delta = self.RelightNet(I_low, R_low)
                S = R_low * I_delta.repeat(1, 3, 1, 1)

            enhanced = S.squeeze(0).permute(1, 2, 0).cpu().numpy()
            enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
            
            cv2.imwrite(output_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
            return enhanced
        except Exception as e:
            print(f"Enhancement failed: {e}")
            return None


# ====================== Main GUI ======================
class PedestrianDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Pedestrian Detection System")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        self.model = None
        self.enhancer = None
        self.load_model()
        self.load_enhancer()
        
        self.current_img_path = None
        self.current_result_img = None

        if hasattr(self, "orig_photo"):
            del self.orig_photo
        if hasattr(self, "enhanced_photo"):
            del self.enhanced_photo
        if hasattr(self, "result_photo"):
            del self.result_photo

        self.create_widgets()

    def load_model(self):
        model_path = r"yolov8n.pt"
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model not found:\n{model_path}\n\nPlease check the path!")
            self.root.quit()
            return
        try:
            self.model = YOLO(model_path)
            print("YOLOv8 model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Model load failed: {e}")
            self.root.quit()

    def load_enhancer(self):
        ckpt_dir = "./ckpts"
        if os.path.exists(ckpt_dir):
            self.enhancer = RetinexNet()
            if self.enhancer.load(ckpt_dir):
                print("RetinexNet enhancer loaded successfully")
            else:
                self.enhancer = None
        else:
            print("No ckpts folder found - enhancement disabled")
            self.enhancer = None

    def create_widgets(self):
        title_frame = tk.Frame(self.root, bg="#2c3e50")
        title_frame.pack(fill=tk.X)
        tk.Label(title_frame, text="Pedestrian Detection System", font=("Arial", 22, "bold"), fg="white", bg="#2c3e50", pady=15).pack()
        tk.Label(title_frame, text="YOLOv8 + Low-light Enhancement", font=("Arial", 10), fg="#ecf0f1", bg="#2c3e50").pack()

        btn_frame = tk.Frame(self.root, bg="#ecf0f1", pady=15)
        btn_frame.pack(fill=tk.X)
        tk.Button(btn_frame, text="Select Image", font=("Arial", 12, "bold"), command=self.select_image, bg="#3498db", fg="white", padx=25, pady=8).pack(side=tk.LEFT, padx=20)
        tk.Button(btn_frame, text="Clear", font=("Arial", 12, "bold"), command=self.clear_image, bg="#95a5a6", fg="white", padx=25, pady=8).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Save Result", font=("Arial", 12, "bold"), command=self.save_result, bg="#27ae60", fg="white", padx=25, pady=8).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Exit", font=("Arial", 12, "bold"), command=self.root.quit, bg="#e74c3c", fg="white", padx=25, pady=8).pack(side=tk.RIGHT, padx=20)

        img_frame = tk.Frame(self.root, bg="#34495e")
        img_frame.pack(pady=10, padx=10, expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(img_frame, bg="#34495e")
        scrollbar = tk.Scrollbar(img_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.image_frame = tk.Frame(self.canvas, bg="#34495e")
        self.canvas.create_window((0, 0), window=self.image_frame, anchor=tk.NW)
        self.image_frame.bind("<Configure>", self.on_frame_configure)

        self.image_label = tk.Label(self.image_frame, text="Click 'Select Image' to start detection", 
                                    font=("Arial", 14), bg="#34495e", fg="#bdc3c7", width=70, height=20)
        self.image_label.pack(pady=50)

        result_frame = tk.Frame(self.root, bg="#ecf0f1", pady=15)
        result_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.result_label = tk.Label(result_frame, text="", font=("Arial", 14, "bold"), bg="#ecf0f1", fg="#2c3e50")
        self.result_label.pack()
        self.detail_label = tk.Label(result_frame, text="", font=("Arial", 10), bg="#ecf0f1", fg="#7f8c8d")
        self.detail_label.pack()

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def judge_low_light(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return False, 0, 0, 0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        avg_bright = np.mean(hsv[..., 2])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        dark_ratio = np.sum(gray < 100) / gray.size
        satisfied = sum([avg_bright < 100, contrast < 40, dark_ratio > 0.55])
        return satisfied >= 2, avg_bright, contrast, dark_ratio

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", 
                                             filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return
        self.result_label.config(text="Processing...", fg="#f39c12")
        self.root.update()
        self.detect(file_path)

    def detect(self, img_path):
        try:
            is_low, bright, cont, dark_r = self.judge_low_light(img_path)
            img_to_detect = img_path
            status = f"Normal Light | Brightness: {bright:.1f}"

            if is_low and self.enhancer:
                self.result_label.config(text="Low light detected, enhancing...", fg="#f39c12")
                self.root.update()
                
                enhanced_path = os.path.splitext(img_path)[0] + "_enhanced.jpg"
                enhanced_np = self.enhancer.predict(img_path, enhanced_path)
                if enhanced_np is not None:
                    img_to_detect = enhanced_path
                    status = f"ENHANCED | Original Brightness: {bright:.1f}"

            results = self.model(img_to_detect)
            boxes = results[0].boxes
            num_people = len(boxes) if boxes is not None else 0

            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            h, w = result_img.shape[:2]
            scale = min(950 / w, 650 / h)
            result_img = cv2.resize(result_img, (int(w * scale), int(h * scale)))

            # ================= 显示图片 =================
            for widget in self.image_frame.winfo_children():
                widget.destroy()

            display_frame = tk.Frame(self.image_frame, bg="#34495e")
            display_frame.pack(pady=10)

            # 原图
            orig_img = cv2.imread(img_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

            h0, w0 = orig_img.shape[:2]
            scale0 = min(350 / w0, 350 / h0)

            orig_img_show = cv2.resize(
                orig_img,
                (int(w0 * scale0), int(h0 * scale0))
            )

            orig_pil = Image.fromarray(orig_img_show)
            self.orig_photo = ImageTk.PhotoImage(orig_pil)

            orig_frame = tk.Frame(display_frame, bg="#34495e")
            orig_frame.pack(side=tk.LEFT, padx=15)

            tk.Label(orig_frame, text="Original Image",
                     font=("Arial", 12, "bold"),
                     bg="#34495e", fg="white").pack()

            tk.Label(orig_frame,
                     image=self.orig_photo,
                     bg="#34495e").pack()

            # 增强图（仅低光）
            if is_low and self.enhancer and img_to_detect != img_path:

                enhanced_show = cv2.imread(img_to_detect)
                enhanced_show = cv2.cvtColor(
                    enhanced_show,
                    cv2.COLOR_BGR2RGB
                )

                h1, w1 = enhanced_show.shape[:2]
                scale1 = min(350 / w1, 350 / h1)

                enhanced_show = cv2.resize(
                    enhanced_show,
                    (int(w1 * scale1), int(h1 * scale1))
                )

                enhanced_pil = Image.fromarray(enhanced_show)
                self.enhanced_photo = ImageTk.PhotoImage(enhanced_pil)

                enhanced_frame = tk.Frame(display_frame, bg="#34495e")
                enhanced_frame.pack(side=tk.LEFT, padx=15)

                tk.Label(enhanced_frame,
                         text="Retinex Enhanced",
                         font=("Arial", 12, "bold"),
                         bg="#34495e",
                         fg="white").pack()

                tk.Label(enhanced_frame,
                         image=self.enhanced_photo,
                         bg="#34495e").pack()

            # 检测结果
            result_pil = Image.fromarray(result_img)
            self.result_photo = ImageTk.PhotoImage(result_pil)

            detect_frame = tk.Frame(display_frame, bg="#34495e")
            detect_frame.pack(side=tk.LEFT, padx=15)

            tk.Label(detect_frame,
                     text="Detection Result",
                     font=("Arial", 12, "bold"),
                     bg="#34495e",
                     fg="white").pack()

            tk.Label(detect_frame,
                     image=self.result_photo,
                     bg="#34495e").pack()

            self.result_label.config(text=f"{status}\nDetected {num_people} people", fg="#27ae60")
            if num_people > 0:
                confs = [f"{b.conf.item():.1%}" for b in boxes]
                self.detail_label.config(text="Confidence: " + ", ".join(confs))
            else:
                self.detail_label.config(text="No person detected")

            self.current_img_path = img_path
            self.current_result_img = result_img

        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.result_label.config(text="Detection Failed", fg="#e74c3c")

    def clear_image(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        self.image_label = tk.Label(self.image_frame, text="Click 'Select Image' to start detection", 
                                    font=("Arial", 14), bg="#34495e", fg="#bdc3c7", width=70, height=20)
        self.image_label.pack(pady=50)
        self.result_label.config(text="")
        self.detail_label.config(text="")
        self.current_img_path = None
        self.current_result_img = None

    def save_result(self):
        if self.current_result_img is None:
            messagebox.showwarning("Warning", "No result to save")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", 
                                                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if save_path:
            save_img = cv2.cvtColor(self.current_result_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            messagebox.showinfo("Success", f"Saved to:\n{save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PedestrianDetector(root)
    root.mainloop()