import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import cv2
import numpy as np
from model.flol import create_model
from options.options import parse

def pad_tensor(tensor, multiple=8):
    '''Thêm viền (pad) để kích thước ảnh chia hết cho 8 (yêu cầu của mô hình)'''
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
    return tensor

def main(opt):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Đang chạy trên thiết bị: {device} ---")

    # Khởi tạo và Load Model
    print("Đang tải mô hình...")
    model = create_model()
    
    # Lấy đường dẫn file weight từ file config
    weights_path = opt['settings']['weight']
    
    # Load trọng số (Weights)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['params'])
    model.to(device)
    model.eval() 
    print("Tải mô hình thành công!")

    # Webcam
    cap = cv2.VideoCapture(1) 
    if not cap.isOpened():
        print("Không thể mở Webcam. Hãy kiểm tra lại kết nối!")
        return

    to_tensor = transforms.ToTensor()

    print("Đang chạy Real-time. Nhấn phím 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame nhỏ lại nếu máy chạy chậm
        # frame = cv2.resize(frame, (640, 480))
        
        # --- TIỀN XỬ LÝ ---
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        img_tensor = to_tensor(img_rgb).unsqueeze(0).to(device)
        
        _, _, H, W = img_tensor.shape
        
        img_padded = pad_tensor(img_tensor)

        # --- CHẠY AI  ---
        with torch.no_grad():
            output = model(img_padded)

        # --- HẬU XỬ LÝ ---
        output = torch.clamp(output, 0., 1.)
        
        output = output[:, :, :H, :W]

        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        output_bgr = (output_np * 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_bgr, cv2.COLOR_RGB2BGR)

        # --- HIỂN THỊ ---
        combined = np.hstack((frame, output_bgr))
        
        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "FLOL Enhanced", (frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('FLOL Real-time Demo (Press Q to exit)', combined)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, default="./options/LOLv2-Real.yml")
    
    args = parser.parse_args()
    opt = parse(args.config)

    main(opt)