import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import cv2
import numpy as np
import time
from model.flol import create_model
from options.options import parse

def pad_tensor(tensor, multiple=8):
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
    return tensor

def main(opt, input_path, output_path, scale_percent):
    # 1. Cấu hình thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Đang chạy trên thiết bị: {device} ---")

    # 2. Load Model FLOL
    print("⏳ Đang tải mô hình FLOL...")
    model = create_model()
    weights_path = opt['settings']['weight']
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['params'])
    model.to(device)
    model.eval()
    print("Đã tải mô hình thành công!")

    # 3. Mở Video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Lỗi: Không mở được video {input_path}")
        return

    # Lấy thông số gốc
    org_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    org_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Nếu scale = 100 thì giữ nguyên, ngược lại thì thu nhỏ
    if scale_percent < 100:
        new_width = int(org_width * scale_percent / 100)
        new_height = int(org_height * scale_percent / 100)
        print(f"Đang RESIZE video: {org_width}x{org_height} -> {new_width}x{new_height} (Giảm còn {scale_percent}%)")
    else:
        new_width = org_width
        new_height = org_height
        print(f"Giữ nguyên độ phân giải: {org_width}x{org_height}")

    # 4. Video Writer 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    to_tensor = transforms.ToTensor()
    frame_count = 0
    start_time = time.time()

    print("Bắt đầu xử lý... (Nhấn 'q' để dừng sớm)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- RESIZE ---
        if scale_percent < 100:
            # Dùng INTER_AREA để ảnh nhỏ lại mà vẫn mịn, ít bị vỡ hạt
            frame_processing = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            frame_processing = frame

        # --- XỬ LÝ FLOL ---
        img_rgb = cv2.cvtColor(frame_processing, cv2.COLOR_BGR2RGB)
        img_tensor = to_tensor(img_rgb).unsqueeze(0).to(device)
        
        # Padding 
        _, _, H, W = img_tensor.shape
        img_padded = pad_tensor(img_tensor)

        with torch.no_grad():
            output = model(img_padded)

        # Hậu xử lý
        output = torch.clamp(output, 0., 1.)
        output = output[:, :, :H, :W] # Cắt bỏ phần padding
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        output_bgr = (output_np * 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_bgr, cv2.COLOR_RGB2BGR)

        # Ghi vào video
        out.write(output_bgr)

        frame_count += 1
        if frame_count % 10 == 0: # Cứ 10 frame cập nhật 1 lần cho đỡ lag console
            elapsed = time.time() - start_time
            process_fps = frame_count / elapsed
            print(f"\rTiến độ: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) | Tốc độ: {process_fps:.1f} FPS", end="")

        # cv2.imshow('Preview', output_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n\n✅ Xong! Video đã lưu tại: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./options/LOLv2-Real.yml")
    
    # INPUT
    parser.add_argument("--input", type=str, 
                        default="datasets/LOLv2-Real/test/Low/IMG_E2228.MOV", 
                        help="Đường dẫn file video đầu vào")

    # OUTPUT
    parser.add_argument("--output", type=str, 
                        default="results/LOLv2-Real/ket_qua.mp4", 
                        help="Đường dẫn file video kết quả")

    # SCALE
    parser.add_argument("--scale", type=int, default=30, help="Tỷ lệ % resize")
    
    args = parser.parse_args()
    opt = parse(args.config)

    
    import os
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tự động tạo thư mục: {output_dir}")

    main(opt, args.input, args.output, args.scale)