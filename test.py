import torch
from utils import load_model, move_to
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
from utils.data_utils import check_extension, save_dataset


def get_best_routes(model_path, dataset_path, opts):
    """
    Sử dụng mô hình đã train để lấy các tuyến đường tốt nhất từ dữ liệu đầu vào.

    Args:
        model_path (str): Đường dẫn đến file checkpoint .pt chứa model đã train.
        dataset_path (str): Đường dẫn đến dataset (ví dụ file .txt hoặc .tsplib).
        opts (Namespace): Các tham số cấu hình (giống argparse).

    Returns:
        best_routes (list of list): Danh sách các tuyến đường tốt nhất.
        costs (list of float): Chi phí tương ứng với mỗi tuyến đường.
    """
    # Load mô hình
    agent_num = opts.agent_num
    model, _ = load_model(model_path, agent_num=agent_num, ft=opts.ft)
    model.agent_num = agent_num

    # Chọn thiết bị (GPU/CPU)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    model.eval()

    # Load dataset
    dataset = model.problem.make_dataset(
        filename=dataset_path, num_samples=opts.sample_size, offset=opts.offset
    )

    # Cấu hình chiến lược giải mã
    model.set_decode_type(
        "greedy" if opts.decode_strategy in ("greedy") else "sampling",
        temp=opts.softmax_temperature,
    )

    # DataLoader để xử lý batch
    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    best_routes = []
    costs = []

    # Duyệt qua từng batch
    for batch in tqdm(dataloader, desc="Evaluating", disable=opts.no_progress_bar):
        batch = move_to(batch, device)

        with torch.no_grad():
            # Dự đoán tuyến đường
            sequences, batch_costs = model.sample_many(
                batch, batch_rep=1, iter_rep=1, agent_num=opts.agent_num, aug=opts.N_aug
            )

        # Ghi lại kết quả
        best_routes.extend(sequences.cpu().numpy().tolist())  # Danh sách tuyến đường
        costs.extend(batch_costs.cpu().numpy().tolist())  # Danh sách chi phí

    return best_routes, costs
if __name__ == "__main__":
    model_path = "outputs/mtsp_500/epoch-1.pt"  # Đường dẫn tới checkpoint model
    n = 500
    coordinates = [[random.randint(0, 100), random.randint(0, 100)] for _ in range(n)]
    dataset_path = "test"  # Đường dẫn tới dataset
    save_dataset(coordinates, dataset_path)

    # Lấy các tuyến đường và chi phí
    best_routes, costs = get_best_routes(model_path, dataset_path, opts)

    # Hiển thị kết quả
    for i, (route, cost) in enumerate(zip(best_routes, costs)):
        print(f"Tuyến đường tốt nhất {i + 1}: {route} (Chi phí: {cost})")
