import torch

if __name__=='__main__':
    print("This is a test file.")
    is_gpu = torch.cuda.is_available()
    print(f"Is GPU available: {is_gpu}")
    if is_gpu:
        gpu_counts = torch.cuda.device_count()
        print(f"Number of GPU devices: {gpu_counts}")
        for count in range(gpu_counts):

            print(f"GPU {count} Device Name: {torch.cuda.get_device_name(count)}")