# SEMI-SUPERVISED LEARNING

| **Phương pháp**                 | **Định nghĩa** | **Cách huấn luyện** | **Ưu điểm** | **Nhược điểm** |
|--------------------------------|---------------|------------------|------------|--------------|
| **Continual Learning (CL)**    | Học liên tục trên nhiều tác vụ, không quên kiến thức cũ. | Huấn luyện theo từng tập dữ liệu mới mà không cần lưu dữ liệu cũ (hoặc chỉ lưu một phần nhỏ). | Giảm chi phí lưu trữ dữ liệu, học theo thời gian thực. | Dễ bị quên kiến thức cũ (catastrophic forgetting). |
| **Unsupervised Learning (UL)** | Học không có nhãn, chỉ dựa vào cấu trúc dữ liệu. | Sử dụng các phương pháp như clustering (K-Means, DBSCAN), autoencoder, GANs. | Không cần nhãn, phù hợp với dữ liệu lớn. | Kết quả khó đánh giá, cần phương pháp đánh giá khác. |
| **Semi-supervised Learning (SSL)** | Học từ một phần dữ liệu có nhãn và phần lớn dữ liệu không có nhãn. | Dùng một lượng nhỏ dữ liệu có nhãn để huấn luyện ban đầu, sau đó dùng dữ liệu không nhãn để cải thiện mô hình (pseudo-labeling, consistency regularization). | Giảm chi phí gán nhãn dữ liệu. | Hiệu quả phụ thuộc vào chiến lược khai thác dữ liệu không nhãn. |
| **Self-supervised Learning (Self-SL)** | Học bằng cách tạo ra nhãn từ dữ liệu mà không cần nhãn bên ngoài. | Tạo nhãn từ chính dữ liệu (contrastive learning, predictive task) và huấn luyện mô hình như supervised learning. | Giảm chi phí gán nhãn, hiệu quả cao với dữ liệu lớn. | Cần thiết kế bài toán proxy hợp lý. |


1. About model used
[Install model](/src/MODEL.md)



### Running
CML normal
```python
python main.py --dataset cifar100 --num_classes 100 --model LeNet --epochs 10 --batch_size 64 --lr 0.005 --optimizer sgd --device cuda
```
Semi-Supervised Learning
1. Pseudo Label
```python
python main_pseudo_label.py --dataset cifar100 --num_classes 100 --model LeNet --epochs 10 --batch_size 64 --lr 0.005 --optimizer sgd --device cuda
```
"# semi-supervised-learning" 
