# Slide Media Retrieval

Project xây dựng hệ thống text-to-image retrieval cho phép người dùng nhập mô tả văn bản và truy xuất top-k ảnh phù hợp từ tập ảnh có sẵn.

## Cấu trúc thư mục

- `data/raw/images/`: ảnh gốc
- `data/raw/captions/`: file caption
- `data/processed/`: metadata, embeddings, FAISS index
- `notebooks/`: notebook thử nghiệm
- `src/`: mã nguồn chính
- `app/`: Streamlit demo
- `results/`: kết quả test và hình minh họa