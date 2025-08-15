"""
Hệ thống phát hiện bất thường trong Log
Tác giả: Phạm Ngọc Minh
Email: phamngocminh1230@gmail.com
Phiên bản: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import re
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer, KNNImputer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import gensim
from gensim.models import Word2Vec
from collections import Counter
import nltk

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Hệ thống phát hiện bất thường Log", 
    page_icon="🔍", 
    layout="wide"
)

st.markdown("**Phát triển bởi:** Phạm Ngọc Minh | **Email:** phamngocminh1230@gmail.com")

if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def assess_data_quality(df):
    """Đánh giá chất lượng dữ liệu"""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100
    
    duplicates = df.duplicated().sum()
    duplicate_percentage = (duplicates / len(df)) * 100
    
    quality_score = 100 - missing_percentage - duplicate_percentage
    
    if quality_score >= 90:
        quality_level = "Xuất sắc"
    elif quality_score >= 75:
        quality_level = "Tốt"
    elif quality_score >= 50:
        quality_level = "Khá"
    else:
        quality_level = "Yếu"
    
    return {
        'score': quality_score,
        'level': quality_level,
        'missing_percentage': missing_percentage,
        'duplicate_percentage': duplicate_percentage,
        'missing_cells': missing_cells,
        'duplicates': duplicates
    }

def robust_data_preprocessing(df, selected_features, imputation_strategy='mean'):
    """Tiền xử lý dữ liệu mạnh mẽ"""
    processed_df = df[selected_features].copy()
    
    initial_shape = processed_df.shape
    initial_missing = processed_df.isnull().sum().sum()
    
    preprocessing_log = []
    preprocessing_log.append(f"Kích thước dữ liệu ban đầu: {initial_shape}")
    preprocessing_log.append(f"Giá trị thiếu ban đầu: {initial_missing}")
    
    numeric_columns = []
    categorical_columns = []
    
    for col in processed_df.columns:
        if processed_df[col].dtype in ['int64', 'float64']:
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)
    
    preprocessing_log.append(f"Cột số: {len(numeric_columns)}")
    preprocessing_log.append(f"Cột phân loại: {len(categorical_columns)}")
    
    # Xử lý cột số
    if numeric_columns:
        for col in numeric_columns:
            missing_count = processed_df[col].isnull().sum()
            if missing_count > 0:
                if imputation_strategy == 'mean':
                    fill_value = processed_df[col].mean()
                elif imputation_strategy == 'median':
                    fill_value = processed_df[col].median()
                elif imputation_strategy == 'mode':
                    fill_value = processed_df[col].mode()[0] if len(processed_df[col].mode()) > 0 else 0
                else:
                    processed_df[col] = processed_df[col].fillna(method='ffill').fillna(method='bfill')
                    continue
                
                processed_df[col] = processed_df[col].fillna(fill_value)
                preprocessing_log.append(f"{col}: Điền {missing_count} giá trị thiếu bằng {imputation_strategy}")
        
        # KNN Imputation cho giá trị thiếu còn lại
        if processed_df[numeric_columns].isnull().sum().sum() > 0:
            try:
                knn_imputer = KNNImputer(n_neighbors=min(5, len(processed_df)//2))
                processed_df[numeric_columns] = knn_imputer.fit_transform(processed_df[numeric_columns])
                preprocessing_log.append("Áp dụng KNN imputation cho giá trị thiếu còn lại")
            except:
                for col in numeric_columns:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                preprocessing_log.append("Dự phòng: Sử dụng median imputation")
    
    # Xử lý cột phân loại
    if categorical_columns:
        for col in categorical_columns:
            missing_count = processed_df[col].isnull().sum()
            if missing_count > 0:
                mode_value = processed_df[col].mode()
                fill_value = mode_value[0] if len(mode_value) > 0 else 'unknown'
                processed_df[col] = processed_df[col].fillna(fill_value)
                preprocessing_log.append(f"{col}: Điền {missing_count} giá trị thiếu bằng '{fill_value}'")
    
    # Label Encoding
    label_encoders = {}
    if categorical_columns:
        for col in categorical_columns:
            le = LabelEncoder()
            processed_df[col] = processed_df[col].astype(str).fillna('unknown')
            processed_df[col] = le.fit_transform(processed_df[col])
            label_encoders[col] = le
            preprocessing_log.append(f"{col}: Mã hóa {len(le.classes_)} danh mục")
    
    # Xử lý ngoại lệ (outliers)
    outlier_counts = {}
    if numeric_columns:
        for col in numeric_columns:
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)).sum()
            outlier_counts[col] = outliers
            
            if outliers > 0:
                processed_df[col] = processed_df[col].clip(lower=lower_bound, upper=upper_bound)
                preprocessing_log.append(f"{col}: Cắt {outliers} ngoại lệ")
    
    # Dọn dẹp cuối cùng
    final_missing = processed_df.isnull().sum().sum()
    if final_missing > 0:
        threshold = len(processed_df.columns) * 0.5
        processed_df = processed_df.dropna(thresh=threshold)
        processed_df = processed_df.fillna(0)
        preprocessing_log.append(f"Dọn dẹp cuối: Điền {final_missing} giá trị còn lại bằng 0")
    
    processed_df = processed_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    final_shape = processed_df.shape
    preprocessing_log.append(f"Kích thước dữ liệu cuối: {final_shape}")
    preprocessing_log.append(f"Giá trị thiếu cuối: {processed_df.isnull().sum().sum()}")
    
    return processed_df, label_encoders, preprocessing_log

def clean_log_text(text):
    """Làm sạch văn bản log"""
    if pd.isna(text) or text is None:
        return "empty_log"
    
    try:
        text = str(text)
        # Loại bỏ timestamp
        text = re.sub(r'\d{4}-\d{2}-\d{2}[\s\\T]\d{2}:\d{2}:\d{2}', '', text)
        text = re.sub(r'\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}', '', text)
        text = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '', text)
        # Loại bỏ IP address
        text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '', text)
        # Loại bỏ URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Loại bỏ ký tự đặc biệt
        text = re.sub(r'[^\w\s]', ' ', text)
        # Loại bỏ số
        text = re.sub(r'\b\d+\b', '', text)
        text = ' '.join(text.lower().split())
        
        return text if text.strip() else "empty_log"
    except Exception as e:
        return "error_log"

def get_robust_embeddings(texts, method='tfidf', **kwargs):
    """Tạo vector embedding mạnh mẽ cho văn bản"""
    try:
        cleaned_texts = [clean_log_text(text) for text in texts]
        
        # Kiểm tra nếu tất cả text đều trống
        non_empty_texts = [t for t in cleaned_texts if t and t.strip() and t != "empty_log"]
        if len(non_empty_texts) < 2:
            # Fallback: tạo embedding đơn giản
            st.warning("Dữ liệu văn bản quá ít hoặc trống. Sử dụng phương pháp dự phòng.")
            embedding_dim = kwargs.get('max_features', 50)
            embeddings = np.random.rand(len(texts), embedding_dim)
            return embeddings, None
        
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=kwargs.get('max_features', 1000),
                ngram_range=kwargs.get('ngram_range', (1, 2)),
                stop_words='english' if not NLTK_AVAILABLE else None,
                min_df=max(1, min(2, len(non_empty_texts) // 20)),  # Động điều chỉnh min_df
                max_df=min(0.95, (len(non_empty_texts) - 1) / len(non_empty_texts)),  # Động điều chỉnh max_df
                token_pattern=r'\b\w+\b'
            )
            try:
                embeddings = vectorizer.fit_transform(cleaned_texts).toarray()
                return embeddings, vectorizer
            except ValueError as e:
                if "After pruning" in str(e):
                    # Thử lại với tham số linh hoạt hơn
                    vectorizer = TfidfVectorizer(
                        max_features=min(100, len(non_empty_texts) * 2),
                        ngram_range=(1, 1),
                        min_df=1,
                        max_df=1.0,
                        token_pattern=r'\w+'
                    )
                    embeddings = vectorizer.fit_transform(cleaned_texts).toarray()
                    return embeddings, vectorizer
                else:
                    raise e
        
        elif method == 'count':
            vectorizer = CountVectorizer(
                max_features=kwargs.get('max_features', 1000),
                ngram_range=kwargs.get('ngram_range', (1, 2)),
                stop_words='english' if not NLTK_AVAILABLE else None,
                min_df=max(1, min(2, len(non_empty_texts) // 20)),
                max_df=min(0.95, (len(non_empty_texts) - 1) / len(non_empty_texts)),
                token_pattern=r'\b\w+\b'
            )
            try:
                embeddings = vectorizer.fit_transform(cleaned_texts).toarray()
                return embeddings, vectorizer
            except ValueError as e:
                if "After pruning" in str(e):
                    vectorizer = CountVectorizer(
                        max_features=min(100, len(non_empty_texts) * 2),
                        ngram_range=(1, 1),
                        min_df=1,
                        max_df=1.0,
                        token_pattern=r'\w+'
                    )
                    embeddings = vectorizer.fit_transform(cleaned_texts).toarray()
                    return embeddings, vectorizer
                else:
                    raise e
        
        elif method == 'word2vec':
            tokenized_texts = []
            for text in cleaned_texts:
                if NLTK_AVAILABLE:
                    try:
                        tokens = word_tokenize(text)
                        stop_words_set = set(stopwords.words('english'))
                        tokens = [token for token in tokens if token.isalpha() and token not in stop_words_set]
                    except:
                        tokens = text.split()
                else:
                    tokens = text.split()
                
                if not tokens:
                    tokens = ['empty']
                tokenized_texts.append(tokens)
            
            vector_size = kwargs.get('vector_size', 100)
            try:
                model = Word2Vec(
                    tokenized_texts, 
                    vector_size=vector_size, 
                    window=kwargs.get('window', 5), 
                    min_count=1,
                    workers=1,
                    sg=1
                )
                
                embeddings = []
                for tokens in tokenized_texts:
                    if tokens:
                        vectors = []
                        for word in tokens:
                            if word in model.wv:
                                vectors.append(model.wv[word])
                        
                        if vectors:
                            doc_vector = np.mean(vectors, axis=0)
                        else:
                            doc_vector = np.zeros(vector_size)
                    else:
                        doc_vector = np.zeros(vector_size)
                    
                    embeddings.append(doc_vector)
                
                return np.array(embeddings), model
            except Exception as e:
                st.warning(f"Lỗi Word2Vec: {e}. Chuyển sang TF-IDF.")
                return get_robust_embeddings(texts, method='tfidf', **kwargs)
        
    except Exception as e:
        st.error(f"Lỗi trong tạo embedding: {str(e)}")
        try:
            # Fallback cuối cùng
            vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 1), min_df=1, max_df=1.0)
            embeddings = vectorizer.fit_transform([str(t) if t else "empty" for t in texts]).toarray()
            return embeddings, vectorizer
        except:
            # Fallback hoàn toàn
            embedding_dim = 50
            embeddings = np.random.rand(len(texts), embedding_dim)
            return embeddings, None

def robust_anomaly_detection(data, algorithm, **params):
    """Thuật toán phát hiện bất thường mạnh mẽ"""
    try:
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)
        
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data)
        scaled_data = np.nan_to_num(scaled_data, nan=0.0)
        
        if algorithm == "Isolation Forest":
            model = IsolationForest(
                contamination=params.get('contamination', 0.1),
                n_estimators=params.get('n_estimators', 100),
                random_state=42
            )
            labels = model.fit_predict(scaled_data)
            scores = model.score_samples(scaled_data)
            
        elif algorithm == "DBSCAN":
            model = DBSCAN(
                eps=params.get('eps', 0.5),
                min_samples=params.get('min_samples', 5)
            )
            cluster_labels = model.fit_predict(scaled_data)
            labels = np.where(cluster_labels == -1, -1, 1)
            scores = np.array([np.min(np.linalg.norm(scaled_data - scaled_data[i], axis=1)) 
                              for i in range(len(scaled_data))])
            
        elif algorithm == "One-Class SVM":
            model = OneClassSVM(
                nu=params.get('nu', 0.1),
                kernel=params.get('kernel', 'rbf'),
                gamma='scale'
            )
            labels = model.fit_predict(scaled_data)
            scores = model.decision_function(scaled_data)
            
        elif algorithm == "Local Outlier Factor":
            n_neighbors = min(params.get('n_neighbors', 20), len(scaled_data) - 1)
            model = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=params.get('contamination', 0.1)
            )
            labels = model.fit_predict(scaled_data)
            scores = model.negative_outlier_factor_
            
        elif algorithm == "K-Means Clustering":
            n_clusters = params.get('n_clusters', 8)
            if n_clusters == 'auto':
                n_clusters = min(8, max(2, len(scaled_data) // 10))
            
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = model.fit_predict(scaled_data)
            
            distances = []
            for i, point in enumerate(scaled_data):
                cluster_center = model.cluster_centers_[cluster_labels[i]]
                distance = np.linalg.norm(point - cluster_center)
                distances.append(distance)
            
            distances = np.array(distances)
            threshold = np.percentile(distances, params.get('threshold_percentile', 95))
            labels = np.where(distances > threshold, -1, 1)
            scores = distances
            
        else:  # Statistical Methods
            method = params.get('method', 'zscore')
            threshold = params.get('threshold', 3)
            
            if method == 'zscore':
                z_scores = np.abs(scaled_data)
                scores = np.max(z_scores, axis=1)
            else:
                median = np.median(scaled_data, axis=0)
                mad = np.median(np.abs(scaled_data - median), axis=0)
                mad = np.where(mad == 0, 1, mad)
                modified_z_scores = 0.6745 * (scaled_data - median) / mad
                scores = np.max(np.abs(modified_z_scores), axis=1)
            
            labels = np.where(scores > threshold, -1, 1)
        
        scores = np.nan_to_num(scores, nan=0.0)
        
        return labels, scores, {'scaler': scaler, 'model': model if 'model' in locals() else None}
    
    except Exception as e:
        st.error(f"Lỗi trong {algorithm}: {str(e)}")
        dummy_labels = np.random.choice([-1, 1], size=len(data), p=[0.1, 0.9])
        dummy_scores = np.random.rand(len(data))
        return dummy_labels, dummy_scores, {'scaler': None, 'model': None}

def get_suspicion_level(score, all_scores):
    """Phân loại mức độ nghi ngờ dựa trên điểm số"""
    if len(all_scores) == 0:
        return "Không xác định"
    
    # Tính percentile của điểm số trong tập bất thường
    percentile = (np.sum(all_scores <= score) / len(all_scores)) * 100
    
    if percentile >= 90:
        return "Cực kỳ cao"
    elif percentile >= 70:
        return "Cao" 
    elif percentile >= 40:
        return "Trung bình"
    else:
        return "Thấp"

def create_advanced_visualization(data, labels, scores, method_name):
    """Tạo biểu đồ trực quan hóa nâng cao"""
    try:
        n_components = min(3, data.shape[1], len(data))
        if n_components < 2:
            data_viz = np.column_stack([data.flatten() if data.ndim > 1 else data, 
                                       np.zeros(len(data))])
        else:
            pca = PCA(n_components=n_components)
            data_viz = pca.fit_transform(data)
        
        plot_df = pd.DataFrame({
            'PC1': data_viz[:, 0],
            'PC2': data_viz[:, 1] if data_viz.shape[1] > 1 else np.zeros(len(data_viz)),
            'PC3': data_viz[:, 2] if data_viz.shape[1] > 2 else np.zeros(len(data_viz)),
            'Label': ['Bất thường' if x == -1 else 'Bình thường' for x in labels],
            'Score': scores,
            'Index': range(len(labels))
        })
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{method_name} - Biểu đồ 2D',
                f'{method_name} - Xem 3D',
                'Phân bố điểm số',
                'Dòng thời gian bất thường'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter3d"}],
                [{"type": "histogram"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.15,  # Tăng khoảng cách dọc
            horizontal_spacing=0.12  # Tăng khoảng cách ngang
        )
        
        # Biểu đồ 2D
        fig.add_trace(
            go.Scatter(
                x=plot_df[plot_df['Label'] == 'Bình thường']['PC1'],
                y=plot_df[plot_df['Label'] == 'Bình thường']['PC2'],
                mode='markers',
                name='Bình thường',
                marker=dict(color='#3498db', size=6, opacity=0.7),
                hovertemplate='<b>Bình thường</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>Điểm: %{customdata:.3f}',
                customdata=plot_df[plot_df['Label'] == 'Bình thường']['Score']
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=plot_df[plot_df['Label'] == 'Bất thường']['PC1'],
                y=plot_df[plot_df['Label'] == 'Bất thường']['PC2'],
                mode='markers',
                name='Bất thường',
                marker=dict(color='#e74c3c', size=10, opacity=0.8),
                hovertemplate='<b>Bất thường</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>Điểm: %{customdata:.3f}',
                customdata=plot_df[plot_df['Label'] == 'Bất thường']['Score']
            ),
            row=1, col=1
        )
        
        # Biểu đồ 3D
        fig.add_trace(
            go.Scatter3d(
                x=plot_df[plot_df['Label'] == 'Bình thường']['PC1'],
                y=plot_df[plot_df['Label'] == 'Bình thường']['PC2'],
                z=plot_df[plot_df['Label'] == 'Bình thường']['PC3'],
                mode='markers',
                name='Bình thường 3D',
                marker=dict(color='#3498db', size=4, opacity=0.6),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter3d(
                x=plot_df[plot_df['Label'] == 'Bất thường']['PC1'],
                y=plot_df[plot_df['Label'] == 'Bất thường']['PC2'],
                z=plot_df[plot_df['Label'] == 'Bất thường']['PC3'],
                mode='markers',
                name='Bất thường 3D',
                marker=dict(color='#e74c3c', size=6, opacity=0.8),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Histogram điểm số
        fig.add_trace(
            go.Histogram(
                x=plot_df[plot_df['Label'] == 'Bình thường']['Score'],
                name='Điểm bình thường',
                marker_color='#3498db',
                opacity=0.7,
                nbinsx=30
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=plot_df[plot_df['Label'] == 'Bất thường']['Score'],
                name='Điểm bất thường',
                marker_color='#e74c3c',
                opacity=0.7,
                nbinsx=30
            ),
            row=2, col=1
        )
        
        # Timeline
        colors = ['#3498db' if x == 'Bình thường' else '#e74c3c' for x in plot_df['Label']]
        sizes = [6 if x == 'Bình thường' else 10 for x in plot_df['Label']]
        
        fig.add_trace(
            go.Scatter(
                x=plot_df['Index'],
                y=plot_df['Score'],
                mode='markers+lines',
                name='Timeline điểm số',
                marker=dict(
                    color=colors,
                    size=sizes,
                    opacity=0.7
                ),
                line=dict(color='#95a5a6', width=1),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,  # Tăng chiều cao
            title_text=f"Kết quả phân tích - {method_name}",
            title_x=0.5,
            template="plotly_white"
        )
        
        # Cập nhật nhãn trục
        fig.update_xaxes(title_text="Thành phần chính 1", row=1, col=1)
        fig.update_yaxes(title_text="Thành phần chính 2", row=1, col=1)
        fig.update_xaxes(title_text="Điểm bất thường", row=2, col=1)
        fig.update_yaxes(title_text="Số lượng", row=2, col=1)
        fig.update_xaxes(title_text="Chỉ số điểm dữ liệu", row=2, col=2)
        fig.update_yaxes(title_text="Điểm bất thường", row=2, col=2)
        
        return fig
        
    except Exception as e:
        st.error(f"Lỗi tạo biểu đồ: {str(e)}")
        return go.Figure()

def get_suspicion_level(score, all_scores):
    """Phân loại mức độ nghi ngờ dựa trên điểm số"""
    if len(all_scores) == 0:
        return "Không xác định"
    
    # Tính percentile của điểm số trong tập bất thường
    percentile = (np.sum(all_scores <= score) / len(all_scores)) * 100
    
    if percentile >= 90:
        return "Cực kỳ cao"
    elif percentile >= 70:
        return "Cao" 
    elif percentile >= 40:
        return "Trung bình"
    else:
        return "Thấp"

def load_data_with_validation(uploaded_file):
    """Tải và kiểm tra dữ liệu"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return None, "Không thể đọc file CSV với các encoding hỗ trợ"
                
        elif file_extension == 'json':
            content = uploaded_file.read()
            json_data = json.loads(content)
            
            if isinstance(json_data, list):
                df = pd.json_normalize(json_data)
            elif isinstance(json_data, dict):
                df = pd.json_normalize(json_data)
            else:
                return None, "Định dạng JSON không được hỗ trợ"
        else:
            return None, f"Định dạng file không được hỗ trợ: {file_extension}"
        
        if df.empty:
            return None, "File không chứa dữ liệu"
        
        if len(df.columns) == 0:
            return None, "File không chứa cột nào"
        
        return df, None
        
    except Exception as e:
        return None, f"Lỗi đọc file: {str(e)}"

# Giao diện chính
st.title("🔍 Hệ thống phát hiện bất thường trong Log")
st.write("Tải lên dữ liệu log của bạn và phát hiện bất thường bằng nhiều thuật toán machine learning khác nhau")

# Thêm hướng dẫn giải thích các thuật ngữ
with st.expander("📚 Giải thích thuật ngữ và chỉ số"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🤖 Thuật toán phát hiện bất thường:**")
        st.write("• **Isolation Forest**: Phân lập điểm bất thường dựa trên độ khó tách biệt")
        st.write("• **DBSCAN**: Phân cụm dựa trên mật độ, điểm cô lập được coi là bất thường")
        st.write("• **One-Class SVM**: Tạo ranh giới xung quanh dữ liệu bình thường")
        st.write("• **Local Outlier Factor (LOF)**: Đánh giá mức độ bất thường dựa trên láng giềng")
        st.write("• **K-Means**: Phân cụm và phát hiện điểm xa tâm cụm")
        st.write("• **Statistical Methods**: Dùng Z-score hoặc Modified Z-score")
        
        st.markdown("**📊 Chỉ số chất lượng dữ liệu:**")
        st.write("• **Điểm chất lượng**: 100 - % thiếu - % trùng lặp")
        st.write("• **Xuất sắc** (≥90): Dữ liệu rất tốt, ít lỗi")
        st.write("• **Tốt** (75-89): Chất lượng ổn định")
        st.write("• **Khá** (50-74): Cần xử lý thêm")
        st.write("• **Yếu** (<50): Nhiều vấn đề, cần làm sạch")
    
    with col2:
        st.markdown("**📈 Chỉ số kết quả phân tích:**")
        st.write("• **Anomaly Score**: Điểm bất thường - cao hơn = nghi ngờ hơn")
        st.write("• **Contamination**: Tỷ lệ bất thường dự kiến trong dữ liệu (0.01-0.3)")
        st.write("• **Principal Component (PC)**: Thành phần chính từ PCA để giảm chiều")
        st.write("• **N-gram**: Chuỗi n từ liên tiếp (1-gram = từ đơn, 2-gram = cặp từ)")
        
        st.markdown("**🔧 Phương pháp xử lý văn bản:**")
        st.write("• **TF-IDF**: Tần suất từ có trọng số (phổ biến nhất)")
        st.write("• **Count Vectorizer**: Đếm tần suất từ đơn giản")
        st.write("• **Word2Vec**: Vector từ học sâu (chậm hơn nhưng chính xác hơn)")
        
        st.markdown("**⚙️ Phương pháp điền giá trị thiếu:**")
        st.write("• **Mean**: Giá trị trung bình")
        st.write("• **Median**: Giá trị trung vị (tốt với outlier)")
        st.write("• **Mode**: Giá trị xuất hiện nhiều nhất")
        st.write("• **Forward Fill**: Dùng giá trị trước đó")

with st.sidebar:
    st.header("📁 Tải dữ liệu")
    
    uploaded_file = st.file_uploader(
        "Chọn file log của bạn",
        type=['csv', 'json'],
        help="Hỗ trợ định dạng CSV và JSON"
    )
    
    if uploaded_file is not None:
        with st.spinner('Đang tải dữ liệu...'):
            data, error = load_data_with_validation(uploaded_file)
            
            if error:
                st.error(error)
            else:
                st.session_state.data = data
                quality_info = assess_data_quality(data)
                
                st.success("Tải dữ liệu thành công!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Số hàng", f"{len(data):,}")
                with col2:
                    st.metric("Số cột", len(data.columns))
                
                # Hiển thị chất lượng dữ liệu với màu sắc
                quality_color = {
                    "Xuất sắc": "🟢",
                    "Tốt": "🔵", 
                    "Khá": "🟡",
                    "Yếu": "🔴"
                }
                
                st.info(f"{quality_color.get(quality_info['level'], '⚪')} Chất lượng dữ liệu: {quality_info['level']} ({quality_info['score']:.1f}%)")

if st.session_state.data is not None:
    data = st.session_state.data
    
    st.header("📊 Tổng quan dữ liệu")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Mẫu dữ liệu")
        st.dataframe(data.head(10), use_container_width=True)
    
    with col2:
        quality_info = assess_data_quality(data)
        
        st.metric("Tổng bản ghi", f"{len(data):,}")
        st.metric("Số đặc trưng", len(data.columns))
        st.metric("% Thiếu dữ liệu", f"{quality_info['missing_percentage']:.1f}%")
        st.metric("% Trùng lặp", f"{quality_info['duplicate_percentage']:.1f}%")
        
        # Thêm thông tin chi tiết
        with st.expander("Chi tiết chất lượng"):
            st.write(f"Ô thiếu: {quality_info['missing_cells']:,}")
            st.write(f"Bản ghi trùng: {quality_info['duplicates']:,}")
    
    st.header("⚙️ Cấu hình phân tích")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Chọn đặc trưng")
        
        available_features = list(data.columns)
        selected_features = st.multiselect(
            "Chọn cột để phân tích:",
            available_features,
            default=available_features[:min(10, len(available_features))],
            help="Chọn các cột để đưa vào phân tích"
        )
        
        if selected_features:
            numeric_features = []
            text_features = []
            
            for feature in selected_features:
                if data[feature].dtype in ['int64', 'float64']:
                    numeric_features.append(feature)
                elif data[feature].dtype == 'object':
                    sample_values = data[feature].dropna().astype(str)
                    if len(sample_values) > 0:
                        avg_length = sample_values.str.len().mean()
                        if avg_length > 10:
                            text_features.append(feature)
                        else:
                            text_features.append(feature)
                else:
                    text_features.append(feature)
            
            st.info(f"🔢 Cột số: {len(numeric_features)} | 📝 Cột văn bản: {len(text_features)}")
    
    with col2:
        st.subheader("Chọn thuật toán")
        
        available_algorithms = [
            "Isolation Forest",
            "DBSCAN", 
            "One-Class SVM",
            "Local Outlier Factor",
            "K-Means Clustering",
            "Statistical Methods"
        ]
        
        selected_algorithms = st.multiselect(
            "Chọn thuật toán phát hiện:",
            available_algorithms,
            default=["Isolation Forest", "Local Outlier Factor"],
            help="Chọn một hoặc nhiều thuật toán"
        )
        
        st.subheader("Tiền xử lý")
        
        imputation_strategy = st.selectbox(
            "Chiến lược xử lý giá trị thiếu:",
            ["mean", "median", "mode", "forward_fill"],
            index=1,  # Mặc định median
            help="Phương pháp xử lý giá trị thiếu"
        )
        
        if text_features:
            st.subheader("Xử lý văn bản")
            
            primary_text_column = st.selectbox(
                "Cột văn bản chính:",
                text_features,
                help="Cột văn bản chính để phân tích"
            )
            
            embedding_method = st.selectbox(
                "Phương pháp embedding văn bản:",
                ["tfidf", "count", "word2vec"],
                help="Phương pháp vector hóa văn bản"
            )
        else:
            embedding_method = None
            primary_text_column = None
    
    if selected_algorithms:
        with st.expander("⚙️ Tham số thuật toán", expanded=False):
            algorithm_params = {}
            
            tabs = st.tabs(selected_algorithms)
            
            for idx, algorithm in enumerate(selected_algorithms):
                with tabs[idx]:
                    if algorithm == "Isolation Forest":
                        st.write("**Isolation Forest** phân lập điểm bất thường bằng cách tạo cây ngẫu nhiên")
                        contamination = st.slider("Contamination (tỷ lệ bất thường dự kiến):", 0.01, 0.3, 0.1, 0.01, key=f"iso_contamination_{idx}")
                        n_estimators = st.slider("Số cây (nhiều hơn = chính xác hơn nhưng chậm hơn):", 50, 300, 100, 50, key=f"iso_estimators_{idx}")
                        algorithm_params[algorithm] = {
                            'contamination': contamination,
                            'n_estimators': n_estimators
                        }
                    
                    elif algorithm == "Local Outlier Factor":
                        st.write("**LOF** đánh giá mức độ bất thường dựa trên mật độ láng giềng")
                        n_neighbors = st.slider("Số láng giềng (nhiều hơn = ổn định hơn):", 5, 50, 20, 5, key=f"lof_neighbors_{idx}")
                        contamination = st.slider("Contamination:", 0.01, 0.3, 0.1, 0.01, key=f"lof_contamination_{idx}")
                        algorithm_params[algorithm] = {
                            'n_neighbors': n_neighbors,
                            'contamination': contamination
                        }
                    
                    elif algorithm == "DBSCAN":
                        st.write("**DBSCAN** nhóm điểm theo mật độ, điểm cô lập là bất thường")
                        eps = st.slider("Eps (bán kính láng giềng):", 0.1, 2.0, 0.5, 0.1, key=f"dbscan_eps_{idx}")
                        min_samples = st.slider("Số điểm tối thiểu trong cụm:", 2, 20, 5, 1, key=f"dbscan_min_samples_{idx}")
                        algorithm_params[algorithm] = {
                            'eps': eps,
                            'min_samples': min_samples
                        }
                    
                    elif algorithm == "One-Class SVM":
                        st.write("**One-Class SVM** tạo ranh giới xung quanh dữ liệu bình thường")
                        nu = st.slider("Nu (tỷ lệ lỗi và support vector):", 0.01, 0.5, 0.1, 0.01, key=f"svm_nu_{idx}")
                        kernel = st.selectbox("Kernel (hàm nhân):", ['rbf', 'linear', 'poly'], key=f"svm_kernel_{idx}")
                        algorithm_params[algorithm] = {
                            'nu': nu,
                            'kernel': kernel
                        }
                    
                    elif algorithm == "K-Means Clustering":
                        st.write("**K-Means** phân cụm và coi điểm xa tâm cụm là bất thường")
                        n_clusters = st.selectbox("Số cụm:", ['auto', 3, 4, 5, 6, 7, 8, 9, 10], key=f"kmeans_clusters_{idx}")
                        threshold_percentile = st.slider("Ngưỡng phần trăm (cao hơn = ít bất thường hơn):", 85, 99, 95, 1, key=f"kmeans_threshold_{idx}")
                        algorithm_params[algorithm] = {
                            'n_clusters': n_clusters,
                            'threshold_percentile': threshold_percentile
                        }
                    
                    elif algorithm == "Statistical Methods":
                        st.write("**Phương pháp thống kê** dùng Z-score để phát hiện ngoại lệ")
                        method = st.selectbox("Phương pháp:", ['zscore', 'modified_zscore'], key=f"stat_method_{idx}")
                        threshold = st.slider("Ngưỡng (cao hơn = ít bất thường hơn):", 2.0, 5.0, 3.0, 0.5, key=f"stat_threshold_{idx}")
                        algorithm_params[algorithm] = {
                            'method': method,
                            'threshold': threshold
                        }

    if text_features and embedding_method:
        with st.expander("📝 Tham số xử lý văn bản", expanded=False):
            max_features = st.slider("Số đặc trưng tối đa (nhiều hơn = chi tiết hơn nhưng chậm hơn):", 100, 5000, 1000, 100, key="text_max_features")
            if embedding_method in ["tfidf", "count"]:
                ngram_min = st.slider("N-gram tối thiểu (1 = từ đơn):", 1, 3, 1, key="text_ngram_min")
                ngram_max = st.slider("N-gram tối đa (2 = cặp từ):", 1, 5, 2, key="text_ngram_max")
            elif embedding_method == "word2vec":
                vector_size = st.slider("Kích thước vector (lớn hơn = biểu diễn tốt hơn):", 50, 300, 100, 50, key="text_vector_size")
                window_size = st.slider("Kích thước cửa sổ (bối cảnh từ):", 3, 10, 5, key="text_window_size")
    else:
        max_features = 1000
        ngram_min = 1
        ngram_max = 2
        vector_size = 100
        window_size = 5

    if selected_features and selected_algorithms:
        if st.button("🚀 Chạy phân tích", type="primary", use_container_width=True):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Đang tiền xử lý dữ liệu...")
                progress_bar.progress(10)
                
                processed_data, label_encoders, preprocessing_log = robust_data_preprocessing(
                    data, selected_features, imputation_strategy
                )
                
                if text_features and embedding_method and primary_text_column:
                    status_text.text("Đang tạo embedding văn bản...")
                    progress_bar.progress(30)
                    
                    if embedding_method == "word2vec":
                        text_embeddings, text_model = get_robust_embeddings(
                            data[primary_text_column],
                            method=embedding_method,
                            vector_size=vector_size,
                            window=window_size
                        )
                    else:
                        text_embeddings, text_model = get_robust_embeddings(
                            data[primary_text_column],
                            method=embedding_method,
                            max_features=max_features,
                            ngram_range=(ngram_min, ngram_max)
                        )
                    
                    if len(numeric_features) > 0:
                        numerical_data = processed_data[numeric_features].values
                        scaler = StandardScaler()
                        numerical_data_scaled = scaler.fit_transform(numerical_data)
                        final_data = np.hstack([text_embeddings, numerical_data_scaled])
                    else:
                        final_data = text_embeddings
                else:
                    final_data = processed_data.values
                
                results = {}
                total_algorithms = len(selected_algorithms)
                
                for idx, algorithm in enumerate(selected_algorithms):
                    status_text.text(f"Đang chạy {algorithm}...")
                    progress_bar.progress(40 + (idx * 40 // total_algorithms))
                    
                    params = algorithm_params.get(algorithm, {})
                    labels, scores, model_info = robust_anomaly_detection(
                        final_data, algorithm, **params
                    )
                    
                    results[algorithm] = {
                        'labels': labels,
                        'scores': scores,
                        'model_info': model_info,
                        'anomaly_count': np.sum(labels == -1),
                        'anomaly_rate': np.sum(labels == -1) / len(labels) * 100
                    }
                
                progress_bar.progress(100)
                status_text.text("Hoàn thành phân tích!")
                
                st.session_state.analysis_results = {
                    'results': results,
                    'final_data': final_data,
                    'preprocessing_log': preprocessing_log,
                    'selected_features': selected_features,
                    'embedding_method': embedding_method
                }
                
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"Phân tích thất bại: {str(e)}")
                st.write("Chi tiết lỗi:", str(e))

if st.session_state.analysis_results is not None:
    st.header("📊 Kết quả phân tích")
    
    analysis_data = st.session_state.analysis_results
    results = analysis_data['results']
    final_data = analysis_data['final_data']
    
    # Hiển thị log tiền xử lý
    with st.expander("📋 Chi tiết tiền xử lý dữ liệu"):
        preprocessing_log = analysis_data.get('preprocessing_log', [])
        for log_entry in preprocessing_log:
            st.write(f"• {log_entry}")
    
    st.subheader("🔍 So sánh thuật toán")
    
    comparison_data = []
    for algorithm, result in results.items():
        comparison_data.append({
            'Thuật toán': algorithm,
            'Số bất thường': result['anomaly_count'],
            'Tỷ lệ (%)': f"{result['anomaly_rate']:.2f}%",
            'Số bình thường': len(result['labels']) - result['anomaly_count']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Metrics với màu sắc
    cols = st.columns(len(results))
    colors = ['🔴', '🟠', '🟡', '🟢', '🔵', '🟣']
    for idx, (algorithm, result) in enumerate(results.items()):
        with cols[idx]:
            color = colors[idx % len(colors)]
            st.metric(
                f"{color} {algorithm}",
                f"{result['anomaly_count']} bất thường",
                f"{result['anomaly_rate']:.1f}% tổng dữ liệu"
            )
    
    st.subheader("📈 Phân tích chi tiết")
    
    for algorithm, result in results.items():
        with st.expander(f"📊 Kết quả {algorithm}", expanded=True):
            
            fig = create_advanced_visualization(
                final_data, result['labels'], result['scores'], algorithm
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🚨 Tổng bất thường", result['anomaly_count'])
            
            with col2:
                normal_count = len(result['labels']) - result['anomaly_count']
                st.metric("✅ Điểm bình thường", normal_count)
            
            with col3:
                scores = result['scores']
                avg_score = np.mean(scores)
                st.metric("📊 Điểm TB", f"{avg_score:.3f}")
            
            with col4:
                scores = result['scores']
                min_score = np.min(scores)
                max_score = np.max(scores)
                st.metric("📏 Khoảng điểm", f"{min_score:.3f} - {max_score:.3f}")
            
            # Hiển thị danh sách chi tiết các điểm bất thường
            if result['anomaly_count'] > 0:
                st.markdown("---")
                st.subheader(f"🔍 Danh sách chi tiết {result['anomaly_count']} điểm bất thường")
                
                # Tạo DataFrame cho các điểm bất thường
                anomaly_indices = np.where(result['labels'] == -1)[0]
                anomaly_scores = result['scores'][anomaly_indices]
                
                # Sắp xếp theo điểm số từ cao đến thấp (nghi ngờ nhất trước)
                sorted_indices = np.argsort(-anomaly_scores)
                sorted_anomaly_indices = anomaly_indices[sorted_indices]
                sorted_anomaly_scores = anomaly_scores[sorted_indices]
                
                anomaly_details = []
                for i, (idx, score) in enumerate(zip(sorted_anomaly_indices, sorted_anomaly_scores)):
                    detail_row = {
                        'STT': i + 1,
                        'Chỉ số hàng': idx + 1,  # +1 để match với Excel numbering
                        'Điểm bất thường': f"{score:.4f}",
                        'Mức độ nghi ngờ': get_suspicion_level(score, sorted_anomaly_scores)
                    }
                    
                    # Thêm dữ liệu gốc từ các cột đã chọn
                    for col in analysis_data['selected_features'][:5]:  # Giới hạn 5 cột đầu để không quá rộng
                        if col in data.columns:
                            value = data.iloc[idx][col]
                            if pd.isna(value):
                                detail_row[col] = "N/A"
                            elif isinstance(value, (int, float)):
                                detail_row[col] = f"{value:.3f}" if isinstance(value, float) else str(value)
                            else:
                                # Cắt ngắn text dài
                                str_value = str(value)
                                detail_row[col] = str_value[:50] + "..." if len(str_value) > 50 else str_value
                    
                    anomaly_details.append(detail_row)
                
                anomaly_df = pd.DataFrame(anomaly_details)
                
                # Tabs cho different views
                tab1, tab2, tab3 = st.tabs(["📋 Bảng tổng hợp", "📊 Top 10 nghi ngờ nhất", "🔍 Tìm kiếm & Lọc"])
                
                with tab1:
                    st.write(f"**Hiển thị tất cả {len(anomaly_df)} điểm bất thường, sắp xếp theo mức độ nghi ngờ:**")
                    
                    # Color coding cho mức độ nghi ngờ
                    def highlight_suspicion(row):
                        if row['Mức độ nghi ngờ'] == 'Cực kỳ cao':
                            return ['background-color: #ffebee'] * len(row)
                        elif row['Mức độ nghi ngờ'] == 'Cao':
                            return ['background-color: #fff3e0'] * len(row)
                        elif row['Mức độ nghi ngờ'] == 'Trung bình':
                            return ['background-color: #f3e5f5'] * len(row)
                        else:
                            return ['background-color: #e8f5e8'] * len(row)
                    
                    styled_df = anomaly_df.style.apply(highlight_suspicion, axis=1)
                    st.dataframe(styled_df, use_container_width=True, height=400)
                
                with tab2:
                    st.write("**Top 10 điểm nghi ngờ nhất:**")
                    top_10 = anomaly_df.head(10)
                    
                    for i, row in top_10.iterrows():
                        with st.container():
                            col_a, col_b, col_c = st.columns([1, 2, 3])
                            
                            with col_a:
                                # Icon based on suspicion level
                                icon = "🔴" if row['Mức độ nghi ngờ'] == 'Cực kỳ cao' else \
                                       "🟠" if row['Mức độ nghi ngờ'] == 'Cao' else \
                                       "🟡" if row['Mức độ nghi ngờ'] == 'Trung bình' else "🟢"
                                st.markdown(f"### {icon} #{row['STT']}")
                                st.metric("Hàng", row['Chỉ số hàng'])
                            
                            with col_b:
                                st.metric("Điểm số", row['Điểm bất thường'])
                                st.write(f"**{row['Mức độ nghi ngờ']}**")
                            
                            with col_c:
                                st.write("**Dữ liệu gốc:**")
                                data_preview = {k: v for k, v in row.items() 
                                              if k not in ['STT', 'Chỉ số hàng', 'Điểm bất thường', 'Mức độ nghi ngờ']}
                                for key, value in data_preview.items():
                                    st.write(f"• **{key}**: {value}")
                        
                        st.markdown("---")
                
                with tab3:
                    st.write("**Tìm kiếm và lọc dữ liệu bất thường:**")
                    
                    col_search1, col_search2 = st.columns(2)
                    
                    with col_search1:
                        suspicion_filter = st.selectbox(
                            "Lọc theo mức độ nghi ngờ:",
                            ["Tất cả", "Cực kỳ cao", "Cao", "Trung bình", "Thấp"],
                            key=f"suspicion_filter_{algorithm}"
                        )
                    
                    with col_search2:
                        row_search = st.number_input(
                            "Tìm hàng cụ thể:",
                            min_value=1,
                            max_value=len(data),
                            value=1,
                            key=f"row_search_{algorithm}"
                        )
                    
                    # Apply filters
                    filtered_df = anomaly_df.copy()
                    
                    if suspicion_filter != "Tất cả":
                        filtered_df = filtered_df[filtered_df['Mức độ nghi ngờ'] == suspicion_filter]
                    
                    # Search for specific row
                    if st.button(f"🔍 Tìm hàng {row_search}", key=f"search_btn_{algorithm}"):
                        row_found = anomaly_df[anomaly_df['Chỉ số hàng'] == row_search]
                        if len(row_found) > 0:
                            st.success(f"✅ Tìm thấy! Hàng {row_search} là điểm bất thường với điểm số {row_found.iloc[0]['Điểm bất thường']}")
                            st.dataframe(row_found, use_container_width=True)
                        else:
                            st.info(f"ℹ️ Hàng {row_search} không phải là điểm bất thường theo thuật toán {algorithm}")
                    
                    st.write(f"**Kết quả lọc: {len(filtered_df)} điểm**")
                    st.dataframe(filtered_df, use_container_width=True, height=300)
                
                # Quick stats
                st.markdown("---")
                st.subheader("📈 Thống kê nhanh")
                
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                suspicion_counts = anomaly_df['Mức độ nghi ngờ'].value_counts()
                
                with col_stat1:
                    st.metric("🔴 Cực kỳ cao", suspicion_counts.get('Cực kỳ cao', 0))
                
                with col_stat2:
                    st.metric("🟠 Cao", suspicion_counts.get('Cao', 0))
                
                with col_stat3:
                    st.metric("🟡 Trung bình", suspicion_counts.get('Trung bình', 0))
                
                with col_stat4:
                    st.metric("🟢 Thấp", suspicion_counts.get('Thấp', 0))
            
            else:
                st.info("✅ Thuật toán này không phát hiện điểm bất thường nào.")
    
    st.subheader("💾 Xuất kết quả")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if results:
            export_data = data.copy()
            
            # Sử dụng thuật toán đầu tiên làm chính
            primary_algorithm = list(results.keys())[0]
            primary_result = results[primary_algorithm]
            
            export_data['Nhan_Bat_Thuong'] = primary_result['labels']
            export_data['Diem_Bat_Thuong'] = primary_result['scores']
            export_data['La_Bat_Thuong'] = primary_result['labels'] == -1
            export_data['Thuat_Toan'] = primary_algorithm
            
            csv_data = export_data.to_csv(index=False)
            
            st.download_button(
                label="📥 Tải kết quả đầy đủ",
                data=csv_data,
                file_name=f"phan_tich_bat_thuong_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        if results and len(export_data[export_data['La_Bat_Thuong']]) > 0:
            anomaly_data = export_data[export_data['La_Bat_Thuong']].copy()
            anomaly_csv = anomaly_data.to_csv(index=False)
            
            st.download_button(
                label="🚨 Chỉ tải bất thường",
                data=anomaly_csv,
                file_name=f"chi_bat_thuong_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.success(f"Tìm thấy {len(anomaly_data)} điểm bất thường!")

else:
    st.info("👆 Tải lên file log bằng thanh bên để bắt đầu!")
    
    st.subheader("✨ Tính năng hệ thống")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🤖 Nhiều thuật toán ML**")
        st.write("• Isolation Forest (phổ biến nhất)")
        st.write("• DBSCAN Clustering") 
        st.write("• One-Class SVM")
        st.write("• Local Outlier Factor")
        st.write("• K-Means Detection")
        st.write("• Phương pháp thống kê")
        
        st.markdown("**📊 Trực quan hóa phong phú**")
        st.write("• Biểu đồ 2D/3D tương tác")
        st.write("• So sánh thuật toán")
        st.write("• Phân bố điểm số")
        st.write("• Phân tích timeline")
    
    with col2:
        st.markdown("**📝 Xử lý văn bản nâng cao**")
        st.write("• Vector hóa TF-IDF")
        st.write("• Count vectorizer")
        st.write("• Word2Vec embedding")
        st.write("• Phân tích N-gram")
        st.write("• Tiền xử lý văn bản")
        
        st.markdown("**🔧 Xử lý dữ liệu mạnh mẽ**")
        st.write("• Tự động xử lý NaN")
        st.write("• Nhiều phương pháp điền thiếu")
        st.write("• Phát hiện ngoại lệ")
        st.write("• Đánh giá chất lượng dữ liệu")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Hệ thống phát hiện bất thường Log"
    "</div>", 
    unsafe_allow_html=True
)