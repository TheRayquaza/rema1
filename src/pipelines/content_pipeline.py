import pandas as pd
from src.pipelines.base import AbstractPipeline
import numpy as np
from functools import reduce
from typing import Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_users(interactions: pd.DataFrame, user_features: pd.DataFrame, 
                    video_categories: pd.DataFrame, video_daily: pd.DataFrame) -> pd.DataFrame:
    ###########
    # Generic user stats
    user_stats = pd.DataFrame({'user_id': user_features['user_id']})
    user_stats['follower_fan_ratio'] = user_features['follow_user_num'] / (user_features['fans_user_num'] + 1)
    user_stats['popularity_score'] = user_features['follow_user_num'] + user_features['fans_user_num']
    user_stats['video_id_count'] = interactions.groupby('user_id').agg({'video_id': 'count'}).reset_index()['video_id']
    user_stats['is_user_active'] = (user_features['user_active_degree'] == 'full_active').astype(int)

    ###########
    # Additional user features
    user_features = user_features[['user_id', 'user_active_degree', 'is_lowactive_period', 'is_live_streamer', 'is_video_author']].copy()

    ###########
    # Get video information for user preference extraction
    video_info = video_daily[['video_id', 'video_duration', 'upload_type', 'video_type']].drop_duplicates('video_id')
    video_categories = video_categories.explode('feat').rename(columns={'feat': 'category_id'})
    video_info = pd.merge(video_info, video_categories, on='video_id', how='left')

    # Find user preferred categories and video types based on their interactions
    user_video_interactions = pd.merge(interactions[['video_id', 'user_id', 'watch_ratio']], video_info, on='video_id', how='left')

    # Get preferred categories
    top_categories = (user_video_interactions
        .groupby(['user_id', 'category_id'])
        .agg({
            'watch_ratio': 'sum',
            'video_id': 'count'
        })
        .reset_index()
        .sort_values(['user_id', 'watch_ratio'], ascending=[True, False])
        .groupby('user_id')
        .first()
        .reset_index()
        .rename(columns={'category_id': 'preferred_category'})
    )
    top_categories['preferred_category'] = top_categories['preferred_category'].astype(str).str.replace('[', '').str.replace(']', '')

    # Get preferred upload types
    top_upload_types = (user_video_interactions
        .groupby(['user_id', 'upload_type'])
        .agg({
            'watch_ratio': 'sum',
            'video_id': 'count'
        })
        .reset_index()
        .sort_values(['user_id', 'watch_ratio'], ascending=[True, False])
        .groupby('user_id')
        .first()
        .reset_index()
        .rename(columns={'upload_type': 'preferred_upload_type'})
    )

    # Get preferred video_type
    top_video_types = (user_video_interactions
        .groupby(['user_id', 'video_type'])
        .agg({
            'watch_ratio': 'sum',
            'video_id': 'count'
        })
        .reset_index()
        .sort_values(['user_id', 'watch_ratio'], ascending=[True, False])
        .groupby('user_id')
        .first()
        .reset_index()
        .rename(columns={'video_type': 'preferred_video_type'})
    )
    
    # Get preferred video duration (average of watched videos)
    user_duration_prefs = (user_video_interactions
        .groupby('user_id')
        .agg({
            'video_duration': 'mean'
        })
        .reset_index()
        .rename(columns={'video_duration': 'preferred_duration'})
    )

    # Merge all user preference data
    dfs = [
        user_features,
        user_stats,
        top_categories[['user_id', 'preferred_category']],
        top_video_types[['user_id', 'preferred_video_type']],
        top_upload_types[['user_id', 'preferred_upload_type']],
        user_duration_prefs
    ]
    users = reduce(lambda left, right: pd.merge(left, right, on='user_id', how='left'), dfs)

    # Fill NaN values
    fill_dict = {
        'video_id_count': 0,
        'follower_fan_ratio': 0,
        'popularity_score': 0,
        'preferred_category': -1,
        'preferred_duration': users['preferred_duration'].median() if not users['preferred_duration'].isna().all() else 0
    }
    users.fillna(fill_dict, inplace=True)

    return users

def preprocess_videos(video_features: pd.DataFrame, video_daily: pd.DataFrame, 
                         video_categories: pd.DataFrame) -> pd.DataFrame:
    # Computing generic video stats and information
    video_stats = video_daily.groupby('video_id').agg({
        'play_cnt': 'mean',
        'play_duration': 'mean',
        'like_cnt': 'mean',
        'cancel_like_cnt': 'mean',
        'comment_cnt': 'mean',
        #'reply_comment_cnt': 'mean',
        'share_cnt': 'mean',
        'download_cnt': 'mean',
        'report_cnt': 'mean',
        'follow_cnt': 'mean',
        'cancel_follow_cnt': 'mean',
        # first value for categorical/constant features
        'video_duration': 'first', 
        'video_type': 'first',
        #'video_tag_id': 'first',
        #'video_tag_name': 'first',
        #'author_id': 'first',
        'upload_type': 'first'
    }).reset_index()

    # Engagement metrics
    video_stats['like_play_ratio'] = video_stats['like_cnt'] / (video_stats['play_cnt'] + 1)
    video_stats['comment_play_ratio'] = video_stats['comment_cnt'] / (video_stats['play_cnt'] + 1)
    video_stats['share_play_ratio'] = video_stats['share_cnt'] / (video_stats['play_cnt'] + 1)
    video_stats['follow_play_ratio'] = video_stats['follow_cnt'] / (video_stats['play_cnt'] + 1)
    video_stats['follow_cancel_ratio'] = video_stats['cancel_follow_cnt'] / (video_stats['follow_cnt'] + 1)
    video_stats['like_cancel_ratio'] = video_stats['cancel_like_cnt'] / (video_stats['like_cnt'] + 1)
    video_stats['like_to_comment_ratio'] = video_stats['like_cnt'] / (video_stats['comment_cnt'] + 1)

    # Binarize cat
    video_stats["is_add"] = (video_stats["video_type"] == "AD").astype(int)

    # Keep all relevant columns
    video_stats = video_stats[['video_id', 'like_play_ratio', 'comment_play_ratio', 'share_play_ratio',
                'like_cancel_ratio', 'video_duration', 'is_add', 'like_to_comment_ratio', 
                'upload_type', 'follow_play_ratio', 'follow_cancel_ratio']]

    # Join with video features and categories
    videos = pd.merge(video_stats, video_features[['video_id', 'manual_cover_text', 'caption', 'topic_tag']], 
                     on='video_id', how='left').merge(video_categories, on='video_id', how='left')

    # Process text features if needed
    def parse_tags(tag_str):
        if isinstance(tag_str, str):
            tag_str = tag_str.strip("[]")
            tags = [tag.strip() for tag in tag_str.split(",") if tag.strip()]
            return tags[:10]
        return []

    # Parse tags and concatenate with caption
    videos['parsed_tags'] = videos['topic_tag'].apply(parse_tags)
    videos['caption'] = videos['caption'].fillna('')
    videos['manual_cover_text'] = videos['manual_cover_text'].apply(lambda x: '' if x == 'UNKNOWN' else x)
    videos['tags_caption_cover'] = videos.apply(
        lambda row: ' '.join(row['parsed_tags']) + ' ' + row['caption'] + ' ' + row['manual_cover_text'], axis=1
    )
    videos.drop(columns=['parsed_tags', 'caption', 'topic_tag', 'manual_cover_text'], inplace=True)
    return videos

class ContentPipeline(AbstractPipeline):
    def __init__(self, tfidf_max_features=64):
        super().__init__()
        #  Representations, should be computed once by fit
        self.users = None
        self.videos = None
        # Registering different features
        self.cat_cols = ['upload_type']
        self.binary_cols = ['is_add', 'category_match', 'upload_type_match', 'is_weekend', 'is_user_active',
                            'is_lowactive_period', 'is_live_streamer', 	'is_video_author']
        self.num_cols = ['duration_diff', 'like_play_ratio', 'comment_play_ratio',
            'share_play_ratio', 'follow_play_ratio', 'follow_cancel_ratio',
            'like_cancel_ratio', 'like_to_comment_ratio', 'popularity_score', 'follower_fan_ratio'
        ]
        self.target_col = "engagement"
        # Some variables that should be reused in transform step
        self.cat_values = {}
        self.num_cols_medians = {}
        self.cat_cols_modes = {}
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_video = None
        self.standard_scaler = None
        self.ohe = None

    def _generate_features(self, df: pd.DataFrame) -> None:
        df['category_match'] = df.apply(lambda row: int(row['preferred_category'] in row['feat']), axis=1)
        df['upload_type_match'] = (df['preferred_upload_type'] == df['upload_type']).astype(int)
        #df['video_type_match'] = (df['preferred_video_type'] == df['video_type']).astype(int)
        df['duration_diff'] = abs(df['video_duration'] - df['preferred_duration'])

    def _merge_representations(self, df: pd.DataFrame) -> pd.DataFrame:
        df_merged = df[['video_id', 'user_id', 'engagement']].merge(self.videos, on='video_id', how='left')
        df_merged.dropna(subset=['video_id', 'user_id', 'engagement'], inplace=True)
        return df_merged.merge(self.users, on='user_id', how='left').dropna(subset=['video_id', 'user_id', 'engagement'])

    def _encode_categoricals(self, df: pd.DataFrame, is_fit: bool) -> None:
        if is_fit:
            self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded = self.ohe.fit_transform(df[self.cat_cols])
        else:
            encoded = self.ohe.transform(df[self.cat_cols])

        encoded_df = pd.DataFrame(
            encoded.astype(np.int16),
            columns=self.ohe.get_feature_names_out(self.cat_cols),
            index=df.index
        )

        df.drop(columns=self.cat_cols, inplace=True)
        df[encoded_df.columns] = encoded_df

    def _drop_and_fill_missing(self, df: pd.DataFrame, is_fit: bool) -> None:
        col_to_remove = (set(self.num_cols) | set(self.cat_cols) | set(self.binary_cols) | set(["tags_caption_cover"]) | set([self.target_col])) - set(df.columns)
        self.logger.info(f"Removing columns: {col_to_remove}")
        df.drop(columns=list(col_to_remove), inplace=True, errors='ignore')

        # NUMERIC
        for col in self.num_cols:
            if is_fit:
                self.num_cols_medians[col] = df[col].median()
            df[col].fillna(self.num_cols_medians.get(col, df[col].median()), inplace=True)

        # CATEGORICAL
        for col in self.cat_cols:
            mode = self.cat_cols_modes.get(col) if not is_fit else (df[col].mode()[0] if not df[col].mode().empty else 'unknown')
            if is_fit:
                self.cat_cols_modes[col] = mode
            df[col].fillna(mode, inplace=True)

    def _scale_numerical(self, df: pd.DataFrame, is_fit: bool) -> None:
        if is_fit:
            self.standard_scaler = StandardScaler()
            df[self.num_cols] = self.standard_scaler.fit_transform(df[self.num_cols])
        else:
            df[self.num_cols] = self.standard_scaler.transform(df[self.num_cols])

    def transform(self, interactions: pd.DataFrame, video_features: pd.DataFrame, video_daily: pd.DataFrame, 
                  video_categories: pd.DataFrame, user_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.users is None or self.videos is None or self.standard_scaler is None or self.ohe is None:
            raise ValueError("Preprocessor has not been fitted. Call fit_transform first.")

        interactions_clean = interactions.copy()

        self.logger.info("Processing interaction data...")
        interactions_clean.drop_duplicates(['user_id', 'video_id'], keep='first', inplace=True)
        interactions_clean['engagement'] = interactions_clean['watch_ratio'] * (
            np.log1p(interactions_clean['play_duration']) / np.log1p(interactions_clean['video_duration'])
        )
        #interactions_clean['time'] = pd.to_datetime(interactions_clean['timestamp'], unit='s')
        #interactions_clean['is_weekend'] = (interactions_clean['time'].dt.dayofweek >= 5).astype(int)

        self.logger.info("Combining all features...")
        df_interactions = self._merge_representations(interactions_clean)

        self.logger.info("Generating new features...")
        self._generate_features(df_interactions)

        self.logger.info("Handling missing features (cat / num)...")
        self._drop_and_fill_missing(df_interactions, is_fit=False)

        self.logger.info("Encoding Categorical features...")
        self._encode_categoricals(df_interactions, is_fit=False)

        self.logger.info("Scaling Numerical features...")
        self._scale_numerical(df_interactions, is_fit=False)

        #self.logger.info("TF-IDF on text")
        #tfidf_matrix = self.tfidf_video.fit_transform(df_interactions['tags_caption_cover'])
        #tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[
        #    f'tfidf_feature_{i}' for i in range(tfidf_matrix.shape[1])
        #])
        #df_interactions = pd.concat([df_interactions.reset_index(drop=True), tfidf_df], axis=1)
        #df_interactions.drop(columns=['tags_caption_cover'], inplace=True)

        for col in df_interactions.select_dtypes(include=['int', 'int64', 'int32']).columns:
            df_interactions[col] = df_interactions[col].astype(np.int16)
            for col in df_interactions.select_dtypes(include=['float', 'float64']).columns:
                df_interactions[col] = df_interactions[col].astype(np.float32)
        df_interactions.drop(columns=['preferred_category', 'preferred_video_type', 'preferred_upload_type', 'feat', 'tags_caption_cover', 'user_active_degree'], inplace=True)
        self.logger.info(f"Final dataset shape: {df_interactions.shape}")
        self.logger.info(df_interactions.columns)

        return df_interactions

    def fit(self):
        self.logger.info("Nothing implemented... Call fit_transform instead")
        return self

    def fit_transform(self, interactions: pd.DataFrame, video_features: pd.DataFrame, video_daily: pd.DataFrame, 
                   video_categories: pd.DataFrame, user_features: pd.DataFrame) -> tuple:
        interactions_clean = interactions.drop_duplicates(['user_id', 'video_id'], keep='first')
    
        self.logger.info("Processing interaction data...")
        interactions_clean['engagement'] = interactions_clean['watch_ratio'] * (
            np.log1p(interactions_clean['play_duration']) / np.log1p(interactions_clean['video_duration'])
        )
        interactions_clean['time'] = pd.to_datetime(interactions_clean['timestamp'], unit='s')
        interactions_clean['is_weekend'] = interactions_clean['time'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

        self.logger.info("Processing videos...")
        self.videos = preprocess_videos(video_features, video_daily, video_categories)
        self.logger.info(self.videos.head())

        self.logger.info("Processing users...")
        self.users = preprocess_users(interactions_clean, user_features, video_categories, video_daily)
        self.logger.info(self.users.head())
    
        self.logger.info("Combining all features...")
        df_interactions = self._merge_representations(interactions_clean)

        del interactions_clean

        self.logger.info("Generating new features...")
        self._generate_features(df_interactions)

        self.logger.info("Dropping features...")
        self._drop_and_fill_missing(df_interactions, is_fit=True)

        self.logger.info("Encoding Categoricals...")
        self._encode_categoricals(df_interactions, is_fit=True)

        self.logger.info("Scaling numerical...")
        self._scale_numerical(df_interactions, is_fit=True)

        #self.logger.info("TF-IDF on Text..")
        #self.tfidf_video = TfidfVectorizer(max_features=self.tfidf_max_features)
        #tfidf_matrix = self.tfidf_video.fit_transform(df_interactions['tags_caption_cover'])
        #tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[
        #    f'tfidf_feature_{i}' for i in range(tfidf_matrix.shape[1])
        #])
        #df_interactions = pd.concat([df_interactions.reset_index(drop=True), tfidf_df], axis=1)
        #df_interactions.drop(columns=['tags_caption_cover'], inplace=True)

        for col in df_interactions.select_dtypes(include=['int', 'int64', 'int32']).columns:
            df_interactions[col] = df_interactions[col].astype(np.int16)
        for col in df_interactions.select_dtypes(include=['float', 'float64']).columns:
            df_interactions[col] = df_interactions[col].astype(np.float32)

        df_interactions.drop(columns=['preferred_category', 'preferred_video_type', 'preferred_upload_type', 'feat', 'tags_caption_cover', 'user_active_degree'], inplace=True)        
        self.logger.info(f"Final dataset shape: {df_interactions.shape}")
        self.logger.info(df_interactions.columns)

        return df_interactions
