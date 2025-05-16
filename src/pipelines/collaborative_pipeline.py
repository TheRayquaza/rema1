import pandas as pd
from src.pipelines.base import AbstractPipeline

class CollaborativePipeline(AbstractPipeline):
    def transform(self, data: pd.DataFrame):
        data.drop_duplicates(['user_id', 'video_id'], keep='first', inplace=True)
        data.drop(columns=['play_duration', 'video_duration', 'time', 'date', 'timestamp'])
        return data.pivot(index='user_id', columns='video_id', values='watch_ratio')

    def fit(self, data: pd.DataFrame):
        return self
