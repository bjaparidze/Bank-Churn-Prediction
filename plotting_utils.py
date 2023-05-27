import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import learning_curve


class BarCharts:

    @staticmethod
    def plot_bar_chart(df: pd.DataFrame, column_name: str):
        count = df[column_name].value_counts()
        count.plot(kind='bar')

        plt.title(f'Distribution of {column_name}')
        plt.xlabel(f'{column_name}')
        plt.ylabel('Count')

        plt.show()

    @staticmethod
    def plot_bar_charts_for_each_categorical_var(df: pd.DataFrame):
        # filter out non-binary columns
        binary_cols = [col for col in df.columns if set(df[col].unique()) == set([0, 1])]
        
        # select columns of categorical type
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        categorical_cols += binary_cols

        for column in categorical_cols:
            count = df[column].value_counts()
            count.plot(kind='bar')
            plt.title(f'Distribution of {column}')
            plt.xlabel(f'{column}')
            plt.ylabel('Count')
            plt.show()

class BoxPlots:

    @staticmethod
    def plot_for_each_numeric_var(df: pd.DataFrame):
        numeric_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and df[col].nunique() > 2]
        for column in numeric_columns:
            plt.figure()
            sns.boxplot(x=column, data=df)
            plt.show()

class Histograms:

    @staticmethod
    def plot_for_each_numeric_var(df: pd.DataFrame):
        numeric_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and df[col].nunique() > 2]
        for column in numeric_columns:
            plt.figure()
            df[column].hist()
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()

class LearningCurve:

    @staticmethod
    def plot_learning_curves(estimator, X, y, train_sizes, cv):
        train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
            estimator, X, y, train_sizes=train_sizes, cv=cv, scoring='accuracy', return_times=True
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Testing')
        plt.fill_between(
            train_sizes, train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std, alpha=0.2, color='r'
        )
        plt.fill_between(
            train_sizes, test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std, alpha=0.2, color='g'
        )
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
