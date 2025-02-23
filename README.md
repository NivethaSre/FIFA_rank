# FIFA_rank# FIFA Ranking Prediction

## Overview
This project predicts FIFA team rankings using machine learning models. The system analyzes historical data to determine ranking trends and provides insights into team performance.

## Features
- Data preprocessing and feature engineering
- Model training and evaluation using various algorithms
- Visualization of ranking trends
- Prediction of future rankings based on historical patterns
- Comparative analysis of different teams over time

## Dataset
The dataset includes historical FIFA rankings, match results, team statistics, and other relevant football data. Key attributes include:
- Team name
- Match outcomes (win/loss/draw)
- Goals scored/conceded
- FIFA ranking points
- Date of matches
- Confederation details

## Models Used
The following machine learning models were explored:
- **Random Forest**: Used for ranking predictions due to its robustness.
- **Gradient Boosting**: Provides improved accuracy by minimizing prediction errors.
- **ARIMA**: Time series forecasting model for trend analysis.
- **Prophet**: Forecasting model to analyze seasonal trends in rankings.

## Installation
To install dependencies, run:
```sh
pip install -r requirements.txt
```

## Usage
Run the main script to train models and generate predictions:
```sh
python main.py
```

## Dependencies
The project requires the following libraries:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- statsmodels
- prophet

## Evaluation Metrics
Model performance is assessed using:
- **Mean Absolute Error (MAE)**
- **Root Mean Square Error (RMSE)**
- **R-squared Score (R²)**
- **Confusion Matrix (for classification tasks)**

## Usage
To run the app locally, execute the following command: bash streamlit run streamlit_app.py This will start the Streamlit web app, which you can access locally in your browser at http://localhost:8501.

## Input
Input historical match data (e.g., teams, goals scored, match results) to predict FIFA rankings.

## Results
The models successfully predict FIFA rankings with high accuracy. The visualizations provide insights into ranking trends, dominant teams, and performance fluctuations over time.
![image](https://github.com/user-attachments/assets/905563b0-22f5-4b7b-9de3-fca7c2f42edd)

![image](https://github.com/user-attachments/assets/03e7c31a-37eb-46d3-9634-92c3d9d4d54a)

![image](https://github.com/user-attachments/assets/9807b09d-6c41-4925-ab6f-27729c598dab)

![image](https://github.com/user-attachments/assets/9a7b6df1-beec-4686-ab9b-859de8b42968)

![image](https://github.com/user-attachments/assets/d8926e8d-60f4-4835-a436-eace4ab19616)

![image](https://github.com/user-attachments/assets/1f5a78b9-5857-476e-9324-62557bca83d1)

![image](https://github.com/user-attachments/assets/d88b214e-59c0-4d68-84c8-803c2d104efb)



## Future Improvements
- Integration of additional features like player performance and team formations.
- Enhanced forecasting using deep learning models.
- Automated model selection for optimal performance.
- Real-time ranking updates using API integration.

## Technologies
Programming Language: Python Libraries:Streamlit, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn Deployment: Streamlit Cloud

## License

MIT License




