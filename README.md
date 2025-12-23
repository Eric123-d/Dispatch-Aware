# Dispatch-Aware
A simple electricity load forecasting project using XGBoost.

Project Overview
This project predicts future electricity demand (load forecasting) from historical data.
Normal forecasting methods only try to be as accurate as possible. However, in real power grids, the biggest fear is under-forecasting (predicting too low), which can cause power shortages.

To solve this, I added a small trick: slightly increase the forecast by a conservative bias (+3%). Result: prediction accuracy drops only a little, but the risk of power shortage is greatly reduced (from 112 events to 44 events in the test period).
This makes the method much more practical for real-world grid operation — better to generate a bit more power than to risk blackouts.

Files
	•	dispatch_comparison.py: Run this to see the comparison plot (standard vs dispatch-aware) and key numbers (RMSE, shortage events, etc.)
	•	sensitivity_analysis.py: Run this to see a table and curve showing the effect of different bias levels (1%, 2%, 3%, etc.)
	•	load_data.csv: Hourly electricity load data (public dataset, already included)
	•	requirements.txt: List of required Python packages

How to Run (Very Easy)
	1.Download or copy the entire folder to your computer.
	2.Open Terminal (Mac: Command + Space → type “Terminal”; Windows: search “cmd”).
	3.Go to the folder (example if it’s on Desktop): cd Desktop/Your-Folder-Name	
	4.Install the required packages (only need to do this once): pip install -r requirements.txt

	5.Run the scripts: python dispatch_comparison.py or python sensitivity_analysis.py
  
Data Source
The data is public hourly load data from American Electric Power (AEP), downloaded from Kaggle.

Author: Eric

A project completed by a freshman student through self-study. Feedback welcome!
If you find it useful, feel free to give it a Star ⭐ !
